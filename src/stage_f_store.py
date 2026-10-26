"""The Stage-F store, provenance and the fail-closed completion barrier (§H, §I).

Two classes, deliberately with **different APIs** rather than one class with a
mode flag:

* `FormalStageFStore` writes the one authoritative store. Its root is derived
  from the repository and is **not a parameter** — there is no `--out`, no
  `--root`, no environment override and no keyword by which a caller could
  redirect an authoritative run (§I.2). A non-empty root fails closed before
  anything is written: the store is never overwritten, merged into, resumed or
  cleaned, and the root is checked for redirection in its **lexical** form
  before any canonical comparison, so a symbolic link planted at the
  authoritative root is refused rather than silently followed. Formal authority
  is a **capability bound to one object**, not a flag and not a token: the
  direct constructor is refused outright, the factory takes no object from its
  caller, and there is no authorisation container anywhere — no mapping, list,
  set or binding record into which an independently allocated store could be
  inserted. One immutable record names the one authorised instance, the instance
  carries the same record, and both are checked for identity. Completion
  re-derives all of it — the absence of insertable state, the class, the run
  kind, the lexical root, the symlink question and the canonical root — at the
  moment the barrier runs, so neither direct construction, nor a later attribute
  assignment, nor a subclass, nor state copied from a genuine store confers it.
* `SmokeStageFStore` writes anywhere **except** the authoritative root, and has
  no `complete_run` at all, so no argument value and no call sequence can make a
  smoke run authoritative. Its writer refuses `COMPLETION.json` outright.

Every registered artifact passes **one** write-time barrier before its bytes
reach disk — exact registered schema, unique registered logical key, genuinely
numeric scientific fields, coherent status / reason / degeneracy semantics,
exact cardinality — and is written through an atomic temp-file rename whose
bytes are fsynced before the rename and whose directory entries are fsynced
after it, up to the store root. There is no second write path, so no module can
persist a table that skipped any of this, and an artifact counts as validated
only once every one of those checks has passed.

`COMPLETION.json` is terminal: written last, atomically, after the store has
been re-read from disk, its ledger parsed strictly and re-verified line by line,
its digests recomputed, and the registered / written / validated / ledgered
artifact sets shown to be **identical**. A store without it is incomplete by
definition and may never be read as a result.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import FunctionType, ModuleType
from typing import NamedTuple

import pandas as pd

import stage_f_registry as reg

ROOT = reg.ROOT


class StageFStoreError(AssertionError):
    """A structural violation of the Stage-F store contract."""


class StaleStoreRootError(StageFStoreError):
    """The authoritative root already exists and is non-empty (§I.2)."""


class CompletionBarrierError(StageFStoreError):
    """The seven-step barrier failed. The store never becomes authoritative."""


class RootRedirectionError(StageFStoreError):
    """The authoritative root is reached through a symbolic link (§I.2)."""


def authoritative_root(root: Path = ROOT) -> Path:
    """The authoritative root **as written** — from the repository, lexical.

    Deliberately *not* resolved. `resolve()` follows symbolic links, so a
    resolved root answers "where does this path end up", which is the wrong
    question for a root of trust: a link planted at the authoritative root would
    resolve to its target and then compare equal to itself, and the store would
    write an authoritative run into whatever the link pointed at. The lexical
    form is what the protocol froze, and it is what the authority barrier checks
    first; canonical comparison happens only after
    `assert_root_not_redirected` has ruled a redirection out.
    """
    return Path(root) / reg.STORE_ROOT


def assert_root_not_redirected(lexical: Path) -> Path:
    """Refuse a redirected authoritative root, then return its canonical form.

    Three checks, in this order (§I.2):

    1. the root itself is not a symbolic link — including a *dangling* one, which
       `exists()` would miss but `is_symlink()` catches;
    2. no path component redirects it: the fully resolved path must equal the
       purely lexical normalisation of the same path, so a link anywhere along
       `results/` is caught as well as one at the leaf;
    3. only then is the canonical form computed and returned, and only then may
       any caller compare resolved paths at all.

    The repository's own prefix cannot smuggle a redirection past this: `ROOT` is
    resolved at import, so `ROOT / results / stage_f` has a link-free prefix by
    construction and any link introduced under it makes the two forms differ.
    """
    lexical = Path(lexical)
    if lexical.is_symlink():
        raise RootRedirectionError(
            f"{lexical} is a symbolic link; the authoritative Stage-F root is a "
            "real directory inside the repository. A run is refused rather than "
            "written through a redirection (§I.2)")
    resolved = lexical.resolve()
    normalised = Path(os.path.normpath(os.path.abspath(str(lexical))))
    if resolved != normalised:
        raise RootRedirectionError(
            f"{lexical} resolves to {resolved} but names {normalised}; a "
            "component of the authoritative root is a symbolic link that "
            "redirects it away from the repository path the protocol froze "
            "(§I.2)")
    return resolved


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory_chain(leaf: Path, stop: Path | None = None) -> int:
    """fsync `leaf` and every ancestor up to and including `stop`.

    A nested artifact needs more than its own directory synced: the entry that
    names `rq4/` inside the store root is itself a directory write, and is not
    durable until the **root** is synced too. Syncing only the leaf would leave
    a freshly created subdirectory recoverable-but-empty after a crash, which is
    exactly the state the barrier's reread is meant to exclude.
    """
    synced = 0
    current = leaf.resolve()
    stop_at = None if stop is None else stop.resolve()
    while True:
        _fsync_directory(current)
        synced += 1
        if stop_at is None or current == stop_at or current == current.parent:
            break
        if stop_at not in current.parents:
            break
        current = current.parent
    return synced


def _atomic_write(target: Path, blob: bytes, fsync_up_to: Path | None = None) -> str:
    """Write via a sibling temp file, fsync, then rename — never in place.

    The bytes are on stable storage before the rename, the rename is atomic, and
    the directory entries that name the file are synced after it (up to
    `fsync_up_to`, normally the store root). Nothing here reads, repairs or
    replaces a pre-existing result: the temp file is one this call just created.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    _fsync_directory_chain(target.parent, fsync_up_to)
    return reg.sha256_file(target)


# ---------------------------------------------------------------------------
# write-time validation (§H.1-§H.4, §G) — one path, no module bypasses it
# ---------------------------------------------------------------------------
class SchemaViolationError(StageFStoreError):
    """A table does not present exactly its registered columns."""


class DuplicateLogicalKeyError(StageFStoreError):
    """A table repeats a registered logical key."""


class StatusSemanticsError(StageFStoreError):
    """A row's status / reason / degeneracy fields are not internally coherent."""


def _validate_schema(relpath: str, frame: pd.DataFrame) -> tuple:
    """Exactly the registered columns, in exactly the registered order."""
    expected = reg.table_columns(relpath)
    if expected is None:
        return ()
    observed = tuple(frame.columns)
    missing = [c for c in expected if c not in observed]
    extra = [c for c in observed if c not in expected]
    if missing or extra:
        raise SchemaViolationError(
            f"{relpath}: {len(missing)} registered columns missing {missing}, "
            f"{len(extra)} unregistered columns present {extra}; the frozen "
            "registry authorises no projection, so the table is rejected "
            "rather than trimmed (§H.1)")
    if observed != tuple(expected):
        raise SchemaViolationError(
            f"{relpath}: the column order {observed[:6]}... is not the frozen "
            f"order {tuple(expected)[:6]}...")
    return tuple(expected)


def _validate_logical_key(relpath: str, frame: pd.DataFrame) -> tuple:
    """The registered logical key identifies at most one row."""
    key = reg.logical_key(relpath)
    if not key:
        return ()
    absent = [c for c in key if c not in frame.columns]
    if absent:
        raise SchemaViolationError(
            f"{relpath}: the registered logical key {key} needs columns "
            f"{absent}, which the table does not carry")
    duplicated = frame.duplicated(subset=list(key), keep=False)
    if bool(duplicated.any()):
        offenders = (frame.loc[duplicated, list(key)]
                     .drop_duplicates().head(3).to_dict("records"))
        raise DuplicateLogicalKeyError(
            f"{relpath}: {int(duplicated.sum())} rows repeat the registered "
            f"logical key {key}, e.g. {offenders}; duplicate keys are never "
            "permitted (§H.1)")
    return key


def _cell(value):
    """One CSV cell as a plain Python value, with pandas' NA collapsed to None."""
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):          # a list, dict or other container
        return value
    return None if missing else value


class NumericSemanticsError(StageFStoreError):
    """A registered scientific numeric field did not arrive as a number."""


def _is_number(value) -> bool:
    """A genuine real number — never a bool, and never a numeric-looking string.

    `"0.5"` is not a coverage, `"nan"` is not a missing value and `"abc"` is not
    a balanced accuracy. A scientific quantity that reached persistence as text
    did not come from the estimator that was supposed to produce it, so the
    store refuses it rather than coercing it into a float (§H.1).
    """
    import numbers

    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _validate_numeric(relpath: str, frame: pd.DataFrame) -> int:
    """Every registered scientific numeric field holds a **finite** number.

    Two independent conditions, both per column and both registry-driven:

    * the value is a genuine real number — not a numeric-looking string, not a
      bool, not an arbitrary object;
    * the value is finite, unless this exact artifact and this exact column are
      registered in `reg.NONFINITE_PERMITTED_FIELDS`. The registered exception is
      the all-inclusive conformal threshold `t = +inf` (§D.10). There is no
      global infinity exemption, so a `+inf` balanced accuracy, a `nan` coverage
      or a `-inf` risk excess is refused wherever it appears.

    Screened column-wise rather than cell by cell: a column pandas already types
    as numeric cannot contain a string at all, so only object columns — the ones
    that mix real values with the missing representation — are inspected
    element by element.
    """
    import numpy as np

    fields = [f for f in reg.numeric_fields(relpath) if f in frame.columns]
    checked = 0
    for field in fields:
        column = frame[field]
        if pd.api.types.is_bool_dtype(column):
            raise NumericSemanticsError(
                f"{relpath}: {field} arrived as a boolean column; it is a "
                "registered scientific numeric field")
        if pd.api.types.is_numeric_dtype(column):
            # pandas could not have typed it numeric with a string in it
            present = column[~column.isna()]
            checked += int(len(present))
        else:
            present = column[~column.isna()]
            present = present[~present.isin(reg.MISSING_STRING_VALUES)]
            offending = present[~present.map(_is_number)]
            if len(offending):
                position, value = offending.index[0], offending.iloc[0]
                raise NumericSemanticsError(
                    f"{relpath}: row {position} carries {field}={value!r} "
                    f"({type(value).__name__}); {field} is a registered "
                    "scientific numeric field and is never coerced from a "
                    f"string (reg.NUMERIC_STRINGS_COERCED="
                    f"{reg.NUMERIC_STRINGS_COERCED})")
            checked += int(len(present))
        if not reg.numeric_field_requires_finite(relpath, field) or not len(present):
            continue
        values = np.asarray(present.tolist(), dtype=float)
        bad = np.flatnonzero(~np.isfinite(values))
        if bad.size:
            position = present.index[int(bad[0])]
            raise NumericSemanticsError(
                f"{relpath}: row {position} carries a non-finite "
                f"{field}={present.iloc[int(bad[0])]!r}; {field} is a "
                "registered finite scientific field. The registered non-finite "
                f"values here are {reg.NONFINITE_PERMITTED_FIELDS.get(relpath, ())} "
                f"(reg.NONFINITE_PERMITTED_ANYWHERE="
                f"{reg.NONFINITE_PERMITTED_ANYWHERE}); an unavailable quantity "
                f"uses {reg.MISSING_REPRESENTATIONS}, never an infinity")
    return checked


class ClosedDomainError(StageFStoreError):
    """A registered closed-domain categorical field left its frozen vocabulary."""


def _validate_closed_domains(relpath: str, frame: pd.DataFrame) -> int:
    """Every registered closed-domain field holds a **registered** member (§H.1).

    The companion rule to `_validate_numeric`, for the columns that report a
    *state* rather than a quantity. `svm/per_site_bacc.csv:selected_pca` is the
    one such column today: the inner CV chooses from the frozen §F grid
    `{no-PCA, 50, 100, 200}`, and `no-PCA` is a completed selection whose only
    faithful serialisation is that literal. Numeric validation is the wrong rule
    there — it refused the producer's own legal output — but the answer is a
    *narrower* rule, not a weaker one, so membership is enforced by value and by
    kind:

    * `"50"` is refused, exactly as a stringified balanced accuracy is refused
      in a numeric field: text that spells a dimension did not come from the
      selector (`reg.CLOSED_DOMAIN_STRINGS_COERCED` is False);
    * `0` and `-1` are refused — they are not members of the grid, and neither
      is a stand-in for "no PCA" (§G.4: zero is never a missing representation);
    * a bool is refused, here as everywhere;
    * the missing representation is admitted only where the row's own status
      says no selection was reached. An `ESTIMABLE` fold with no `selected_pca`
      would report a `bacc` from an unnamed model, so it is refused
      (`reg.CLOSED_DOMAIN_REQUIRED_WHEN_ESTIMABLE`).
    """
    fields = reg.closed_domain_fields(relpath)
    fields = {f: d for f, d in fields.items() if f in frame.columns}
    if not fields or frame.empty:
        return 0
    statuses = (frame["status"].tolist() if "status" in frame.columns
                else [None] * len(frame))
    checked = 0
    for field, domain in sorted(fields.items()):
        column = frame[field]
        if pd.api.types.is_bool_dtype(column):
            raise ClosedDomainError(
                f"{relpath}: {field} arrived as a boolean column; its "
                f"registered vocabulary is {domain}")
        for position, value in enumerate(column.tolist()):
            cell = _cell(value)
            if reg.is_missing(cell):
                status = _cell(statuses[position])
                if reg.CLOSED_DOMAIN_REQUIRED_WHEN_ESTIMABLE \
                        and str(status) == reg.ESTIMABLE:
                    raise ClosedDomainError(
                        f"{relpath}: row {position} is {reg.ESTIMABLE} and "
                        f"leaves {field} missing; a row that reports a result "
                        "reports the state it selected, and the registered "
                        f"vocabulary {domain} contains no missing member. "
                        "'no-PCA' is a selection, not an absence")
                continue
            checked += 1
            if not reg.in_closed_domain(relpath, field, cell):
                raise ClosedDomainError(
                    f"{relpath}: row {position} carries {field}={cell!r} "
                    f"({type(cell).__name__}); the registered closed domain is "
                    f"{domain} and a value outside it is never coerced into it "
                    f"(reg.CLOSED_DOMAIN_STRINGS_COERCED="
                    f"{reg.CLOSED_DOMAIN_STRINGS_COERCED})")
    return checked


def _is_count(value) -> bool:
    """A genuine integral count — never a bool, never fractional, never text."""
    if not _is_number(value):
        return False
    try:
        return float(value).is_integer()
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_site_identifiers(relpath: str, frame: pd.DataFrame) -> int:
    """A worst-site identifier names one of the 23 frozen eligible sites."""
    fields = [f for f in reg.SITE_IDENTIFIER_FIELDS.get(relpath, ())
              if f in frame.columns]
    if not fields:
        return 0
    import contracts

    eligible = set(contracts.FROZEN_SITES)
    checked = 0
    for field in fields:
        for position, value in enumerate(frame[field].tolist()):
            cell = _cell(value)
            if reg.is_missing(cell):
                continue
            checked += 1
            if str(cell) not in eligible:
                raise StatusSemanticsError(
                    f"{relpath}: row {position} names {field}={cell!r}, which "
                    f"is not one of the {len(eligible)} frozen eligible sites; "
                    "a worst-site identifier is a site, never a value carried "
                    "over from another column")
    return checked


def _validate_interval_row(relpath: str, row: dict, status, reason) -> None:
    """§D.12 — a valid interval is present, ordered, and above the validity floor.

    Three states, kept apart exactly as §D.12.3 keeps them:

    * a normal `ESTIMABLE` row must carry a real point estimate **and** two real,
      finite, ordered endpoints, on a replicate count that actually meets the
      floor;
    * `CI_INSUFFICIENT_VALID_REPLICATES` invalidates the **interval alone** — the
      point estimate survives and the row stays `ESTIMABLE`, both endpoints go
      missing, and the replicate count must really be below the floor being
      invoked. The floor may never be used to convert the whole scientific
      quantity to `NOT_ESTIMABLE`: that would silently withdraw a point estimate
      the protocol says is still estimable;
    * `NOT_ESTIMABLE` leaves the point estimate missing with the endpoints, and
      may not borrow the interval-only reason.

    Both replicate counts are checked in every state: they are *counts*, so a
    fractional or boolean value is rejected outright, the total is the frozen
    `B`, and the valid count can never exceed it.
    """
    low, high = (_cell(row[f]) for f in reg.INTERVAL_ENDPOINTS)
    valid_replicates = _cell(row[reg.INTERVAL_VALID_FIELD])
    total = _cell(row[reg.INTERVAL_TOTAL_FIELD])
    floor_invoked = (not reg.is_missing(reason)
                     and str(reason) == reg.CI_VALIDITY_FLOOR_REASON)

    # -- the two replicate counts, in every state ---------------------------
    for name, value in ((reg.INTERVAL_TOTAL_FIELD, total),
                        (reg.INTERVAL_VALID_FIELD, valid_replicates)):
        if reg.is_missing(value):
            raise StatusSemanticsError(
                f"{relpath}: {name} is missing; the replicate counts describe "
                "how the interval was computed and are reported in every state, "
                "including NOT_ESTIMABLE (§D.12)")
        if not _is_count(value):
            raise StatusSemanticsError(
                f"{relpath}: {name}={value!r} ({type(value).__name__}) is not a "
                "genuine integral count; a bootstrap replicate count is a whole "
                "number of replicates, never a fraction, a bool or a string")
    if int(total) != reg.SITE_BOOTSTRAP_B:
        raise StatusSemanticsError(
            f"{relpath}: {reg.INTERVAL_TOTAL_FIELD}={total!r}, the frozen B is "
            f"{reg.SITE_BOOTSTRAP_B}")
    if int(valid_replicates) < 0 or int(valid_replicates) > int(total):
        raise StatusSemanticsError(
            f"{relpath}: {int(valid_replicates)} valid replicates of "
            f"{int(total)}; the valid count is a subset of the total")

    # -- the interval itself, state by state --------------------------------
    if floor_invoked and status != reg.ESTIMABLE:
        raise StatusSemanticsError(
            f"{relpath}: {reg.CI_VALIDITY_FLOOR_REASON} is invoked on a "
            f"{status} row. The validity floor invalidates the **interval**; "
            "the point estimate remains estimable and the row remains "
            f"{reg.ESTIMABLE} (§D.12.3)")
    if status == reg.ESTIMABLE and not floor_invoked:
        for name, value in zip(reg.INTERVAL_ENDPOINTS, (low, high)):
            if reg.is_missing(value):
                raise StatusSemanticsError(
                    f"{relpath}: a valid interval row leaves {name} missing; "
                    "an interval with one endpoint is not an interval, and the "
                    f"only registered way to withhold it is "
                    f"{reg.CI_VALIDITY_FLOOR_REASON} (§D.12.3)")
            if not _is_number(value) or not _finite(value):
                raise StatusSemanticsError(
                    f"{relpath}: {name}={value!r} is not a finite number")
        if float(low) > float(high):
            raise StatusSemanticsError(
                f"{relpath}: ci_low={low!r} exceeds ci_high={high!r}")
        if int(valid_replicates) < reg.MIN_VALID_REPLICATES:
            raise StatusSemanticsError(
                f"{relpath}: a valid interval reports "
                f"{valid_replicates!r} valid replicates, below the frozen "
                f"validity floor {reg.MIN_VALID_REPLICATES}, without invoking "
                f"{reg.CI_VALIDITY_FLOOR_REASON}")
    elif floor_invoked:
        # the endpoints are already required missing by the caller's
        # `estimable_reasons` branch; what remains is that the floor was real
        if int(valid_replicates) >= reg.MIN_VALID_REPLICATES:
            raise StatusSemanticsError(
                f"{relpath}: {reg.CI_VALIDITY_FLOOR_REASON} is invoked with "
                f"{valid_replicates!r} valid replicates, which meets the frozen "
                f"floor {reg.MIN_VALID_REPLICATES}")
    n_sites, n_estimable = (_cell(row["n_sites"]), _cell(row["n_estimable_sites"]))
    for name, value in (("n_sites", n_sites), ("n_estimable_sites", n_estimable)):
        if not reg.is_missing(value) and not _is_count(value):
            raise StatusSemanticsError(
                f"{relpath}: {name}={value!r} is not a genuine integral count")
    if not reg.is_missing(n_sites) and not reg.is_missing(n_estimable) \
            and int(n_estimable) > int(n_sites):
        raise StatusSemanticsError(
            f"{relpath}: {n_estimable} estimable sites of {n_sites}")


def _key_tuple(row: dict, key) -> tuple:
    """One row's logical key, normalised so `None`/NaN compare equal."""
    out = []
    for field in key:
        value = _cell(row[field])
        if reg.is_missing(value):
            out.append(None)
        elif isinstance(value, float) and float(value).is_integer():
            out.append(int(value))
        elif _is_number(value):
            out.append(float(value) if isinstance(value, float) else value)
        else:
            out.append(str(value))
    return tuple(out)


def _frozen_key_tuple(key) -> tuple:
    """The same normalisation applied to a frozen registry enumeration."""
    out = []
    for value in key:
        if value is None:
            out.append(None)
        elif isinstance(value, bool):
            out.append(value)
        elif isinstance(value, float) and float(value).is_integer():
            out.append(int(value))
        elif isinstance(value, (int, float)):
            out.append(value)
        else:
            out.append(str(value))
    return tuple(out)


def _validate_structural_cells(relpath: str, frame: pd.DataFrame) -> int:
    """Which cells may exist at all — the §G.2 reading of `SITE_NOT_ELIGIBLE_AT_N`.

    §G.2 defines that code as "the site is not in the frozen per-`N` eligible
    set, **so the cell does not exist**", and §H.2's cardinalities enumerate the
    eligible cells alone. A cell that does not exist is therefore represented by
    **absence**, never by a row saying so, and this is where absence is made to
    mean something: a table of the registered length assembled from the wrong
    cells is refused before it reaches disk.

    Three shapes, all read from the frozen registry:

    * `EXACT_KEY_SET_TABLES` — the 20 registered conformal cells and the 80
      registered interval slots are enumerated in `stage_f_registry`, so their
      key sets are compared for **exact equality**, not for size;
    * `RQ4_SITE_CELL_TABLES` — the site axis is not enumerable from frozen
      constants (the per-`N` eligible identities live in the frozen split
      registry, a run-time input), so its **shape** is enforced instead: the
      registered methods and `alpha` grid, `n_labels` null exactly for the two
      zero-label methods, every site one of the 23 frozen eligible sites,
      `expected_sites_in_cell` distinct sites per cell, one site set per `N`
      across `alpha`, and eligibility monotone in `N` (§D.6);
    * the two SVM tables — one outer fold per frozen eligible site, and the full
      inner `(pca, C)` grid at each of them (§H.3).
    """
    import contracts

    key = reg.logical_key(relpath)
    governed = (relpath in reg.EXACT_KEY_SET_TABLES
                or relpath in reg.RQ4_SITE_CELL_TABLES
                or relpath in (reg.SVM_PER_SITE_TABLE,
                               reg.SVM_FOLD_SELECTION_TABLE))
    if not key or frame.empty or not governed:
        return 0
    rows = frame.to_dict("records")
    observed = [_key_tuple(row, key) for row in rows]

    frozen = reg.EXACT_KEY_SET_TABLES.get(relpath)
    if frozen is not None:
        expected = {_frozen_key_tuple(k) for k in frozen}
        seen = set(observed)
        if seen != expected:
            missing = sorted(expected - seen, key=repr)[:3]
            extra = sorted(seen - expected, key=repr)[:3]
            raise StatusSemanticsError(
                f"{relpath}: the persisted logical keys are not the frozen "
                f"registry of {len(expected)} cells "
                f"({len(expected - seen)} missing, e.g. {missing}; "
                f"{len(seen - expected)} unregistered, e.g. {extra}). §H.2 "
                "enumerates the cells that exist, and a cell outside it is "
                "represented by absence, never by a row (§G.2)")
        return len(observed)

    sites = set(contracts.FROZEN_SITES)
    if relpath == reg.SVM_PER_SITE_TABLE:
        seen = {t[key.index(reg.SVM_SITE_FIELD)] for t in observed}
        if seen != sites:
            raise StatusSemanticsError(
                f"{relpath}: the outer folds are the {len(sites)} frozen "
                f"eligible sites; {len(sites - seen)} are missing and "
                f"{len(seen - sites)} are not eligible sites (§H.3)")
        return len(observed)
    if relpath == reg.SVM_FOLD_SELECTION_TABLE:
        grid = {(s, ("no-PCA" if p is None else int(p)), float(c))
                for s in sites for p in reg.SVM_PCA_CANDIDATES
                for c in reg.SVM_C_GRID}
        seen = set()
        for position, row in enumerate(rows):
            pca, penalty = _cell(row["pca"]), _cell(row["C"])
            if reg.is_missing(pca) or not _is_number(penalty):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} carries pca={pca!r}, C="
                    f"{penalty!r}; `stage_f_svm.run_outer_fold` writes the "
                    "literal 'no-PCA' for the uncompressed candidate and an "
                    "integer dimension otherwise — a key coordinate is never "
                    "the missing representation")
            pca = str(pca) if isinstance(pca, str) else int(pca)
            seen.add((str(_cell(row[reg.SVM_SITE_FIELD])), pca, float(penalty)))
        if seen != grid:
            raise StatusSemanticsError(
                f"{relpath}: the inner selection grid is {len(grid)} "
                f"(site, pca, C) combinations; {len(grid - seen)} are missing "
                f"and {len(seen - grid)} are off the frozen grid (§H.3)")
        return len(observed)

    per_repeat = relpath == "rq4/per_repeat_target_local.csv"
    site_sets, cells = {}, {}
    for position, row in enumerate(rows):
        method = (reg.RQ4_PER_REPEAT_METHOD if per_repeat
                  else str(_cell(row["method"])))
        if method not in reg.CONFORMAL_METHODS:
            raise StatusSemanticsError(
                f"{relpath}: row {position} names method {method!r}; the "
                f"registered §D.2 methods are {reg.CONFORMAL_METHODS}")
        alpha = _cell(row["alpha"])
        if not _is_number(alpha) or float(alpha) not in reg.ALPHA_GRID:
            raise StatusSemanticsError(
                f"{relpath}: row {position} names alpha={alpha!r}; the frozen "
                f"§D.5 grid is {reg.ALPHA_GRID}")
        n_labels = _cell(row["n_labels"])
        n_labels = None if reg.is_missing(n_labels) else int(n_labels)
        try:
            expected_sites = reg.expected_sites_in_cell(method, n_labels)
        except reg.StageFRegistryError as exc:
            raise StatusSemanticsError(f"{relpath}: row {position}: {exc}") from exc
        site = _cell(row[reg.RQ4_SITE_FIELD])
        if reg.is_missing(site) or str(site) not in sites:
            raise StatusSemanticsError(
                f"{relpath}: row {position} names held_out_site={site!r}, "
                f"which is not one of the {len(sites)} frozen eligible sites; "
                "an ineligible site has no cell (§G.2)")
        if per_repeat:
            repeat = _cell(row[reg.RQ4_REPEAT_FIELD])
            if not _is_count(repeat) or not 0 <= int(repeat) < reg.R_SPLIT:
                raise StatusSemanticsError(
                    f"{relpath}: row {position} names repeat={repeat!r}; the "
                    f"frozen split registry holds {reg.R_SPLIT} repeats")
        cell = (method, float(alpha), n_labels)
        cells[cell] = expected_sites
        group = cell + ((int(_cell(row[reg.RQ4_REPEAT_FIELD])),) if per_repeat
                        else ())
        site_sets.setdefault(group, set()).add(str(site))

    registered = {(m, float(a), n) for m, a, n in reg.REGISTERED_CELLS
                  if not per_repeat or m == reg.RQ4_PER_REPEAT_METHOD}
    if set(cells) != registered:
        raise StatusSemanticsError(
            f"{relpath}: the persisted cells are not the §H.2 registry "
            f"({len(registered - set(cells))} missing, "
            f"{len(set(cells) - registered)} unregistered)")
    for group, seen in sorted(site_sets.items(), key=repr):
        expected_sites = cells[group[:3]]
        if len(seen) != expected_sites:
            raise StatusSemanticsError(
                f"{relpath}: cell {group} holds {len(seen)} distinct sites; "
                f"the frozen eligible count at N={group[2]} is "
                f"{expected_sites} (stage_f_registry.SITES_PER_N). An "
                "ineligible site x N cell is absent, not a row (§G.2)")

    by_n = {}
    for group, seen in site_sets.items():
        method, _alpha, n_labels = group[:3]
        previous = by_n.setdefault((method, n_labels), seen)
        if previous != seen:
            raise StatusSemanticsError(
                f"{relpath}: {method} at N={n_labels} evaluates a different "
                f"site set at different alpha values "
                f"({sorted(previous ^ seen)}); alpha does not change which "
                "sites are eligible (§D.6)")
    if reg.ELIGIBILITY_IS_MONOTONE_IN_N:
        local = sorted((n, s) for (m, n), s in by_n.items()
                       if m == reg.METHOD_TARGET_LOCAL and n is not None)
        for (small, wide), (large, narrow) in zip(local, local[1:]):
            if not narrow <= wide:
                raise StatusSemanticsError(
                    f"{relpath}: sites {sorted(narrow - wide)} are evaluated at "
                    f"N={large} but not at N={small}; a site that can supply "
                    f"{large} calibration labels can supply {small}, so the "
                    "eligible sets are nested (§D.6)")
    return len(observed)


class SplitIdentityError(StageFStoreError):
    """A persisted eligible-site identity is not the verified registry's."""


def _verified_identities(binding, where: str):
    """The run's own verified split-registry binding, or a hard refusal.

    A store that never obtained one cannot persist a single site x `N` identity.
    Verification that happened somewhere else, earlier, or for some other
    registry authorises nothing here: the binding is an attribute of *this* run,
    carried from the `verify_split_registry` call that produced it.
    """
    import stage_f_inputs as inputs

    if binding is None:
        raise SplitIdentityError(
            f"{where}: this store holds no verified split-registry binding, so "
            "which sites are eligible at each N is unknown to it. The eligible "
            "identities are run-time data derived from the frozen production "
            "split registry by stage_f_inputs.derive_verified_split_identities, "
            "and an artifact keyed on them may not be written without it (§D.9)")
    if type(binding) is not inputs.VerifiedSplitIdentities:
        raise SplitIdentityError(
            f"{where}: the split-registry binding is a "
            f"{type(binding).__name__}; only the immutable record that "
            "stage_f_inputs.derive_verified_split_identities returns after a "
            "successful verify_split_registry is a binding")
    return binding


def _validate_split_identity(binding, relpath: str, frame: pd.DataFrame) -> dict:
    """The persisted site x `N` identities **are** the verified registry's.

    §D.6 defines eligibility at `N` as "the site can supply `N` target
    calibration labels", and the frozen production split registry is the only
    place that fact lives. `SITES_PER_N` is its arithmetic shadow, not a second
    authority: the counts, the 23-site universe, the nesting in `N` and the
    agreement between the two RQ4 tables are all preserved when one eligible
    site is swapped for another, consistently, everywhere. So the identities
    themselves are compared, against the mapping this run verified.

    The comparison is exact and in both directions — no missing site, no extra
    site, no substitution, no alias, no duplicate — and it narrows the verified
    mapping by the artifact's own frozen restriction (SENS-4's 11 common sites,
    §E.5) rather than admitting anything the registry did not.
    """
    spec = reg.split_identity_spec(relpath)
    if spec is None or frame.empty:
        return {}
    binding = _verified_identities(binding, relpath)

    site_col = spec[reg.SPLIT_IDENTITY_SITE_FIELD]
    n_col = spec[reg.SPLIT_IDENTITY_N_FIELD]
    governed = frame
    row_filter = spec["row_filter"]
    if row_filter is not None:
        column, value = row_filter
        governed = frame[frame[column].astype(str) == str(value)]
    if governed.empty:
        raise SplitIdentityError(
            f"{relpath}: no row carries {row_filter[0]}={row_filter[1]!r}, so "
            "the table persists no target-local eligible-site identity at all; "
            "the registered cells are not optional (§H.2)")

    budgets = pd.to_numeric(governed[n_col], errors="coerce")
    if budgets.isna().any():
        position = int(budgets.isna().to_numpy().argmax())
        raise SplitIdentityError(
            f"{relpath}: a governed row carries {n_col}="
            f"{governed[n_col].to_numpy()[position]!r}; a target-local row "
            "always names the label budget whose eligible set it belongs to")
    observed = {int(n): frozenset(block[site_col].astype(str))
                for n, block in governed.groupby(budgets.astype(int))}

    expected_budgets = {int(n) for n in spec["n_values"]}
    if set(observed) != expected_budgets:
        raise SplitIdentityError(
            f"{relpath}: the governed rows cover label budgets "
            f"{sorted(observed)}; this artifact is registered over "
            f"{sorted(expected_budgets)}")

    checked = []
    for n in sorted(expected_budgets):
        expected = reg.expected_identity_sites(spec, n, binding.sites_at(n))
        seen = observed[n]
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise SplitIdentityError(
                f"{relpath}: at N={n} the persisted sites are not the eligible "
                f"identities of the verified split registry "
                f"({binding.registry_path} @ {binding.registry_sha256[:12]}...): "
                f"{len(missing)} eligible sites are absent {missing}, "
                f"{len(extra)} persisted sites are not eligible at N={n} "
                f"{extra}. The count, the 23-site universe, the nesting in N and "
                "the agreement between the two RQ4 tables all survive a "
                "substitution; the identities do not (§D.6, §D.9)")
        checked.append((n, tuple(sorted(seen))))
    return {"sites_by_n": tuple(checked),
            "registry_sha256": binding.registry_sha256,
            "n_governed_rows": int(len(governed))}


def _validate_rq3_status(relpath: str, frame: pd.DataFrame) -> int:
    """The **parent's** own status / failing-condition relationship (§9.5.1).

    Reproduced from `rq3_store`'s validation-record check and `rq3_inference`'s
    `BlockResult`, not reinterpreted here: a unit is `ESTIMABLE` with no failing
    condition, or `NOT_ESTIMABLE` with exactly one of the parent's registered
    numerical-failure codes. `ESTIMABLE` is a *status*, never a failing
    condition, and the node counts a validation record carries are the parent's
    frozen 40 / 60.
    """
    import rq3_protocol as proto

    allowed_status = (proto.STATUS_OK, proto.NOT_ESTIMABLE)
    allowed_conditions = set(proto.NUMERICAL_FAILURE_CODES)
    checked = 0
    for position, row in enumerate(frame.to_dict("records")):
        checked += 1
        status = _cell(row["status"])
        condition = _cell(row["failing_condition"])
        if status not in allowed_status:
            raise StatusSemanticsError(
                f"{relpath}: row {position} carries status {status!r}; the "
                f"parent's validation records are {allowed_status} and Stage F "
                "never widens the parent's vocabulary")
        if status == proto.STATUS_OK:
            if not reg.is_missing(condition):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} is {proto.STATUS_OK} and still "
                    f"carries failing_condition={condition!r}; a passing "
                    "validation unit has no failing condition")
        else:
            if reg.is_missing(condition):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} is {proto.NOT_ESTIMABLE} with "
                    "no failing_condition; the parent requires a registered "
                    "§9.5.1 code")
            if str(condition) not in allowed_conditions:
                raise StatusSemanticsError(
                    f"{relpath}: row {position} carries failing_condition "
                    f"{condition!r}, which is not one of the parent's "
                    f"{len(allowed_conditions)} registered numerical-failure "
                    "codes (§9.5.1)")
        for field, expected in (("authoritative_nodes", proto.AUTHORITATIVE_NODES),
                                ("validation_nodes", proto.GH_NODES_VALIDATION)):
            value = _cell(row[field])
            if reg.is_missing(value):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} leaves the parent-required "
                    f"{field} missing")
            if int(value) != int(expected):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} reports {field}={value!r}, the "
                    f"parent's frozen value is {expected}")
        _validate_numerical_verdict(relpath, position, row, status, condition)
    return checked


def _validate_rq3_cell_status(relpath: str, frame: pd.DataFrame) -> int:
    """§13 — `sigma_site_sq` and the code that withdrew it are one state.

    The cell's two nested block records (§H.4.3) are validated first, and the
    §13 pair is then bound to them by `_validate_sens56_sigma_binding` —
    `sigma_site_sq_condition` is *computed* from the block that withdrew the
    variance, not merely matched against a vocabulary.

    `rq3_inference.summarize_cell` reports the variance **only** when both of the
    cell's §9.6.5 validation units passed and the cell's own full fit was
    estimable, and it then records no condition at all. Otherwise it emits the
    literal `NOT_ESTIMABLE` beside a registered §9.5.1 code — the failing unit's
    own code, in registered block order. Nothing else is a state the parent can
    produce, so a Stage-F reason code here, a condition beside a reported
    variance, or a withdrawn variance with no condition are all refused.
    """
    import rq3_protocol as proto

    checked = 0
    for position, row in enumerate(frame.to_dict("records")):
        checked += 1
        # The nested block records first: the §13 pair below is bound to them,
        # so they have to be known-good before they can bind anything.
        blocks = _validate_sens56_blocks(relpath, position, row)
        _validate_sens56_sigma_binding(relpath, position, row, blocks)
        value = _cell(row[reg.RQ3_CELL_VALUE_FIELD])
        condition = _cell(row[reg.RQ3_CELL_CONDITION_FIELD])
        withdrawn = (isinstance(value, str) and value == reg.NOT_ESTIMABLE)
        if withdrawn:
            if reg.is_missing(condition):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} withdraws "
                    f"{reg.RQ3_CELL_VALUE_FIELD} with no "
                    f"{reg.RQ3_CELL_CONDITION_FIELD}; the parent names the "
                    "failing unit's own §9.5.1 code")
            if str(condition) not in set(proto.NUMERICAL_FAILURE_CODES):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} withdraws "
                    f"{reg.RQ3_CELL_VALUE_FIELD} with "
                    f"{reg.RQ3_CELL_CONDITION_FIELD}={condition!r}, which is "
                    f"not one of the parent's "
                    f"{len(proto.NUMERICAL_FAILURE_CODES)} registered "
                    "numerical-failure codes. Stage-F's own reason vocabulary "
                    "does not reach the parent's §13 report (§9.5.1)")
            continue
        if not reg.is_missing(condition):
            raise StatusSemanticsError(
                f"{relpath}: row {position} reports "
                f"{reg.RQ3_CELL_VALUE_FIELD}={value!r} and still carries "
                f"{reg.RQ3_CELL_CONDITION_FIELD}={condition!r}; the parent "
                "records a condition only where the variance was withdrawn")
        if reg.is_missing(value):
            raise StatusSemanticsError(
                f"{relpath}: row {position} leaves {reg.RQ3_CELL_VALUE_FIELD} "
                f"missing; the parent writes either a number or the literal "
                f"{reg.NOT_ESTIMABLE!r} (§13)")
        if not _is_number(value) or not _finite(value):
            raise NumericSemanticsError(
                f"{relpath}: row {position} reports "
                f"{reg.RQ3_CELL_VALUE_FIELD}={value!r} "
                f"({type(value).__name__}); a reported variance is a genuine "
                "finite number, never a string and never a bool")
        if float(value) < 0.0:
            raise NumericSemanticsError(
                f"{relpath}: row {position} reports a negative "
                f"{reg.RQ3_CELL_VALUE_FIELD}={value!r}")
    return checked


def _sens_label(relpath: str) -> str:
    """The sensitivity directory name of a `sensitivities/<label>/<file>` path."""
    return relpath.split("/")[1]


def _as_index(value):
    """A CSV-roundtripped candidate index as a genuine `int`, or the value.

    A column that mixes integers with the missing representation arrives as
    float64, so `3` becomes `3.0` on the way through pandas. That is a
    serialisation artefact of the table form, not a claim about the value, and
    normalising it is what lets the **parent's own**
    `s2_decisions.assert_decision_field_consistency` run unmodified on a
    persisted row. A genuinely fractional index is left alone and the parent
    rejects it.
    """
    if isinstance(value, bool) or not _is_number(value):
        return value
    return int(value) if float(value).is_integer() else value


def _missing_mask(frame: pd.DataFrame, column: str):
    """Which cells of one column hold a registered missing representation."""
    col = frame[column]
    mask = col.isna()
    if col.dtype == object:
        mask = mask | col.isin(list(reg.MISSING_STRING_VALUES))
    return mask


def _finite_mask(frame: pd.DataFrame, column: str):
    """Which cells hold a genuine finite number (never a bool, never a string)."""
    import numpy as np

    col = frame[column]
    if pd.api.types.is_bool_dtype(col):
        return pd.Series(False, index=col.index)
    if pd.api.types.is_numeric_dtype(col):
        return pd.Series(np.isfinite(col.to_numpy(dtype=float, na_value=np.nan)),
                         index=col.index)
    return col.map(lambda v: _is_number(v) and _finite(v))


def _count_values(relpath: str, frame: pd.DataFrame, column: str):
    """One column as genuine non-negative integral counts, or a refusal."""
    import numpy as np

    col = frame[column]
    if pd.api.types.is_bool_dtype(col):
        raise StatusSemanticsError(
            f"{relpath}: {column} arrived as a boolean column; it is a count "
            "from the parent sequence")
    if not pd.api.types.is_numeric_dtype(col):
        bad = col[~col.map(_is_count)]
        if len(bad):
            raise StatusSemanticsError(
                f"{relpath}: row {col.index.get_loc(bad.index[0])} reports "
                f"{column}={bad.iloc[0]!r}, which is not a genuine "
                "non-negative count from the parent sequence")
    values = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
    bad = np.flatnonzero(~np.isfinite(values) | (values < 0)
                         | (values % 1 != 0))
    if bad.size:
        raise StatusSemanticsError(
            f"{relpath}: row {int(bad[0])} reports {column}="
            f"{col.iloc[int(bad[0])]!r}, which is not a genuine non-negative "
            "count from the parent sequence")
    return values


def _refuse_where(relpath: str, frame: pd.DataFrame, mask, message: str,
                  columns=()) -> None:
    """Raise on the first row of `mask`, quoting the columns that made it fail."""
    import numpy as np

    array = np.asarray(mask, dtype=bool)
    if not array.any():
        return
    position = int(np.flatnonzero(array)[0])
    row = frame.iloc[position]
    detail = ", ".join(f"{c}={row[c]!r}" for c in columns)
    raise StatusSemanticsError(
        f"{relpath}: row {position} {message}" + (f" ({detail})" if detail else ""))


def _validate_sens_inherited_state(relpath: str, frame: pd.DataFrame) -> int:
    """SENS-1..4 inherit the **parent's** state machine, not a blank cheque (§H.4).

    These tables carry no Stage-F `status` column, and that is not permission to
    invent values: §H.4 gives them "the parent's own registered output schema
    unchanged" and §G.2 adds that RQ2/S2's `not_testable_reason_split`
    vocabulary is reused "only inside its parent's own decision artifacts". So
    every governed value here is read from the parent at validation time —
    `certification.STRATEGY_TIER`, `STRATEGY_SCORE_TYPE`, `GUARANTEE_BY_SCORE`,
    `TIER_STATES`, `REALIZED_FIELD` and `s2_decisions.VALID_REASONS` — and no
    copy of any of them is kept in the Stage-F layer.

    Where the sensitivity leaves the candidate grid alone, the parent's own
    `assert_decision_field_consistency` is **executed on the persisted rows**,
    so the enforcer is the parent's code rather than a restatement of it. It is
    run once per distinct `(state, reason, selected_q, index, tau finite, score
    finite)` combination, which is exactly the argument it is a function of, so
    running it per row would decide nothing further. SENS-2 varies the grid step
    and only the grid step (§E.3), so its rows are held to the same state
    machine against `certification.Q_GRID_SENSITIVITY`.

    The counting fields are the parent sequence's own arithmetic: eligibility is
    a suffix of the candidate list, testing stops at the **first** failure, and
    the selected candidate is the last one that passed — so
    `selected_candidate_index = first_eligible_index + n_passed_candidates - 1`,
    a failed decision has passed nothing, and an untestable one had no eligible
    candidate at all.
    """
    import numpy as np

    import certification as cert
    import s2_decisions as s2d

    if frame.empty:
        return 0
    decisions = relpath in reg.SENS_DECISION_TABLES
    label = _sens_label(relpath)
    grid = (cert.Q_GRID_SENSITIVITY if label == reg.SENS2_LABEL else cert.Q_GRID)
    strategies = (reg.SENS_DECISION_STRATEGIES if decisions
                  else reg.SENS_EVALUATION_STRATEGIES)

    strategy_column = frame[reg.SENS_STRATEGY_FIELD]
    unknown = sorted(set(strategy_column) - set(strategies))
    if unknown:
        raise StatusSemanticsError(
            f"{relpath}: rows name strategies {unknown}; §H.4.2 builds this "
            f"table from {list(strategies)}")

    # -- vocabularies, checked over the distinct combinations that occur ----
    pairs = frame[[reg.SENS_STRATEGY_FIELD,
                   reg.SENS_INHERITED_STATE_FIELD]].drop_duplicates()
    for strategy, state in pairs.itertuples(index=False):
        allowed = cert.allowed_states(strategy)
        if state not in allowed:
            raise StatusSemanticsError(
                f"{relpath}: state {state!r} appears for strategy "
                f"{strategy!r}; the parent's tier ({cert.tier_of(strategy)}) "
                f"admits exactly {list(allowed)} and Stage F never widens the "
                "parent's vocabulary")
    bound = [(reg.SENS_INHERITED_TIER_FIELD, cert.STRATEGY_TIER,
              "certification.STRATEGY_TIER")]
    if decisions:
        # §E.2a — the one registered per-label deviation, and it is metadata
        # only. SENS-1's exact-CP tiers really run
        # `certification.overall_error_upper`, so those rows must name that
        # functional rather than inherit the parent's class-conditional label;
        # every other row, and SENS-1's own empirical tiers, keep the parent's.
        # The expected label is computed by the producer's own helper, so the
        # store cannot drift from what the sensitivity actually persists.
        import stage_f_sensitivities as sens
        score_types = {strategy: sens.persisted_score_type(label, strategy)
                       for strategy in cert.STRATEGY_SCORE_TYPE}
        authority = ("stage_f_sensitivities.persisted_score_type (§E.2a)"
                     if label == reg.SENSITIVITY_LABELS[0]
                     else "certification.STRATEGY_SCORE_TYPE")
        bound += [(reg.SENS_INHERITED_SCORE_FIELD, score_types, authority),
                  # The guarantee tier is **not** relabelled: §E.2a corrects the
                  # score's name, and both bounds are exact Clopper-Pearson on
                  # the strategy's own calibration sample.
                  (reg.SENS_INHERITED_GUARANTEE_FIELD,
                   {s: cert.GUARANTEE_BY_SCORE[t]
                    for s, t in cert.STRATEGY_SCORE_TYPE.items()},
                   "certification.GUARANTEE_BY_SCORE")]
    else:
        bound += [(reg.SENS_INHERITED_REALIZED_FIELD, cert.REALIZED_FIELD,
                   "§7's realized-outcome field")]
    for column, mapping, authority in bound:
        seen = frame[[reg.SENS_STRATEGY_FIELD, column]].drop_duplicates()
        for strategy, value in seen.itertuples(index=False):
            if value != mapping[strategy]:
                raise StatusSemanticsError(
                    f"{relpath}: {column}={value!r} appears for strategy "
                    f"{strategy!r}; {authority} fixes it at "
                    f"{mapping[strategy]!r}. A tier may not borrow another "
                    "tier's vocabulary")

    state = frame[reg.SENS_INHERITED_STATE_FIELD]
    is_pass = state.isin(list(cert.PASS_STATES)).to_numpy()
    is_untestable = (state == cert.NOT_TESTABLE).to_numpy()
    is_fail = ~is_pass & ~is_untestable
    checked = int(len(frame))

    if not decisions:
        _validate_sens_evaluations(relpath, frame, grid, is_pass, is_untestable)
        return checked

    reason = frame[reg.SENS_INHERITED_REASON_FIELD]
    invented = sorted({r for r in reason.where(reason.notna(), None)
                       if r not in s2d.VALID_REASONS}, key=str)
    if invented:
        raise StatusSemanticsError(
            f"{relpath}: rows carry reasons {invented}; the parent's closed "
            f"vocabulary is {list(s2d.VALID_REASONS)} "
            "(`s2_decisions.VALID_REASONS`), and a sensitivity inherits it "
            "unchanged (§G.2)")

    missing = {f: _missing_mask(frame, f).to_numpy()
               for f in reg.SENS_SELECTION_FIELDS}
    any_selection = np.logical_or.reduce(list(~m for m in missing.values()))
    all_selection = np.logical_and.reduce(list(~m for m in missing.values()))
    reason_values = reason.where(reason.notna(), None).to_numpy()

    _refuse_where(relpath, frame, is_untestable & any_selection,
                  "is NOT_TESTABLE and still reports a selection; nothing was "
                  "tested, so nothing may be selected", reg.SENS_SELECTION_FIELDS)
    _refuse_where(relpath, frame,
                  is_untestable & ~np.isin(reason_values,
                                           list(s2d.NOT_TESTABLE_REASONS)),
                  "is NOT_TESTABLE with a reason the parent does not admit "
                  f"there ({list(s2d.NOT_TESTABLE_REASONS)})",
                  (reg.SENS_INHERITED_REASON_FIELD,))
    _refuse_where(relpath, frame, is_fail & any_selection,
                  "is a fail state and still selects a threshold",
                  reg.SENS_SELECTION_FIELDS)
    _refuse_where(relpath, frame, is_fail & (reason_values != s2d.REASON_NO_PASS),
                  f"is a fail state without reason {s2d.REASON_NO_PASS!r}",
                  (reg.SENS_INHERITED_STATE_FIELD,
                   reg.SENS_INHERITED_REASON_FIELD))
    _refuse_where(relpath, frame, is_pass & ~all_selection,
                  "is a pass state and leaves part of its selection missing; a "
                  "passing decision names the threshold it selected",
                  reg.SENS_SELECTION_FIELDS)
    _refuse_where(relpath, frame, is_pass & (reason_values != None),  # noqa: E711
                  "passed and still carries a failure reason",
                  (reg.SENS_INHERITED_REASON_FIELD,))
    for column in ("selected_tau", "selected_score"):
        _refuse_where(relpath, frame,
                      is_pass & ~_finite_mask(frame, column).to_numpy(),
                      f"is a pass state with a non-finite {column}", (column,))

    q_values = pd.to_numeric(frame["selected_q"], errors="coerce").to_numpy()
    off_grid = is_pass & ~np.isin(np.round(q_values, 10),
                                  np.round(np.asarray(grid, dtype=float), 10))
    _refuse_where(relpath, frame, off_grid,
                  f"selected a q off this sensitivity's frozen "
                  f"{len(grid)}-candidate grid", ("selected_q",))
    index_values = pd.to_numeric(frame["selected_candidate_index"],
                                 errors="coerce").to_numpy()
    bad_index = is_pass & (~np.isfinite(index_values) | (index_values % 1 != 0)
                           | (index_values < 0) | (index_values >= len(grid)))
    _refuse_where(relpath, frame, bad_index,
                  "carries a selected_candidate_index that is not a position "
                  "on the frozen grid", ("selected_candidate_index",))
    safe = np.where(is_pass & ~bad_index, index_values, 0).astype(int)
    grid_at = np.asarray(grid, dtype=float)[safe]
    _refuse_where(relpath, frame,
                  is_pass & ~bad_index & (np.round(grid_at, 10)
                                          != np.round(q_values, 10)),
                  "points at a grid position that is not the q it reports",
                  ("selected_q", "selected_candidate_index"))

    n_eligible = _count_values(relpath, frame, "n_eligible")
    n_passed = _count_values(relpath, frame, "n_passed_candidates")
    first_missing = _missing_mask(frame, "first_eligible_index").to_numpy()
    stopped_missing = _missing_mask(frame, "stopped_at_index").to_numpy()
    _refuse_where(relpath, frame,
                  is_untestable & ((n_eligible != 0) | (n_passed != 0)
                                   | ~first_missing | ~stopped_missing),
                  "is NOT_TESTABLE while reporting that it evaluated "
                  "something; the parent reaches that state exactly when no "
                  "candidate was eligible",
                  ("n_eligible", "first_eligible_index", "n_passed_candidates",
                   "stopped_at_index"))
    tested = ~is_untestable
    _refuse_where(relpath, frame, tested & ((n_eligible <= 0) | first_missing),
                  "was tested with no eligible candidate to test",
                  ("n_eligible", "first_eligible_index"))
    _refuse_where(relpath, frame, tested & (n_passed > n_eligible),
                  "passed more candidates than were eligible",
                  ("n_passed_candidates", "n_eligible"))
    _refuse_where(relpath, frame, is_fail & (n_passed != 0),
                  "is a fail state having passed a candidate; the parent "
                  "selects the last candidate that passed",
                  ("n_passed_candidates",))
    _refuse_where(relpath, frame, is_fail & stopped_missing,
                  "is a fail state naming no stopped_at_index; the parent "
                  "reaches it only by stopping at a candidate it failed to "
                  "reject", ("stopped_at_index",))
    first = np.where(first_missing, 0,
                     pd.to_numeric(frame["first_eligible_index"],
                                   errors="coerce").fillna(0).to_numpy())
    _refuse_where(relpath, frame,
                  is_pass & ((n_passed <= 0)
                             | (index_values != first + n_passed - 1)),
                  "selected a candidate that is not "
                  "`first_eligible_index + n_passed_candidates - 1`; "
                  "eligibility is a suffix and testing stops at the first "
                  "failure", ("selected_candidate_index",
                              "first_eligible_index", "n_passed_candidates"))

    for column in reg.SENS_CALIBRATION_COUNT_FIELDS:
        counts = _count_values(relpath, frame, column)
        _refuse_where(relpath, frame, ~is_pass & (counts != 0),
                      f"selected nothing and still reports {column}; the "
                      "parent's convention for an unselected decision is its "
                      "zeroed count record", (column,))

    if label != reg.SENS2_LABEL:
        # the parent's own validator, on every distinct argument it sees
        probe = pd.DataFrame({
            "state": state,
            "reason": reason.where(reason.notna(), None),
            "selected_q": frame["selected_q"],
            "selected_candidate_index": frame["selected_candidate_index"],
            "_tau": _finite_mask(frame, "selected_tau"),
            "_score": _finite_mask(frame, "selected_score"),
            "_tau_missing": _missing_mask(frame, "selected_tau"),
            "_score_missing": _missing_mask(frame, "selected_score"),
        }).drop_duplicates()
        for record in probe.to_dict("records"):
            payload = {
                "state": record["state"], "reason": record["reason"],
                "selected_q": (None if _cell(record["selected_q"]) is None
                               else float(record["selected_q"])),
                "selected_candidate_index":
                    _as_index(_cell(record["selected_candidate_index"])),
                "selected_tau": (None if record["_tau_missing"]
                                 else (1.0 if record["_tau"] else float("inf"))),
                "selected_score": (None if record["_score_missing"]
                                   else (1.0 if record["_score"]
                                         else float("inf"))),
            }
            try:
                s2d.assert_decision_field_consistency(payload)
            except s2d.DecisionArtifactError as exc:
                raise StatusSemanticsError(
                    f"{relpath}: a persisted decision state is refused by the "
                    f"parent's own `s2_decisions."
                    f"assert_decision_field_consistency`: {exc}") from exc
    return checked


def _validate_sens_evaluations(relpath: str, frame: pd.DataFrame, grid,
                               is_pass, is_untestable) -> None:
    """SENS-1..4 evaluations: the realized outcome follows its own decision.

    `stage_f_sensitivities.evaluate_decision` reads the decision's own
    `selected_tau`: no threshold means no retained set, so the coverage, the
    selective BER, the risk excess and the realized outcome are all absent and
    the retained counts are the parent's zeros. Where a threshold exists the BER
    may still be `NOT_ESTIMABLE` (§10) — and then the two quantities derived
    from it go with it, because neither can be computed without it.
    """
    import numpy as np

    tau_missing = _missing_mask(frame, "selected_tau").to_numpy()
    has_tau = ~tau_missing
    _refuse_where(relpath, frame, has_tau != is_pass,
                  "carries a threshold its own decision did not select (only a "
                  "passing decision selected one)",
                  (reg.SENS_INHERITED_STATE_FIELD, "selected_tau"))
    for column in ("selected_q", "selected_candidate_index"):
        _refuse_where(relpath, frame,
                      _missing_mask(frame, column).to_numpy() == has_tau,
                      f"reports {column} out of step with selected_tau; the "
                      "decision's selection fields are jointly present or "
                      "jointly absent", (column, "selected_tau"))
    q_values = pd.to_numeric(frame["selected_q"], errors="coerce").to_numpy()
    _refuse_where(relpath, frame,
                  has_tau & ~np.isin(np.round(q_values, 10),
                                     np.round(np.asarray(grid, dtype=float), 10)),
                  f"carries a selected_q off this sensitivity's frozen "
                  f"{len(grid)}-candidate grid", ("selected_q",))

    counts = {c: _count_values(relpath, frame, c)
              for c in reg.SENS_EVALUATION_COUNT_FIELDS}
    for column, values in counts.items():
        _refuse_where(relpath, frame, ~has_tau & (values != 0),
                      f"selected no threshold and still retained {column}",
                      (column,))
    for column in ("coverage_test",) + reg.SENS_EVALUATION_VALUE_FIELDS:
        _refuse_where(relpath, frame,
                      ~has_tau & ~_missing_mask(frame, column).to_numpy(),
                      f"selected no threshold and still reports {column}; with "
                      "no retained set there is nothing to evaluate", (column,))

    coverage = pd.to_numeric(frame["coverage_test"], errors="coerce").to_numpy()
    _refuse_where(relpath, frame,
                  has_tau & (~np.isfinite(coverage) | (coverage < 0.0)
                             | (coverage > 1.0)),
                  "applied a threshold and reports a realized coverage that is "
                  "not a number in [0, 1]", ("coverage_test",))
    ber_missing = _missing_mask(frame, "ber_test").to_numpy()
    for column in ("risk_excess", "realized_outcome"):
        _refuse_where(relpath, frame,
                      has_tau & ber_missing
                      & ~_missing_mask(frame, column).to_numpy(),
                      f"reports {column} while its selective BER is "
                      "unavailable; both are computed from that BER and "
                      "neither survives it (§10)", ("ber_test", column))
    computed = has_tau & ~ber_missing
    for column in ("ber_test", "r_star", "risk_excess"):
        _refuse_where(relpath, frame,
                      computed & ~_finite_mask(frame, column).to_numpy(),
                      f"reports a {column} that is not a genuine finite number "
                      "where the BER exists", (column,))
    ber = pd.to_numeric(frame["ber_test"], errors="coerce").to_numpy()
    r_star = pd.to_numeric(frame["r_star"], errors="coerce").to_numpy()
    excess = pd.to_numeric(frame["risk_excess"], errors="coerce").to_numpy()
    expected = ber - r_star
    tolerance = 1e-12 * np.maximum(1.0, np.abs(expected))
    _refuse_where(relpath, frame,
                  computed & (np.abs(excess - expected) > tolerance),
                  "reports a risk_excess that is not `ber_test - r_star`",
                  ("ber_test", "r_star", "risk_excess"))
    outcome = frame["realized_outcome"]
    if pd.api.types.is_bool_dtype(outcome):
        raise StatusSemanticsError(
            f"{relpath}: realized_outcome arrived as a boolean column; the "
            "realized violation indicator is 0 or 1")
    outcome_values = pd.to_numeric(outcome, errors="coerce").to_numpy()
    _refuse_where(relpath, frame,
                  computed & ~np.isin(outcome_values, (0.0, 1.0)),
                  "reports a realized_outcome that is not 0 or 1",
                  ("realized_outcome",))
    _refuse_where(relpath, frame,
                  computed & np.isin(outcome_values, (0.0, 1.0))
                  & (outcome_values != (ber > r_star).astype(float)),
                  "reports a realized_outcome that is not exactly "
                  "`ber_test > r_star`",
                  ("ber_test", "r_star", "realized_outcome"))


def _parent_blocks(population: str) -> tuple:
    """The two registered blocks of one population, from `rq3_protocol.BLOCK_DF`."""
    import rq3_protocol as proto

    return tuple(block for pop, block in proto.BLOCK_DF if pop == population)


def _as_mapping(value):
    """A nested record as a mapping, or `None` if it is not one."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _validate_sens56_blocks(relpath: str, position: int, row: dict) -> dict:
    """The two block records **inside** a SENS-5/6 cell row (§H.4.3).

    §H.4.3 is explicit that each cell record "carries both of them within the
    row, exactly as the parent's per-cell serialization does". That makes the
    `blocks` column a transcription of two `rq3_inference.BlockResult` payloads,
    not an opaque blob: it repeats the cell's own identity, the parent's
    validation-unit tuple, the block's registered degrees of freedom, its status,
    its §9.5.1 failing condition, the structured 40-vs-60 verdict and the four
    block-level scientific quantities. Every one of them is checked against the
    parent's own definitions, and the verdict goes through the **same**
    `_validate_numerical_verdict` that governs `validation_units.csv`, so a
    malformed verdict cannot enter through the nested copy.

    Returns the normalised `{block: record}` map for the cross-file binding.
    """
    import rq3_protocol as proto

    population = str(_cell(row["population"]))
    operating_point = str(_cell(row["operating_point"]))
    method = str(_cell(row["method"]))
    expected_blocks = _parent_blocks(population)
    if not expected_blocks:
        raise StatusSemanticsError(
            f"{relpath}: row {position} names population {population!r}, which "
            f"is not one of the parent's {sorted({p for p, _ in proto.BLOCK_DF})}")
    if (operating_point, method) not in proto.CELL_REGISTRY:
        raise StatusSemanticsError(
            f"{relpath}: row {position} names cell "
            f"({operating_point!r}, {method!r}), which is not in the parent's "
            f"{len(proto.CELL_REGISTRY)}-cell `rq3_protocol.CELL_REGISTRY`")

    blocks = _as_mapping(_cell(row[reg.SENS56_CELLS_BLOCKS_FIELD]))
    if blocks is None:
        raise StatusSemanticsError(
            f"{relpath}: row {position} carries "
            f"{reg.SENS56_CELLS_BLOCKS_FIELD}="
            f"{row[reg.SENS56_CELLS_BLOCKS_FIELD]!r}; §H.4.3 carries the two "
            "block records inside the row, as a mapping keyed by block name")
    observed = tuple(sorted(blocks))
    if observed != tuple(sorted(expected_blocks)):
        missing = [b for b in expected_blocks if b not in blocks]
        extra = [b for b in observed if b not in expected_blocks]
        raise StatusSemanticsError(
            f"{relpath}: row {position} carries blocks {list(observed)}; "
            f"{population} has exactly the parent's "
            f"{list(expected_blocks)} (`rq3_protocol.BLOCK_DF`) — "
            f"{len(missing)} missing {missing}, {len(extra)} unregistered "
            f"{extra}. Exactly two block-level units per cell, in every cell "
            "(§H.4.3)")
    if len(blocks) != reg.SENS56_VALIDATION_UNITS_PER_CELL:
        raise StatusSemanticsError(
            f"{relpath}: row {position} carries {len(blocks)} block records; "
            f"the frozen count is {reg.SENS56_VALIDATION_UNITS_PER_CELL}")

    normalised = {}
    for name in expected_blocks:
        record = _as_mapping(blocks[name])
        where = f"{relpath}: row {position} block {name!r}"
        if record is None:
            raise StatusSemanticsError(
                f"{where} is {blocks[name]!r}; the parent writes a structured "
                "`BlockResult.to_payload()` record")
        if tuple(sorted(record)) != reg.SENS56_BLOCK_KEYS:
            got = tuple(sorted(record))
            raise StatusSemanticsError(
                f"{where} carries keys {list(got)}; the parent's "
                f"`BlockResult.to_payload()` record is "
                f"{list(reg.SENS56_BLOCK_KEYS)}")
        for field, expected in (("population", population), ("block", name),
                                ("operating_point", operating_point),
                                ("method", method)):
            if record[field] != expected:
                raise StatusSemanticsError(
                    f"{where} reports {field}={record[field]!r}; the row it is "
                    f"carried in is {expected!r}, and a block record belongs "
                    "to its own cell")
        if record["cell_index"] != proto.CELL_REGISTRY.index(
                (operating_point, method)):
            raise StatusSemanticsError(
                f"{where} reports cell_index={record['cell_index']!r}; the "
                f"parent's `CELL_REGISTRY` position of "
                f"({operating_point}, {method}) is "
                f"{proto.CELL_REGISTRY.index((operating_point, method))}")
        if tuple(record["validation_unit"]) != tuple(proto.VALIDATION_UNIT):
            raise StatusSemanticsError(
                f"{where} declares validation_unit="
                f"{record['validation_unit']!r}; the parent's §9.6.5 unit is "
                f"{list(proto.VALIDATION_UNIT)}")
        expected_df = proto.BLOCK_DF[(population, name)]
        if _as_index(record["df"]) != expected_df:
            raise StatusSemanticsError(
                f"{where} declares df={record['df']!r}; the parent's frozen "
                f"`BLOCK_DF[({population!r}, {name!r})]` is {expected_df}")
        if not isinstance(record["detail"], dict):
            raise StatusSemanticsError(
                f"{where} carries detail of type "
                f"{type(record['detail']).__name__}; the parent writes the "
                "40-node solution's own detail mapping")

        status = record["status"]
        condition = record["failing_condition"]
        if status not in (proto.STATUS_OK, proto.NOT_ESTIMABLE):
            raise StatusSemanticsError(
                f"{where} carries status {status!r}; the parent's block "
                f"statuses are {(proto.STATUS_OK, proto.NOT_ESTIMABLE)}")
        estimable = status == proto.STATUS_OK
        if estimable and condition is not None:
            raise StatusSemanticsError(
                f"{where} is {proto.STATUS_OK} and still names "
                f"failing_condition={condition!r}")
        if not estimable:
            if condition is None or str(condition) not in set(
                    proto.NUMERICAL_FAILURE_CODES):
                raise StatusSemanticsError(
                    f"{where} is {proto.NOT_ESTIMABLE} with failing_condition="
                    f"{condition!r}; the parent names one of its "
                    f"{len(proto.NUMERICAL_FAILURE_CODES)} registered §9.5.1 "
                    "codes")
        expected_numerical = (None if condition is None
                              else bool(proto.is_numerical_failure(condition)))
        if record["failure_is_numerical"] != expected_numerical:
            raise StatusSemanticsError(
                f"{where} reports failure_is_numerical="
                f"{record['failure_is_numerical']!r}; "
                f"`rq3_protocol.is_numerical_failure({condition!r})` is "
                f"{expected_numerical!r}")
        for field in reg.SENS56_BLOCK_VALUE_FIELDS:
            value = record[field]
            if estimable:
                if not _is_number(value) or not _finite(value):
                    raise StatusSemanticsError(
                        f"{where} is {proto.STATUS_OK} and reports "
                        f"{field}={value!r}; an estimable block carries a "
                        "genuine finite number")
            elif value != proto.NOT_ESTIMABLE:
                raise StatusSemanticsError(
                    f"{where} is {proto.NOT_ESTIMABLE} and reports "
                    f"{field}={value!r}; the parent writes the literal "
                    f"{proto.NOT_ESTIMABLE!r}")
        _validate_numerical_verdict(f"{relpath}[blocks/{name}]", position,
                                    record, status, condition,
                                    empty_form=VERDICT_EMPTY_ABSENT)
        verdict = record["numerical_validation"]
        detail = record["failure_detail"]
        if estimable:
            if detail is not None:
                raise StatusSemanticsError(
                    f"{where} is {proto.STATUS_OK} and carries failure_detail="
                    f"{detail!r}")
        elif not isinstance(detail, str) or not detail.strip():
            raise StatusSemanticsError(
                f"{where} is {proto.NOT_ESTIMABLE} without a failure_detail; "
                "the parent always records which model, fold or node count "
                "failed")
        elif verdict:
            # `evaluate_block` writes `failure_detail=str(verdict["reason"])`
            # on exactly the two verdict-bearing branches (§9.6.4).
            if detail != str(verdict["reason"]):
                raise StatusSemanticsError(
                    f"{where} failed numerical validation with reason "
                    f"{verdict['reason']!r} and records failure_detail="
                    f"{detail!r}; the parent carries the verdict's own reason "
                    "through")
        normalised[name] = record
    return normalised


def _validate_sens56_sigma_binding(relpath: str, position: int, row: dict,
                                   blocks: dict) -> None:
    """§13 — the withdrawn variance and the block that withdrew it (CRITICAL F).

    `rq3_inference.summarize_cell` is the whole rule: `sigma_site^2` is reported
    only when **both** of the cell's §9.6.5 units passed (a block cannot pass
    with an unestimable full fit, so `full_fit.estimable` follows), and it is
    then recorded with no condition at all. Otherwise the literal
    `NOT_ESTIMABLE` is written beside **the first failing block's own code, in
    registered block order** — not a code chosen for the cell, and not any
    §9.5.1 code that happens to be registered.

    So the condition is not merely drawn from a vocabulary here: it is computed
    from the cell's own validated nested block records and compared.
    """
    import rq3_protocol as proto

    population = str(_cell(row["population"]))
    order = _parent_blocks(population)
    failing = [blocks[name]["failing_condition"] for name in order
               if blocks[name]["status"] != proto.STATUS_OK]
    value = _cell(row[reg.RQ3_CELL_VALUE_FIELD])
    condition = _cell(row[reg.RQ3_CELL_CONDITION_FIELD])
    withdrawn = isinstance(value, str) and value == reg.NOT_ESTIMABLE

    if failing:
        if not withdrawn:
            raise StatusSemanticsError(
                f"{relpath}: row {position} reports "
                f"{reg.RQ3_CELL_VALUE_FIELD}={value!r} while "
                f"{len(failing)} of its own block records are "
                f"{proto.NOT_ESTIMABLE}; §13 reports the variance only when "
                "both of the cell's validation units passed, because both "
                "carry the very `M_full` the variance comes from")
        if str(condition) != str(failing[0]):
            raise StatusSemanticsError(
                f"{relpath}: row {position} withdraws "
                f"{reg.RQ3_CELL_VALUE_FIELD} naming {condition!r}, but the "
                f"first failing block of {population} in registered order "
                f"{list(order)} failed with {failing[0]!r}; the condition is "
                "the failing unit's own code, not one chosen for the cell")
        return
    if withdrawn:
        raise StatusSemanticsError(
            f"{relpath}: row {position} withdraws {reg.RQ3_CELL_VALUE_FIELD} "
            f"with {reg.RQ3_CELL_CONDITION_FIELD}={condition!r} while both of "
            "its own block records passed; nothing in the cell withdrew it")


#: How a unit that never reached 40-vs-60 records the fact. The two persisted
#: representations differ, and both are the parent's:
#: `BlockResult.numerical_validation` defaults to `{}` and `to_payload` carries
#: that empty mapping into the nested `cells.csv` record, while
#: `rq3_execution._persist_cell_summary` writes
#: `result.numerical_validation or {…}` — substituting a **canonical** empty
#: verdict — into `validation_units.csv`. Neither is optional and neither is
#: interchangeable with the other.
VERDICT_EMPTY_ABSENT = "absent"
VERDICT_EMPTY_CANONICAL = "canonical"


def _canonical_empty_verdict(condition) -> dict:
    """`rq3_execution._persist_cell_summary`'s substitute, rebuilt from the parent."""
    import rq3_protocol as proto

    return {"passed": False, "n_evaluated": 0,
            "n_required": int(proto.N_REQUIRED_AGREEMENT_CRITERIA),
            "n_passed": 0, "reason": condition, "criteria": {}}


def _validate_numerical_verdict(relpath: str, position: int, row: dict,
                                status, condition,
                                empty_form: str = VERDICT_EMPTY_CANONICAL) -> None:
    """The parent's **structured** 40-vs-60 verdict — identity, arithmetic, shape.

    Every rule below is read out of the parent's own code rather than restated
    here, because a Stage-F copy of the seven criteria or of their tolerances
    would be exactly the divergence this check exists to detect:

    * `rq3_protocol.REQUIRED_AGREEMENT_CRITERIA` is the criterion registry.
      `validate_rq3_protocol_freeze.agreement_checks` builds its verdict by
      iterating that tuple and then asserts `tuple(verdict) == ` it, so a
      persisted complete verdict names **exactly** those seven — no rename, no
      alias, no extra, no omission, and (a mapping having unique keys) no
      duplicate;
    * `rq3_protocol.AGREEMENT_TOL[criterion]` is that criterion's tolerance. It
      is looked up by name, so a record carrying another criterion's tolerance —
      or a plausible-looking number of its own — did not come from the parent;
    * `passed` is `discrepancy <= tolerance`, evaluated in that one place. It is
      never an independent flag, so a record may not disagree with its own
      arithmetic in either direction;
    * every discrepancy is `abs(...)`, `sigma2_relative_difference` or
      `coefficient_discrepancy` and is checked finite by the parent, so it is a
      genuine finite non-negative number;
    * `numerical_validation_verdict` has exactly two shapes: seven criteria
      evaluated, or the `ValidationFailure` branch with `n_evaluated = 0` and no
      criteria at all. Any other count never came from the parent;
    * `rq3_inference.evaluate_block` reaches the verdict **only** after both node
      counts produced a solution. A block that failed earlier — `NO_ELIGIBLE_
      START`, `OPTIMIZER_NOT_CONVERGED`, `CV_FOLD_NOT_ESTIMABLE`,
      `NESTED_LIKELIHOOD_INVERSION`, … — carries `BlockResult`'s default empty
      record, and the two verdict-bearing codes are the parent's own
      `CODE_QUADRATURE` and `CODE_VALIDATION_MALFORMED`, imported here by name.
    """
    import rq3_protocol as proto
    import rq3_inference as parent

    required = int(proto.N_REQUIRED_AGREEMENT_CRITERIA)
    registry = tuple(proto.REQUIRED_AGREEMENT_CRITERIA)
    tolerances = dict(proto.AGREEMENT_TOL)
    verdict_bearing = (parent.CODE_QUADRATURE, parent.CODE_VALIDATION_MALFORMED)

    verdict = _cell(row["numerical_validation"])
    if verdict is None or (isinstance(verdict, str) and reg.is_missing(verdict)):
        raise StatusSemanticsError(
            f"{relpath}: row {position} carries no numerical_validation "
            "record; the parent writes one for every unit — a full verdict "
            "where it reached numerical validation, its empty default where it "
            "did not")
    if not isinstance(verdict, dict):
        raise StatusSemanticsError(
            f"{relpath}: row {position} carries numerical_validation of type "
            f"{type(verdict).__name__}; the parent's verdict is a structured "
            "record, never a scalar or a string")

    code = None if reg.is_missing(condition) else str(condition)
    if code is not None and code not in verdict_bearing:
        # §9.6.1 step 6 — the block failed before 40-vs-60 was attempted. The
        # two artifacts record that differently, and each is held to its own
        # producer's form rather than to the laxer of the two.
        if empty_form == VERDICT_EMPTY_ABSENT:
            if verdict:
                raise StatusSemanticsError(
                    f"{relpath}: row {position} failed with {code}, which "
                    f"`rq3_inference.evaluate_block` raises **before** "
                    f"numerical validation is attempted, and still carries a "
                    f"{len(verdict)}-key numerical_validation record. "
                    f"`BlockResult` carries its empty default there; the only "
                    f"codes that carry a verdict are {verdict_bearing}")
            return
        expected = _canonical_empty_verdict(code)
        if verdict != expected:
            raise StatusSemanticsError(
                f"{relpath}: row {position} failed with {code}, which "
                f"`rq3_inference.evaluate_block` raises **before** numerical "
                f"validation is attempted, and records "
                f"numerical_validation={verdict!r}. "
                f"`rq3_execution._persist_cell_summary` writes exactly "
                f"{expected!r} there — no criteria, nothing evaluated, and the "
                f"failing condition as its own reason. The only codes that "
                f"carry a real verdict are {verdict_bearing}")
        return

    expected_keys = ("criteria", "n_evaluated", "n_passed", "n_required",
                     "passed", "reason")
    if tuple(sorted(verdict)) != expected_keys:
        raise StatusSemanticsError(
            f"{relpath}: row {position} carries a numerical_validation record "
            f"with keys {tuple(sorted(verdict))}; the parent's record is "
            f"{expected_keys}")
    if not isinstance(verdict["passed"], bool):
        raise StatusSemanticsError(
            f"{relpath}: row {position} reports passed={verdict['passed']!r}, "
            "which is not a genuine boolean verdict")
    counts = {}
    for name in ("n_evaluated", "n_required", "n_passed"):
        value = verdict[name]
        if not _is_count(value):
            raise StatusSemanticsError(
                f"{relpath}: row {position} reports {name}={value!r}, which is "
                "not a genuine integral count of agreement criteria")
        counts[name] = int(value)
    if counts["n_required"] != required:
        raise StatusSemanticsError(
            f"{relpath}: row {position} declares {counts['n_required']} "
            f"required agreement criteria; the parent's frozen §9.6.4 count is "
            f"{required}")
    if counts["n_evaluated"] not in (0, required):
        raise StatusSemanticsError(
            f"{relpath}: row {position} reports {counts['n_evaluated']} "
            f"criteria evaluated; `numerical_validation_verdict` evaluates the "
            f"whole registry or none of it, so the only counts it can produce "
            f"are 0 and {required}")
    if not 0 <= counts["n_passed"] <= counts["n_evaluated"]:
        raise StatusSemanticsError(
            f"{relpath}: row {position} reports {counts['n_passed']} of "
            f"{counts['n_evaluated']} criteria passed; the counts are not a "
            "nested set")
    if not isinstance(verdict["criteria"], dict):
        raise StatusSemanticsError(
            f"{relpath}: row {position} carries criteria of type "
            f"{type(verdict['criteria']).__name__}; the parent's criteria are a "
            "per-criterion record")
    if len(verdict["criteria"]) != counts["n_evaluated"]:
        raise StatusSemanticsError(
            f"{relpath}: row {position} evaluated {counts['n_evaluated']} "
            f"criteria and records {len(verdict['criteria'])}")

    # -- criterion identity: the parent's registry, exactly ----------------
    observed = tuple(sorted(verdict["criteria"]))
    expected_criteria = () if counts["n_evaluated"] == 0 else tuple(sorted(registry))
    if observed != expected_criteria:
        missing = [c for c in expected_criteria if c not in verdict["criteria"]]
        extra = [c for c in observed if c not in expected_criteria]
        raise StatusSemanticsError(
            f"{relpath}: row {position} records criteria {list(observed)}; the "
            f"parent's §9.6.4 registry is {list(expected_criteria)} "
            f"({len(missing)} missing {missing}, {len(extra)} unregistered "
            f"{extra}). `agreement_checks` iterates "
            "`rq3_protocol.REQUIRED_AGREEMENT_CRITERIA` and asserts the result "
            "equals it, so a renamed, aliased, duplicated, substituted or "
            "omitted criterion did not come from the parent")

    # -- criterion arithmetic: tolerance identity and the passed relation ---
    for name, entry in sorted(verdict["criteria"].items()):
        if not isinstance(entry, dict) \
                or tuple(sorted(entry)) != ("discrepancy", "passed", "tolerance"):
            raise StatusSemanticsError(
                f"{relpath}: row {position} criterion {name!r} is not the "
                "parent's (discrepancy, tolerance, passed) record")
        if not isinstance(entry["passed"], bool):
            raise StatusSemanticsError(
                f"{relpath}: row {position} criterion {name!r} reports "
                f"passed={entry['passed']!r}, which is not a boolean")
        for measure in ("discrepancy", "tolerance"):
            if not _is_number(entry[measure]) or not _finite(entry[measure]):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} criterion {name!r} reports "
                    f"{measure}={entry[measure]!r}, which is not a finite "
                    "number")
        discrepancy, tolerance = float(entry["discrepancy"]), float(entry["tolerance"])
        if discrepancy < 0.0:
            raise StatusSemanticsError(
                f"{relpath}: row {position} criterion {name!r} reports a "
                f"negative discrepancy {discrepancy!r}; every parent "
                "discrepancy is an absolute or relative magnitude")
        frozen = float(tolerances[name])
        if tolerance != frozen:
            raise StatusSemanticsError(
                f"{relpath}: row {position} criterion {name!r} was compared "
                f"against tolerance {tolerance!r}; the parent's frozen "
                f"`rq3_protocol.AGREEMENT_TOL[{name!r}]` is {frozen!r}. A "
                "criterion judged against another criterion's tolerance, or "
                "against one invented for the record, is not the parent's test")
        if entry["passed"] is not bool(discrepancy <= tolerance):
            raise StatusSemanticsError(
                f"{relpath}: row {position} criterion {name!r} reports "
                f"passed={entry['passed']} with discrepancy={discrepancy!r} "
                f"and tolerance={tolerance!r}; the parent's verdict is exactly "
                "`discrepancy <= tolerance` and is never an independent flag")

    n_criteria_passed = sum(1 for e in verdict["criteria"].values() if e["passed"])
    if n_criteria_passed != counts["n_passed"]:
        raise StatusSemanticsError(
            f"{relpath}: row {position} reports n_passed={counts['n_passed']} "
            f"while {n_criteria_passed} recorded criteria passed")
    # `passed` is the parent's seven-of-seven arithmetic, never a free flag.
    should_pass = (counts["n_evaluated"] == required
                   and counts["n_passed"] == required)
    if verdict["passed"] is not should_pass:
        raise StatusSemanticsError(
            f"{relpath}: row {position} reports passed={verdict['passed']} with "
            f"{counts['n_passed']} of {counts['n_evaluated']} criteria met; the "
            f"parent's §9.6.4 verdict passes only on {required} of {required}")
    # -- the reason is the parent's own sentence, not free text -------------
    # `numerical_validation_verdict` constructs `reason` in exactly three ways:
    # `None` on a pass, `f"only {n} of {required} criteria evaluated"` on an
    # incomplete verdict, and `f"criteria beyond tolerance: {breached}"` on a
    # complete one, where `breached` lists the criteria whose own `passed` flag
    # is false, **in `REQUIRED_AGREEMENT_CRITERIA` order**. So a breached-
    # tolerance reason is a function of the record's own criteria: prose, a
    # criterion that is not in the registry, and a criterion that this very
    # record reports as passing are all refused by the same comparison.
    reason = verdict["reason"]
    if verdict["passed"]:
        if reason is not None:
            raise StatusSemanticsError(
                f"{relpath}: row {position} passed numerical validation and "
                f"still records reason={reason!r}")
    elif not isinstance(reason, str) or not reason.strip():
        raise StatusSemanticsError(
            f"{relpath}: row {position} failed numerical validation without a "
            "reason; the parent always states why")
    elif counts["n_evaluated"] == required:
        breached = [name for name in registry
                    if not verdict["criteria"][name]["passed"]]
        expected_reason = f"criteria beyond tolerance: {breached}"
        if reason != expected_reason:
            raise StatusSemanticsError(
                f"{relpath}: row {position} records reason={reason!r}; the "
                f"parent's `numerical_validation_verdict` writes "
                f"{expected_reason!r} for this record — the criteria this very "
                "record reports as breached, in "
                "`rq3_protocol.REQUIRED_AGREEMENT_CRITERIA` order. A "
                "quadrature disagreement names the criterion that was "
                "breached; it is not free text, and a criterion the record "
                "reports as passing cannot be the one that failed")

    # -- the verdict and the unit's status are one decision, not two --------
    if status == proto.STATUS_OK and not verdict["passed"]:
        raise StatusSemanticsError(
            f"{relpath}: row {position} is {proto.STATUS_OK} with a failed or "
            "malformed numerical-validation verdict "
            f"({counts['n_passed']}/{counts['n_evaluated']} of {required}); "
            "§9.6.1 step 7 makes the seven-of-seven verdict a precondition of "
            "estimability, so an empty failing_condition does not rescue it")
    if status != proto.STATUS_OK:
        if verdict["passed"]:
            raise StatusSemanticsError(
                f"{relpath}: row {position} is {proto.NOT_ESTIMABLE} with a "
                "passing numerical-validation verdict")
        if code == parent.CODE_VALIDATION_MALFORMED and counts["n_evaluated"]:
            raise StatusSemanticsError(
                f"{relpath}: row {position} reports "
                f"{parent.CODE_VALIDATION_MALFORMED} with "
                f"{counts['n_evaluated']} criteria evaluated; the parent emits "
                "that code from the `ValidationFailure` branch, where no "
                "criterion could be evaluated at all")
        if code == parent.CODE_QUADRATURE and counts["n_evaluated"] != required:
            raise StatusSemanticsError(
                f"{relpath}: row {position} reports {parent.CODE_QUADRATURE} "
                f"with {counts['n_evaluated']} of {required} criteria "
                "evaluated; a breached tolerance presupposes a complete verdict")


def _validate_reason_state(relpath: str, position: int, row: dict,
                           status, reason) -> None:
    """The field state a reason **asserts**, artifact by artifact (§G.2, §G.4).

    A reason code is a claim about how the producer got here, and the row has to
    be consistent with that claim. `NO_EVALUATION_POINTS` is emitted from
    `n == 0`, so a row invoking it while reporting evaluated cases is claiming
    two incompatible things; `NO_ESTIMABLE_SITE_IN_CELL` is emitted from an empty
    estimable-site list, which forces the derived counts to zero with it;
    `COVERAGE_MATCH_UNATTAINABLE` presupposes a non-empty confidence vector,
    because an empty one is the *other* reason; `SVM_NOT_CONVERGED` is defined as
    liblinear reporting non-convergence, so the row's own convergence flag must
    say so. The mirror-image rules hold on the `ESTIMABLE` side, so the two
    states cannot be told apart only by the label the writer chose.

    Every rule is registry-driven and derived from the producer's own branch
    condition; none of them is a general "blank the row" policy.
    """
    import numpy as np

    code = None if reg.is_missing(reason) else str(reason)

    def count_of(field):
        value = _cell(row[field])
        if reg.is_missing(value) or not _is_count(value):
            raise StatusSemanticsError(
                f"{relpath}: row {position} reports {field}={value!r} while "
                f"its state is governed by that count; a count is reported in "
                "every state and is a genuine whole number (§G.4)")
        return int(value)

    if code is not None:
        for field in reg.REASON_REQUIRED_ZERO_COUNTS.get((relpath, code), ()):
            observed = count_of(field)
            if observed != 0:
                raise StatusSemanticsError(
                    f"{relpath}: row {position} invokes {code} while reporting "
                    f"{field}={observed}; that reason is emitted exactly when "
                    f"{field} is zero, so the row contradicts its own reason")
        for field in reg.REASON_PREDICTION_DERIVED_FIELDS.get((relpath, code), ()):
            value = _cell(row[field])
            if not reg.is_missing(value):
                raise StatusSemanticsError(
                    f"{relpath}: row {position} invokes {code} — §G.2 defines "
                    f"it as a fold that raised **before producing "
                    f"predictions** — while reporting {field}={value!r}, which "
                    "is computed from those predictions")
        for field in reg.REASON_REQUIRED_POSITIVE_COUNTS.get((relpath, code), ()):
            observed = count_of(field)
            if observed <= 0:
                raise StatusSemanticsError(
                    f"{relpath}: row {position} invokes {code} while reporting "
                    f"{field}={observed}; that reason presupposes a non-empty "
                    "set, and the empty case is a different registered reason")
    elif status == reg.ESTIMABLE:
        for field in reg.ESTIMABLE_REQUIRED_POSITIVE_COUNTS.get(relpath, ()):
            observed = count_of(field)
            if observed <= 0:
                raise StatusSemanticsError(
                    f"{relpath}: row {position} is ESTIMABLE while reporting "
                    f"{field}={observed}; the producer could not have computed "
                    "this row's scientific values from nothing")

    field = reg.CONVERGENCE_FLAG_FIELDS.get(relpath)
    if field is not None:
        flag = _cell(row[field])
        if not isinstance(flag, (bool, np.bool_)):
            raise StatusSemanticsError(
                f"{relpath}: row {position} reports {field}={flag!r} "
                f"({type(flag).__name__}); §F.8 records convergence as a "
                "genuine boolean, never a string and never a 0/1 stand-in")
        flag = bool(flag)
        expected = reg.CONVERGENCE_FLAG_BY_REASON.get((relpath, code))
        if expected is None and status == reg.ESTIMABLE:
            expected = True
        if expected is not None and flag is not expected:
            raise StatusSemanticsError(
                f"{relpath}: row {position} is {status}"
                f"{'' if code is None else ' / ' + code} with {field}={flag}; "
                f"that state is defined by {field}={expected}")


def _validate_status(relpath: str, frame: pd.DataFrame) -> int:
    """Artifact-specific `status` / `reason_code` / degeneracy semantics (§G).

    Three concepts stay separate here exactly as they do in the protocol:
    estimability, degeneracy and class composition. A degenerate row and a
    single-class row are both **ESTIMABLE**; only a genuinely uncomputable
    quantity is `NOT_ESTIMABLE`, and it then carries a registered reason and the
    missing representation — never a zero.
    """
    if relpath in reg.RQ3_STATUS_TABLES:
        return _validate_rq3_status(relpath, frame)
    if relpath in reg.RQ3_CELL_TABLES:
        return _validate_rq3_cell_status(relpath, frame)
    if relpath in reg.SENS_DECISION_TABLES + reg.SENS_EVALUATION_TABLES:
        return _validate_sens_inherited_state(relpath, frame)

    semantics = reg.STATUS_SEMANTICS.get(relpath)
    columns = set(frame.columns)
    if semantics is None:
        if "status" in columns:
            raise StatusSemanticsError(
                f"{relpath} carries a `status` column with no registered "
                "semantics; classify it in stage_f_registry.STATUS_SEMANTICS "
                "before it may be persisted")
        return 0

    value_fields = tuple(semantics["value_fields"])
    estimable_reasons = tuple(semantics["estimable_reasons"])
    interval_fields = tuple(semantics.get("interval_fields", ()))
    unavailable_extra = tuple(reg.NOT_ESTIMABLE_UNAVAILABLE_EXTRA.get(relpath, ()))
    checked = 0
    for position, row in enumerate(frame.to_dict("records")):
        checked += 1
        status = _cell(row["status"])
        reason = _cell(row["reason_code"])
        if status not in reg.STATUS_VALUES:
            raise StatusSemanticsError(
                f"{relpath}: status {status!r} is not one of {reg.STATUS_VALUES}")

        if status == reg.ESTIMABLE:
            if not reg.is_missing(reason):
                # Artifact **and** state: a globally registered code is not
                # permission to carry it here, in this state (§G.2).
                reg.assert_reason_code_for(relpath, reg.ESTIMABLE, str(reason))
                # §D.12.3 — the validity floor invalidates the interval alone.
                for field in interval_fields:
                    if not reg.is_missing(_cell(row[field])):
                        raise StatusSemanticsError(
                            f"{relpath}: {reason} must leave {field} at the "
                            "missing representation while the point estimate "
                            "stays ESTIMABLE")
            for field in value_fields:
                value = _cell(row[field])
                if reg.is_missing(value):
                    raise StatusSemanticsError(
                        f"{relpath}: an ESTIMABLE row leaves the scientific "
                        f"value {field} missing")
                if isinstance(value, (int, float)) and not _finite(value):
                    raise StatusSemanticsError(
                        f"{relpath}: an ESTIMABLE row carries a non-finite "
                        f"{field}={value!r}")
        else:
            if reg.is_missing(reason):
                raise StatusSemanticsError(
                    f"{relpath}: a NOT_ESTIMABLE row carries no reason_code")
            if reason in estimable_reasons:
                raise StatusSemanticsError(
                    f"{relpath}: {reason} is registered here as an "
                    "ESTIMABLE-only reason — it withdraws one derived quantity "
                    "while the row's own scientific value stands — and may not "
                    "be used to declare the whole row NOT_ESTIMABLE (§D.12.3)")
            reg.assert_reason_code_for(relpath, reg.NOT_ESTIMABLE, str(reason))
            # Every scientific output this artifact declares unavailable in this
            # state, not just the headline one: the value fields, the interval
            # endpoints, and the artifact-specific extras (§G.4).
            for field in value_fields + interval_fields + unavailable_extra:
                value = _cell(row[field])
                if not reg.is_missing(value):
                    raise StatusSemanticsError(
                        f"{relpath}: a NOT_ESTIMABLE row serialises {field}="
                        f"{value!r}; an unavailable quantity uses the missing "
                        "representation and never a zero substitute (§G.4)")

        if "degeneracy_code" in columns:
            code = _cell(row["degeneracy_code"])
            flag = _cell(row.get("degeneracy_flag"))
            if code is not None and str(code) not in reg.DEGENERACY_CODES:
                raise StatusSemanticsError(
                    f"{relpath}: {code!r} is not a registered degeneracy code")
            if flag is not None and bool(flag) != (code is not None):
                raise StatusSemanticsError(
                    f"{relpath}: degeneracy_flag={flag!r} disagrees with "
                    f"degeneracy_code={code!r}")
            if code == "ALL_INCLUSIVE_THRESHOLD":
                # §G.3 — degeneracy never implies inestimability.
                if status != reg.ESTIMABLE:
                    raise StatusSemanticsError(
                        f"{relpath}: an ALL_INCLUSIVE_THRESHOLD row is "
                        f"{status}; an all-inclusive threshold leaves every "
                        "registered metric computable and the row ESTIMABLE")
                for name, expected in reg.ALL_INCLUSIVE_METRICS.items():
                    if name in columns and _cell(row[name]) != expected:
                        raise StatusSemanticsError(
                            f"{relpath}: an all-inclusive threshold reports "
                            f"{name}={row[name]!r}, the structural value is "
                            f"{expected}")
        if "class_composition_flag" in columns:
            flag = _cell(row["class_composition_flag"])
            if flag is not None and str(flag) not in reg.CLASS_COMPOSITION_FLAGS:
                raise StatusSemanticsError(
                    f"{relpath}: {flag!r} is not a registered composition flag")
        _validate_reason_state(relpath, position, row, status, reason)
        if relpath == reg.INTERVAL_TABLE:
            _validate_interval_row(relpath, row, status, reason)
    return checked


def _validate_json_payload(relpath: str, payload) -> int:
    """§G for a JSON artifact — a status in a JSON file governs the same way.

    Only the registered status-carrying JSON payloads have a schema of their own;
    the configuration and manifest files are provenance records, checked by the
    serialiser alone. `svm/site_macro_bacc.json` is the one headline SVM number
    (§F.9), so its schema is exact, its scientific value is a genuine finite
    number when it is available at all, and its status decides which fields must
    be present and which must be absent.
    """
    if relpath not in reg.JSON_STATUS_ARTIFACTS:
        return 0
    if not isinstance(payload, dict):
        raise SchemaViolationError(
            f"{relpath}: the payload is a {type(payload).__name__}; the "
            "registered artifact is a JSON object")
    observed = tuple(sorted(payload))
    if observed != reg.SVM_MACRO_JSON_KEYS:
        missing = [k for k in reg.SVM_MACRO_JSON_KEYS if k not in payload]
        extra = [k for k in observed if k not in reg.SVM_MACRO_JSON_KEYS]
        raise SchemaViolationError(
            f"{relpath}: {len(missing)} registered keys missing {missing}, "
            f"{len(extra)} unregistered keys present {extra}; the frozen schema "
            f"is {list(reg.SVM_MACRO_JSON_KEYS)}")

    for name in reg.SVM_MACRO_JSON_STRING_FIELDS:
        value = payload[name]
        if not isinstance(value, str) or not value.strip():
            raise SchemaViolationError(
                f"{relpath}: {name}={value!r} is not a non-empty string")
    if payload["estimand"] != reg.SVM_ESTIMAND:
        raise SchemaViolationError(
            f"{relpath}: estimand={payload['estimand']!r}; the frozen estimand "
            f"is {reg.SVM_ESTIMAND!r}")
    for name in reg.SVM_MACRO_JSON_BOOL_FIELDS:
        if not isinstance(payload[name], bool):
            raise SchemaViolationError(
                f"{relpath}: {name}={payload[name]!r} is not a boolean")
    if payload["carries_confidence_interval"] is not reg.SVM_MACRO_CARRIES_CI:
        raise StatusSemanticsError(
            f"{relpath}: carries_confidence_interval="
            f"{payload['carries_confidence_interval']!r}; §D.12.4 registers the "
            "SVM headline number as carrying no interval")

    checked = 0
    for name in reg.SVM_MACRO_JSON_COUNT_FIELDS:
        value = payload[name]
        if not _is_count(value) or int(value) < 0:
            raise NumericSemanticsError(
                f"{relpath}: {name}={value!r} ({type(value).__name__}) is not a "
                "genuine non-negative integral site count; a scientific count "
                "is never coerced from a string, a bool or an object")
        checked += 1
    contributing = int(payload["n_sites_contributing"])
    not_estimable = int(payload["n_sites_not_estimable"])
    if contributing + not_estimable != reg.BASE_COHORT_SITES:
        raise StatusSemanticsError(
            f"{relpath}: {contributing} contributing + {not_estimable} "
            f"inestimable sites is not the frozen {reg.BASE_COHORT_SITES}-site "
            "cohort")

    status, reason = payload["status"], payload["reason_code"]
    if status not in reg.STATUS_VALUES:
        raise StatusSemanticsError(
            f"{relpath}: status {status!r} is not one of {reg.STATUS_VALUES}")
    value = payload["site_macro_bacc"]
    if status == reg.ESTIMABLE:
        if not reg.is_missing(reason):
            raise StatusSemanticsError(
                f"{relpath}: an ESTIMABLE payload carries reason_code "
                f"{reason!r}; the frozen convention is an empty reason, and "
                f"this artifact registers "
                f"{reg.artifact_reason_codes(relpath, reg.ESTIMABLE) or '(none)'} "
                "for an ESTIMABLE payload")
        if reg.is_missing(value):
            raise StatusSemanticsError(
                f"{relpath}: an ESTIMABLE payload leaves site_macro_bacc "
                "missing")
        if not _is_number(value) or not _finite(value):
            raise NumericSemanticsError(
                f"{relpath}: site_macro_bacc={value!r} "
                f"({type(value).__name__}) is not a genuine finite number; the "
                "headline SVM balanced accuracy is never a string, a bool, an "
                "arbitrary object or an infinity")
        if not 0.0 <= float(value) <= 1.0:
            raise NumericSemanticsError(
                f"{relpath}: site_macro_bacc={value!r} is outside [0, 1]")
        if not_estimable or contributing != reg.BASE_COHORT_SITES:
            raise StatusSemanticsError(
                f"{relpath}: the headline number is ESTIMABLE over "
                f"{contributing} of {reg.BASE_COHORT_SITES} sites; §H.3 refuses "
                "one number computed over a reduced site set")
        checked += 1
    else:
        if reg.is_missing(reason):
            raise StatusSemanticsError(
                f"{relpath}: a NOT_ESTIMABLE payload carries no reason_code")
        reg.assert_reason_code_for(relpath, reg.NOT_ESTIMABLE, str(reason))
        if not reg.is_missing(value):
            raise StatusSemanticsError(
                f"{relpath}: a NOT_ESTIMABLE payload serialises "
                f"site_macro_bacc={value!r}; an unavailable quantity uses the "
                "missing representation and never a substitute (§G.4)")
        if not not_estimable and contributing == reg.BASE_COHORT_SITES:
            raise StatusSemanticsError(
                f"{relpath}: the payload is NOT_ESTIMABLE while all "
                f"{reg.BASE_COHORT_SITES} sites contributed")
    return checked


def _finite(value) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


class JsonSerialisationError(StageFStoreError):
    """A payload held something that is not a JSON scientific value."""


def _json_default(value):
    """The **only** conversion the serialiser performs, and it is not `str`.

    A `default=str` fallback is a silent scientific hazard: it turns an arbitrary
    object into text that looks like a value, so an estimator that returned the
    wrong kind of thing is persisted as an apparently acceptable result. The one
    conversion allowed here is numpy's scalar types to their Python equivalents,
    because those *are* the numbers the estimators produce. Anything else stops
    the write (`reg.JSON_SERIALISATION_COERCES_OBJECTS` is False).
    """
    import math

    import numpy as np

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise JsonSerialisationError(
                f"a non-finite scientific value {value!r} reached JSON "
                "serialisation; the registered missing representation is "
                f"{reg.MISSING_REPRESENTATIONS}")
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    raise JsonSerialisationError(
        f"{value!r} ({type(value).__name__}) is not a JSON scientific value; "
        "the Stage-F serialiser never stringifies an arbitrary object into an "
        f"artifact (reg.JSON_SERIALISATION_COERCES_OBJECTS="
        f"{reg.JSON_SERIALISATION_COERCES_OBJECTS})")


def canonical_json(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=_json_default, allow_nan=False).encode("utf-8")


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    out = subprocess.run(("git", "-C", str(ROOT)) + args,
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def build_provenance(run_kind: str, extra: dict | None = None) -> dict:
    """Everything §I.2 requires a Stage-F store to be bound by."""
    import numpy
    import scipy
    import sklearn

    return {
        "stage": "F",
        "run_kind": run_kind,
        "protocol_version": reg.PROTOCOL_VERSION,
        "protocol_tag": reg.PROTOCOL_TAG,
        "protocol_commit": reg.PROTOCOL_COMMIT,
        "protocol_doc_sha256": reg.EXPECTED_PROTOCOL_DOC_SHA256,
        "constants_sha256": reg.EXPECTED_CONSTANTS_SHA256,
        "protocol_validator_sha256": reg.EXPECTED_PROTOCOL_VALIDATOR_SHA256,
        "implementation_commit": _git("rev-parse", "HEAD"),
        "implementation_tree_clean": _git("status", "--porcelain") == "",
        "consumed_inputs": {p: d for p, d in sorted(reg.CONSUMED_ARTIFACTS.items())},
        "n_consumed_inputs": reg.N_CONSUMED_ARTIFACTS,
        "seeds": {
            "site_bootstrap_namespace": list(reg.BOOTSTRAP_ENTROPY),
            "split_rng_namespace": [20260807, 2000],
            "svm_random_state": reg.SVM_RANDOM_STATE,
            "svm_random_state_status": reg.SVM_RANDOM_STATE_STATUS,
            "conformal_constructions_draw_no_random_number": True,
        },
        "estimator_constants": {
            "alpha_grid": list(reg.ALPHA_GRID),
            "n_grid": list(reg.N_GRID),
            "index_rule": reg.INDEX_RULE,
            "interpolation_permitted": reg.INTERPOLATION_PERMITTED,
            "site_bootstrap_b": reg.SITE_BOOTSTRAP_B,
            "min_valid_replicates": reg.MIN_VALID_REPLICATES,
            "svm_c_grid": list(reg.SVM_C_GRID),
            "svm_pca_candidates": list(reg.SVM_PCA_CANDIDATES),
        },
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "python": sys.version.split()[0]},
        "libraries": {"numpy": numpy.__version__, "scipy": scipy.__version__,
                      "scikit-learn": sklearn.__version__,
                      "pandas": pd.__version__},
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }


# ---------------------------------------------------------------------------
# Formal authority — a capability, not a flag
# ---------------------------------------------------------------------------
class FormalAuthorityError(StageFStoreError):
    """A store claimed Formal completion authority without holding it."""


class _FormalAuthority(NamedTuple):
    """The authority record: **immutable**, and the only one that will exist.

    A `NamedTuple` rather than a class with attributes on purpose. It is a
    `tuple`, so there is no item assignment; it declares `__slots__` through
    `NamedTuple`, so there is no instance dictionary and even
    `object.__setattr__` has nowhere to write. Whatever an attacker can *read*
    about the authorised run, there is nothing here they can *edit* into
    authorising a different object.
    """

    store: object
    root_lexical: str
    root_resolved: str
    issued_utc: str
    pid: int


#: Where the authorised store carries its own copy of the authority record. The
#: attribute is not a token: it is checked for **identity** with the record the
#: factory issued, so planting it on another object proves nothing.
_AUTHORITY_ATTR = "_formal_authority"

#: What may legitimately be reachable from the authority mechanism's own
#: closure. The point is the *absence* of a mutable authorisation container: no
#: dictionary, list, set or custom binding record into which an outside caller
#: could insert a separately allocated store. The barrier re-asserts this before
#: it writes the completion marker, so a future refactor that reintroduces one
#: fails closed rather than silently restoring the hole.
_AUTHORITY_CELL_KINDS = (type(None), _FormalAuthority, type, ModuleType,
                         FunctionType, str, int, bool)


def _make_formal_authority_system():
    """The Formal-authority mechanism — with no authorisation registry at all.

    The previous design kept a mapping of authorised stores inside this closure.
    A mapping is a *capability*: anything that can reach it can insert into it,
    and Python reflection reaches closure cells, so an independently allocated
    store could be written into the mapping and would then verify. Renaming the
    mapping, hiding it one closure deeper or refusing suspicious callable names
    would all have left that capability in place.

    So there is no mapping, no list, no set and no mutable binding record here.
    There is one closure variable holding either `None` or a single immutable
    `_FormalAuthority`. At most one store is authorised at any moment: building
    another genuine store *replaces* that record rather than adding to a
    collection of authorised objects, because there is no collection. The
    factory takes no object from its caller — it builds the store itself — so
    there is no operation anywhere, public or private, that turns a
    caller-supplied object into an authorised one. The record is additionally
    carried by the authorised store itself and cross-checked for identity, so
    neither half is sufficient on its own, and copying a genuine store's state
    onto another object copies a record that still names the genuine store.

    Python reflection can of course still *read* this frame. The requirement is
    a fail-closed normal-API boundary with no insertable authorisation state,
    not cryptographic secrecy (§I.2).
    """
    import dataclasses

    authority = None

    def _create(cls, provenance: dict | None = None):
        nonlocal authority

        # At most one authorised store exists at any moment. Building another
        # genuine store *revokes* the previous record rather than adding to a
        # collection of authorised objects — there is no collection. In a real
        # run this can happen at most once anyway: the second call would find a
        # non-empty authoritative root and stop below.
        lexical = authoritative_root()
        root = assert_root_not_redirected(lexical)    # symlinks refused first
        if root.exists() and any(root.iterdir()):
            raise StaleStoreRootError(
                f"{root} exists and is non-empty; a Formal Stage-F run refuses "
                "to start rather than overwrite, merge, resume or clean an "
                "existing authoritative store (§I.2)")
        # The direct constructor is refused unconditionally (§I.2), so the one
        # authorized store is assembled field by field here. A field added to
        # the dataclass and not to this mapping stops the factory rather than
        # silently producing a half-initialised authoritative store.
        # The eligible-site identities are derived **here**, by verifying the
        # frozen production split registry, and are not a parameter of this
        # factory any more than the root is: there is no argument through which
        # a caller could hand an authoritative run a different eligible set.
        import stage_f_inputs as inputs

        values = {"root": root, "run_kind": reg.RUN_KIND_FORMAL,
                  "provenance": provenance or build_provenance(reg.RUN_KIND_FORMAL),
                  "_written": {}, "_validated": {}, "_cross": {},
                  "_splits": inputs.derive_verified_split_identities()}
        declared = {f.name for f in dataclasses.fields(cls)}
        if declared != set(values):
            raise FormalAuthorityError(
                f"the Formal store declares fields {sorted(declared)}; the "
                f"authority factory initialises {sorted(values)}. The factory "
                "is the only construction path and must be updated with it")
        root.mkdir(parents=True, exist_ok=True)
        store = cls.__new__(cls)
        for name, value in values.items():
            object.__setattr__(store, name, value)
        record = _FormalAuthority(
            store=store, root_lexical=str(lexical), root_resolved=str(root),
            issued_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            pid=os.getpid())
        object.__setattr__(store, _AUTHORITY_ATTR, record)
        authority = record
        _verify(store)
        store.write_json(reg.PROVENANCE_FILE, store.provenance)
        return store

    def _verify(store) -> dict:
        """Prove, from scratch, that *this object* may complete a Formal run."""
        record = authority
        if record is None:
            raise FormalAuthorityError(
                "no Formal store has been authorised in this process; "
                "completion authority is issued by "
                "FormalStageFStore.create_authoritative() alone and by nothing "
                "a caller can invoke on an object of its own (§I.2)")
        if type(record) is not _FormalAuthority:
            raise FormalAuthorityError(
                f"the authority record is a {type(record).__name__}; the "
                "mechanism holds one immutable record and no container an "
                "outside caller could add to")
        if record.store is not store \
                or store.__dict__.get(_AUTHORITY_ATTR) is not record:
            raise FormalAuthorityError(
                "this object is not the store the Formal factory built. "
                "Completion authority is bound to that exact instance by object "
                "identity — it is not conferred by a constructor argument, an "
                "attribute, a copied record, a copied __dict__, an environment "
                "variable, a subclass, a twin at the same root or an alternate "
                "factory (§I.2)")
        if type(store) is not FormalStageFStore:
            raise FormalAuthorityError(
                f"{type(store).__name__} is not the Formal store class; a "
                "subclass may not inherit authoritative completion")
        if store.run_kind != reg.RUN_KIND_FORMAL:
            raise FormalAuthorityError(
                f"run_kind is {store.run_kind!r}, not {reg.RUN_KIND_FORMAL!r}")
        if os.getpid() != record.pid:
            raise FormalAuthorityError(
                f"the authority was issued in process {record.pid} and is being "
                f"used in {os.getpid()}; it does not survive a fork")
        # Blocker B — the symlink question is re-asked on the *lexical* root
        # before any canonical comparison, at every use of the authority.
        lexical = authoritative_root()
        expected = assert_root_not_redirected(lexical)
        if str(lexical) != record.root_lexical \
                or str(expected) != record.root_resolved:
            raise FormalAuthorityError(
                f"the authoritative root is now {lexical} ({expected}); the "
                f"authority was issued for {record.root_lexical} "
                f"({record.root_resolved})")
        observed = Path(store.root)
        if observed.is_symlink() or observed.resolve() != expected:
            raise FormalAuthorityError(
                f"the authoritative root is {expected}; this store is rooted at "
                f"{observed}. A resolved path that merely *matches* is not "
                "trust: no caller-supplied path, relative path, symbolic link "
                "or alternate resolution becomes authoritative")
        return {"authoritative_root": str(expected),
                "authoritative_root_lexical": str(lexical),
                "run_kind": store.run_kind, "authorized_utc": record.issued_utc,
                "bound_to_instance": True,
                "root_symlink_checked_before_resolution": True}

    return _create, _verify


_create_authoritative_store, _verify_formal_authority = \
    _make_formal_authority_system()


#: The authority callables **as this module built them**. The invariant below is
#: a statement about the real mechanism, so it is asserted against the objects
#: the closure returned and not against whatever a module global happens to name
#: at the time — a rebound global is a substituted implementation, a different
#: question from whether the mechanism holds insertable state.
_AUTHORITY_CALLABLES = (_create_authoritative_store, _verify_formal_authority)


def _authority_cell_contents() -> tuple:
    """Everything the authority mechanism's own closure holds, flattened."""
    contents = []
    for holder in _AUTHORITY_CALLABLES:
        for cell in (holder.__closure__ or ()):
            try:
                contents.append(cell.cell_contents)
            except ValueError:                    # an empty cell holds nothing
                continue
        contents.extend(holder.__defaults__ or ())
        contents.extend((holder.__kwdefaults__ or {}).values())
        contents.extend(vars(holder).values())
    return tuple(contents)


def assert_no_authorization_container() -> int:
    """Fail closed if the authority mechanism holds *any* insertable state.

    This is the invariant that replaced the forgeable design, asserted rather
    than assumed. The demonstrated forgery was not a naming problem: it was a
    mutable mapping of authorised stores, reachable through `__closure__`, into
    which a separately allocated store could simply be inserted. The mechanism
    now holds one immutable record and nothing else, and the barrier proves that
    again before it writes the completion marker — so a refactor that
    reintroduces a dictionary, a list, a set or a custom binding record stops the
    run instead of quietly restoring the hole.
    """
    inspected = 0
    for value in _authority_cell_contents():
        inspected += 1
        if isinstance(value, _AUTHORITY_CELL_KINDS):
            continue
        raise FormalAuthorityError(
            f"the Formal-authority mechanism holds a {type(value).__name__}; it "
            "may hold only an immutable authority record and inert references "
            f"{tuple(k.__name__ for k in _AUTHORITY_CELL_KINDS)}. A mutable "
            "container here is an insertable authorisation capability, which is "
            "exactly the forgery this design exists to remove (§I.2)")
    return inspected


# ---------------------------------------------------------------------------
# the shared writer layer
# ---------------------------------------------------------------------------
class CrossArtifactError(StageFStoreError):
    """Two artifacts persist the same scientific fact and disagree about it."""


def _cross_bucket(relpath: str):
    """Which cross-file consistency group an artifact belongs to, or `None`."""
    parts = relpath.split("/")
    if len(parts) == 3 and parts[0] == reg.NAMESPACE_SENSITIVITIES \
            and parts[1] in reg.SENS56_APPLIES_TO \
            and parts[2] in (reg.SENS56_CELLS_FILE,
                             reg.SENS56_VALIDATION_UNITS_FILE,
                             reg.SENS56_FAMILIES_FILE):
        return ("sens56", parts[1])
    if relpath in (reg.SVM_PER_SITE_TABLE, reg.SVM_MACRO_JSON,
                   reg.SVM_FOLD_SELECTION_TABLE):
        return ("svm",)
    if relpath in reg.RQ4_SITE_CELL_TABLES:
        return ("rq4",)
    return None


#: Every cross-file group, and the artifacts that must all have been offered
#: before the group's agreement has actually been proved. Checked at the barrier
#: so a group cannot be satisfied by writing only one of its members.
CROSS_ARTIFACT_GROUPS = {
    # §F.7.1 / §F.7.1a / §F.7.2 / §H.3.1 — the candidate table, the fold table
    # and the headline are three views of one F-SVM run, and the fold-level
    # verdict is *derived* from the candidate rows. Attempt #5 was sealable
    # precisely because the three were validated independently.
    ("svm",): (reg.SVM_PER_SITE_TABLE, reg.SVM_MACRO_JSON,
               reg.SVM_FOLD_SELECTION_TABLE),
    ("rq4",): tuple(reg.RQ4_SITE_CELL_TABLES),
    **{("sens56", label): tuple(
        f"{reg.NAMESPACE_SENSITIVITIES}/{label}/{name}"
        for name in (reg.SENS56_CELLS_FILE, reg.SENS56_VALIDATION_UNITS_FILE,
                     reg.SENS56_FAMILIES_FILE))
       for label in reg.SENS56_APPLIES_TO},
}


def _record_cross(state: dict, relpath: str, payload) -> None:
    """Cache one artifact's cross-bound projection and re-check its group.

    Duplicated persisted science is a **consistency** problem, not six
    independent files: `validation_units.csv` and the nested `blocks` column of
    `cells.csv` are two transcriptions of the same `BlockResult`; `families.csv`
    re-serialises each block's own raw p-value; `svm/site_macro_bacc.json`
    counts the rows of `svm/per_site_bacc.csv`; and the two RQ4 site x cell
    tables describe the same eligible site set. Whichever member arrives second
    re-runs the comparison, so no write order can skip it.
    """
    bucket = _cross_bucket(relpath)
    if bucket is None:
        return
    group = state.setdefault(bucket, {})
    group[relpath] = payload
    if bucket[0] == "sens56":
        _check_sens56_group(bucket[1], group)
    elif bucket[0] == "svm":
        _check_svm_group(group)
    else:
        _check_rq4_group(group)


def _project_cross(relpath: str, obj):
    """The minimal projection of an artifact the cross-file checks need."""
    parts = relpath.split("/")
    if relpath == reg.SVM_MACRO_JSON:
        return dict(obj)
    rows = obj.to_dict("records")
    if relpath == reg.SVM_PER_SITE_TABLE:
        return {str(_cell(r[reg.SVM_SITE_FIELD])): {
            field: _cell(r[field]) for field in
            ("status", "reason_code", "bacc", "selected_pca", "selected_C",
             "inner_macro_bacc", "converged", "n_iter_max")}
            for r in rows}
    if relpath == reg.SVM_FOLD_SELECTION_TABLE:
        candidates = {}
        for r in rows:
            key = (_svm_pca_cell(r["pca"]), float(_cell(r["C"])))
            candidates.setdefault(
                str(_cell(r[reg.SVM_SITE_FIELD])), {})[key] = {
                    "status": _cell(r["status"]),
                    "reason_code": _cell(r["reason_code"]),
                    reg.SVM_CANDIDATE_VALUE_FIELD:
                        _cell(r[reg.SVM_CANDIDATE_VALUE_FIELD])}
        return candidates
    if relpath in reg.RQ4_SITE_CELL_TABLES:
        per_repeat = relpath == "rq4/per_repeat_target_local.csv"
        sites = {}
        for r in rows:
            method = (reg.RQ4_PER_REPEAT_METHOD if per_repeat
                      else str(_cell(r["method"])))
            if method != reg.METHOD_TARGET_LOCAL:
                continue
            n_labels = _cell(r["n_labels"])
            sites.setdefault(int(n_labels), set()).add(
                str(_cell(r[reg.RQ4_SITE_FIELD])))
        return sites
    name = parts[2]
    if name == reg.SENS56_CELLS_FILE:
        out = {}
        for position, r in enumerate(rows):
            key = (str(_cell(r["population"])), str(_cell(r["operating_point"])),
                   str(_cell(r["method"])))
            out[key] = _as_mapping(_cell(r[reg.SENS56_CELLS_BLOCKS_FIELD])) or {}
        return out
    if name == reg.SENS56_VALIDATION_UNITS_FILE:
        return {(str(_cell(r["population"])), str(_cell(r["operating_point"])),
                 str(_cell(r["method"])), str(_cell(r["block"]))):
                {"status": _cell(r["status"]),
                 "failing_condition": _cell(r["failing_condition"]),
                 "numerical_validation": _cell(r["numerical_validation"])}
                for r in rows}
    return {(str(_cell(r["population"])), str(_cell(r["block"]))):
            {"slots": _cell(r[reg.SENS56_FAMILIES_SLOTS_FIELD]),
             reg.SENS56_FAMILY_NOT_ESTIMABLE_FIELD:
                 _cell(r[reg.SENS56_FAMILY_NOT_ESTIMABLE_FIELD])}
            for r in rows}


def _check_sens56_group(label: str, group: dict) -> None:
    """`cells.csv` <-> `validation_units.csv` <-> `families.csv` (§H.4.3)."""
    import rq3_protocol as proto

    stem = f"{reg.NAMESPACE_SENSITIVITIES}/{label}"
    cells = group.get(f"{stem}/{reg.SENS56_CELLS_FILE}")
    units = group.get(f"{stem}/{reg.SENS56_VALIDATION_UNITS_FILE}")
    families = group.get(f"{stem}/{reg.SENS56_FAMILIES_FILE}")

    if cells is not None and units is not None:
        expected = {key + (block,) for key, blocks in cells.items()
                    for block in blocks}
        if set(units) != expected:
            missing = sorted(expected - set(units))[:3]
            extra = sorted(set(units) - expected)[:3]
            raise CrossArtifactError(
                f"{stem}: the validation units are not the cells' own blocks "
                f"({len(expected - set(units))} missing, e.g. {missing}; "
                f"{len(set(units) - expected)} with no cell, e.g. {extra}). "
                "§H.4.3 makes the unit key the cell key plus `block`, and both "
                "files transcribe the same `BlockResult`")
        for key, blocks in sorted(cells.items()):
            for block, record in sorted(blocks.items()):
                unit = units[key + (block,)]
                for field in reg.SENS56_BLOCK_UNIT_SHARED_FIELDS:
                    # `_project_cross` already collapsed pandas' NA to None on
                    # the unit side; `NOT_ESTIMABLE` is a *status* here, not a
                    # missing marker, so the two are compared as plain values.
                    left, right = record[field], unit[field]
                    if (left is None) != (right is None) or \
                            (left is not None and str(left) != str(right)):
                        raise CrossArtifactError(
                            f"{stem}: unit {key + (block,)} records "
                            f"{field}={right!r} while the same block inside "
                            f"{reg.SENS56_CELLS_FILE} records {left!r}; one "
                            "`BlockResult` cannot have two states")
                left = record["numerical_validation"]
                right = _as_mapping(unit["numerical_validation"])
                expected_right = (left if left
                                  else _canonical_empty_verdict(
                                      record["failing_condition"]))
                if right != expected_right:
                    raise CrossArtifactError(
                        f"{stem}: unit {key + (block,)} records "
                        f"numerical_validation={right!r}; the same block "
                        f"inside {reg.SENS56_CELLS_FILE} gives "
                        f"{expected_right!r} "
                        f"({'the parent-equivalent canonical empty verdict' if not left else 'the same verdict'})")

    if cells is not None and families is not None:
        for (population, block), family in sorted(families.items()):
            slots = family["slots"]
            if isinstance(slots, str):
                slots = json.loads(slots)
            if not isinstance(slots, list) \
                    or len(slots) != reg.SENS56_HOLM_FAMILY_SIZE:
                raise CrossArtifactError(
                    f"{stem}: family {(population, block)} holds "
                    f"{slots!r}; §12.3 fixes {reg.SENS56_HOLM_FAMILY_SIZE} "
                    "pre-registered slots, one per cell")
            for slot, (operating_point, method) in zip(slots,
                                                       proto.CELL_REGISTRY):
                if tuple(sorted(slot)) != reg.SENS56_FAMILY_SLOT_KEYS:
                    raise CrossArtifactError(
                        f"{stem}: family {(population, block)} slot "
                        f"{sorted(slot)} is not the parent's "
                        f"{list(reg.SENS56_FAMILY_SLOT_KEYS)}")
                if (slot["operating_point"], slot["method"]) != \
                        (operating_point, method):
                    raise CrossArtifactError(
                        f"{stem}: family {(population, block)} slot "
                        f"{slot['cell_index']} names "
                        f"({slot['operating_point']}, {slot['method']}); the "
                        f"parent's `CELL_REGISTRY` order puts "
                        f"({operating_point}, {method}) there")
                record = cells.get((population, operating_point, method), {})
                block_record = record.get(block)
                if block_record is None:
                    raise CrossArtifactError(
                        f"{stem}: family {(population, block)} has a slot for "
                        f"cell ({operating_point}, {method}), which carries no "
                        f"{block!r} block")
                left = block_record[reg.SENS56_BLOCK_PVALUE_FIELD]
                right = slot[reg.SENS56_FAMILY_PVALUE_FIELD]
                if isinstance(left, str) or isinstance(right, str):
                    same = str(left) == str(right)
                else:
                    same = float(left) == float(right)
                if not same:
                    raise CrossArtifactError(
                        f"{stem}: family {(population, block)} slot "
                        f"({operating_point}, {method}) carries "
                        f"p_raw={right!r}; the block it is drawn from records "
                        f"lrt_pvalue_raw={left!r}. §12.3 takes the family's "
                        "raw p-values from the blocks themselves")
            declared = family[reg.SENS56_FAMILY_NOT_ESTIMABLE_FIELD]
            observed = sum(1 for slot in slots
                           if slot[reg.SENS56_FAMILY_PVALUE_FIELD]
                           == proto.NOT_ESTIMABLE)
            if reg.is_missing(declared) or int(declared) != observed:
                raise CrossArtifactError(
                    f"{stem}: family {(population, block)} declares "
                    f"{reg.SENS56_FAMILY_NOT_ESTIMABLE_FIELD}={declared!r} "
                    f"while {observed} of its slots are "
                    f"{proto.NOT_ESTIMABLE}")


def _svm_pca_cell(value):
    """One PCA selection, in the frozen `{"no-PCA", 50, 100, 200}` spelling.

    pandas upcasts an integer column to `float64` as soon as one row of the
    table is missing, so the *same* selection reaches this comparison as `100`
    from one artifact and `100.0` from another. `100.0` is a storage kind, not a
    different model-selection state (the Stage F.4 lesson), so both normalise
    here rather than being refused or compared as strings.
    """
    cell = _cell(value)
    if isinstance(cell, str):
        return cell
    return int(cell)


def _svm_recompute_selection(candidates: dict):
    """§F.7.5's deterministic argmax, recomputed from the persisted rows.

    The store never trusts the producer's own answer about which candidate it
    picked: it re-derives the argmax from the 20 persisted scores and compares.
    The order is read from the registry — highest score, then smaller `C`, then
    stronger compression with `no-PCA` **last** — so a runtime whose tie-break
    drifted back to the `1.0.2` order is caught at write time rather than after
    a Formal run.
    """
    c_order = list(reg.SVM_C_GRID)
    pca_order = [("no-PCA" if p is None else int(p))
                 for p in reg.SVM_TIE_BREAK_PCA_ORDER]

    def rank(key):
        pca, penalty = key
        return (-float(candidates[key][reg.SVM_CANDIDATE_VALUE_FIELD]),
                c_order.index(penalty), pca_order.index(pca))

    return min(candidates, key=rank)


def _check_svm_group(group: dict) -> None:
    """§F.7 / §H.3 — the three F-SVM artifacts are one statement in three files.

    Attempt #5 was mechanically sealable because each SVM artifact was checked
    on its own: a fold could report `ESTIMABLE` with a reported balanced
    accuracy while its own candidate rows recorded that a fit had warned, and
    nothing compared the two. These rules close that gap, and they **derive**
    the fold-level verdict from the candidate rows rather than accepting the
    producer's label for it:

    * a fold is `ESTIMABLE` only if **all 20** of its candidates are, so a
      reduced-grid selection is unrepresentable (§F.7.2);
    * any candidate `SVM_NOT_CONVERGED` forces the §F.7.1 inner-grid row —
      `NOT_ESTIMABLE` / `SVM_NOT_CONVERGED`, `converged = false`, null
      selection, null BAcc, and `n_iter_max` null because no refit was
      attempted;
    * an expected candidate `SVM_FOLD_FAILED` forces the same shape under its
      own code, unless a warning elsewhere in the fold takes precedence (§G.2a);
    * a `NOT_ESTIMABLE` fold whose 20 candidates are all `ESTIMABLE` is the
      §F.7.1a final-refit case, and must **retain** the selection metadata and
      record `n_iter_max = 2000` — the two representations are never
      interchangeable;
    * the selected tuple must equal the argmax recomputed from the complete
      20-candidate grid (§F.7.5);
    * the headline counts partition the fold table (§H.3).
    """
    per_site = group.get(reg.SVM_PER_SITE_TABLE)
    macro = group.get(reg.SVM_MACRO_JSON)
    folds = group.get(reg.SVM_FOLD_SELECTION_TABLE)

    if per_site is not None and macro is not None:
        estimable = sum(1 for row in per_site.values()
                        if row["status"] == reg.ESTIMABLE)
        failed = len(per_site) - estimable
        for field, observed in ((reg.SVM_MACRO_CONTRIBUTING_FIELD, estimable),
                                (reg.SVM_MACRO_NOT_ESTIMABLE_FIELD, failed)):
            if int(macro[field]) != observed:
                raise CrossArtifactError(
                    f"{reg.SVM_MACRO_JSON}: {field}={macro[field]!r} while "
                    f"{reg.SVM_PER_SITE_TABLE} holds {estimable} ESTIMABLE and "
                    f"{failed} NOT_ESTIMABLE folds; "
                    "`stage_f_svm.site_macro_bacc` counts that very table "
                    "(§H.3)")

    if per_site is None or folds is None:
        return

    if set(per_site) != set(folds):
        raise CrossArtifactError(
            f"{reg.SVM_PER_SITE_TABLE} holds folds {sorted(set(per_site))} "
            f"while {reg.SVM_FOLD_SELECTION_TABLE} holds "
            f"{sorted(set(folds))}; the two tables describe the same 23 outer "
            "folds (§H.3)")

    for site in sorted(per_site):
        row, candidates = per_site[site], folds[site]
        where = f"{reg.SVM_PER_SITE_TABLE}/{reg.SVM_FOLD_SELECTION_TABLE}: {site}"

        if len(candidates) != reg.SVM_N_CANDIDATES_PER_FOLD:
            raise CrossArtifactError(
                f"{where} carries {len(candidates)} candidate rows; §F.7.2 "
                f"makes all {reg.SVM_N_CANDIDATES_PER_FOLD} mandatory, "
                "including every row of an invalidated fold")

        warned = sorted(k for k, c in candidates.items()
                        if c["reason_code"] == "SVM_NOT_CONVERGED")
        expected_failed = sorted(k for k, c in candidates.items()
                                 if c["reason_code"] == "SVM_FOLD_FAILED")
        unusable = sorted(k for k, c in candidates.items()
                          if c["status"] != reg.ESTIMABLE)
        status, reason = row["status"], row["reason_code"]

        if status == reg.ESTIMABLE and unusable:
            raise CrossArtifactError(
                f"{where} is ESTIMABLE while {len(unusable)} of its "
                f"{reg.SVM_N_CANDIDATES_PER_FOLD} candidates are not, e.g. "
                f"{unusable[:3]}. §F.7.2 permits selection only over the "
                "complete valid grid, and §F.7.1 invalidates the whole outer "
                "fold as soon as any fit in its lifecycle warns — a "
                "reduced-grid selection is not a lawful state")

        if unusable:
            # §F.7.1 — the inner-grid case, whichever code drove it. No lawful
            # selection occurred, so writing a value into any of the four
            # selection fields would assert an event that did not happen, and
            # the refit was skipped rather than attempted.
            expected_reason = ("SVM_NOT_CONVERGED" if warned
                               else "SVM_FOLD_FAILED")
            if status != reg.NOT_ESTIMABLE or reason != expected_reason:
                raise CrossArtifactError(
                    f"{where} is {status}"
                    f"{'' if reason is None else ' / ' + str(reason)} while "
                    f"{len(warned)} candidates warned and "
                    f"{len(expected_failed)} raised a permitted estimator "
                    f"failure; §F.7.1 fixes the fold at NOT_ESTIMABLE / "
                    f"{expected_reason} (§G.2a makes SVM_NOT_CONVERGED the "
                    "more specific code whenever a warning occurred)")
            for field in ("bacc", "selected_pca", "selected_C",
                          reg.SVM_CANDIDATE_VALUE_FIELD, "n_iter_max"):
                if not reg.is_missing(row[field]):
                    raise CrossArtifactError(
                        f"{where} took the §F.7.1 inner-grid path and still "
                        f"reports {field}={row[field]!r}. No lawful selection "
                        "occurred and no final refit was attempted, so that "
                        "field records an event that did not happen")
            if bool(row["converged"]) is not False:
                raise CrossArtifactError(
                    f"{where} reports converged={row['converged']!r} while an "
                    "inner-grid fit did not converge (§F.7.3 condition 1)")
            continue

        recomputed = _svm_recompute_selection(candidates)
        selected = (row["selected_pca"], row["selected_C"])

        if status == reg.NOT_ESTIMABLE:
            # §F.7.1a — every candidate is ESTIMABLE, so the grid was complete
            # and the selection lawful; only the selected final refit failed.
            # The evidence is real and is retained, and `n_iter_max` is the
            # configured cap because the refit *was* attempted.
            if not reg.is_missing(row["bacc"]):
                raise CrossArtifactError(
                    f"{where} is NOT_ESTIMABLE and still reports "
                    f"bacc={row['bacc']!r}")
            if bool(row["converged"]) is not False:
                raise CrossArtifactError(
                    f"{where} is NOT_ESTIMABLE with converged="
                    f"{row['converged']!r}; §F.7.3 condition 3 forbids it")
            for field in ("selected_pca", "selected_C",
                          reg.SVM_CANDIDATE_VALUE_FIELD):
                if reg.is_missing(row[field]):
                    raise CrossArtifactError(
                        f"{where} has a complete valid 20-candidate grid, so a "
                        f"lawful selection occurred, yet {field} is missing. "
                        "§F.7.1a **retains** the selection metadata; the "
                        "all-null §F.7.1 row is the *other* case and may not "
                        "be written here")
            if int(row["n_iter_max"]) != reg.SVM_N_ITER_MAX_WHEN_REFIT_ATTEMPTED:
                raise CrossArtifactError(
                    f"{where} reports n_iter_max={row['n_iter_max']!r}; the "
                    "final refit was attempted, and §F.7.4 records the "
                    f"configured cap {reg.SVM_N_ITER_MAX_WHEN_REFIT_ATTEMPTED} "
                    "whether or not that attempt warned")
        else:
            if bool(row["converged"]) is not True:
                raise CrossArtifactError(
                    f"{where} is ESTIMABLE with converged="
                    f"{row['converged']!r}; §F.7.3 makes the three conditions "
                    "inseparable")
            if int(row["n_iter_max"]) != reg.SVM_N_ITER_MAX_WHEN_REFIT_ATTEMPTED:
                raise CrossArtifactError(
                    f"{where} is ESTIMABLE with n_iter_max="
                    f"{row['n_iter_max']!r}; §F.7.4 records the configured cap "
                    f"{reg.SVM_N_ITER_MAX_WHEN_REFIT_ATTEMPTED}")

        # §F.7.5 — the selected tuple and its retained inner score are checked
        # against the argmax recomputed from the persisted grid, in both the
        # ESTIMABLE and the §F.7.1a state.
        if (_svm_pca_cell(selected[0]), float(selected[1])) != (
                _svm_pca_cell(recomputed[0]), float(recomputed[1])):
            raise CrossArtifactError(
                f"{where} selected (pca={selected[0]!r}, C={selected[1]!r}) "
                f"while the deterministic §F.7.5 argmax over the complete "
                f"{reg.SVM_N_CANDIDATES_PER_FOLD}-candidate grid is "
                f"(pca={recomputed[0]!r}, C={recomputed[1]!r}). The order is "
                "highest inner macro site-BAcc, then smaller C, then stronger "
                "compression — `no-PCA` is the weakest and therefore last")
        chosen = float(candidates[recomputed][reg.SVM_CANDIDATE_VALUE_FIELD])
        if not _finite(float(row[reg.SVM_CANDIDATE_VALUE_FIELD])) \
                or float(row[reg.SVM_CANDIDATE_VALUE_FIELD]) != chosen:
            raise CrossArtifactError(
                f"{where} reports inner_macro_bacc="
                f"{row[reg.SVM_CANDIDATE_VALUE_FIELD]!r} while its selected "
                f"candidate row holds {chosen!r}; the per-site field is that "
                "candidate's own score, not a second quantity")


def _check_rq4_group(group: dict) -> None:
    """The two RQ4 site x cell tables describe one eligible site set per `N`."""
    per_site = group.get("rq4/per_site_metrics.csv")
    per_repeat = group.get("rq4/per_repeat_target_local.csv")
    if per_site is None or per_repeat is None:
        return
    if set(per_site) != set(per_repeat):
        raise CrossArtifactError(
            f"rq4: per_site_metrics evaluates target_local at N="
            f"{sorted(per_site)} and per_repeat_target_local at "
            f"N={sorted(per_repeat)}")
    for n_labels in sorted(per_site):
        if per_site[n_labels] != per_repeat[n_labels]:
            difference = sorted(per_site[n_labels] ^ per_repeat[n_labels])
            raise CrossArtifactError(
                f"rq4: at N={n_labels} the two target-local tables disagree "
                f"about which sites have a cell ({difference}); eligibility is "
                "one frozen fact, not a per-table choice (§G.2)")


def _assert_split_identity_binding(store) -> dict:
    """Re-prove the split-registry identity binding at the completion barrier.

    Three separate things, none of which the write-time check alone gives:

    1. this store still holds a binding, and it is still the immutable record
       the verification produced;
    2. the frozen production split registry is **re-verified from disk**, with
       the memo bypassed, and admits exactly the identities the binding carries
       — so a binding mutated after construction, or derived from a registry
       that has since moved, stops the run;
    3. every registered artifact keyed on those identities actually went through
       the write-time comparison, and the identities it recorded are the ones
       re-derived here. An artifact that was never governed is a hole, so the
       coverage is checked against the registry rather than against whatever
       happens to be in `_validated`.
    """
    import stage_f_inputs as inputs

    binding = _verified_identities(getattr(store, "_splits", None),
                                   "the completion barrier")
    fresh = inputs.derive_verified_split_identities(use_cache=False)
    if fresh.sites_by_n != binding.sites_by_n \
            or fresh.registry_sha256 != binding.registry_sha256:
        raise SplitIdentityError(
            "the production split registry re-verified at the barrier admits "
            f"{[(n, len(v)) for n, v in fresh.sites_by_n]} "
            f"({fresh.registry_sha256[:12]}...), the binding this run wrote its "
            f"artifacts against carries {binding.registry_sha256[:12]}...; an "
            "authoritative store may not be sealed against identities other "
            "than the ones its own inputs were verified to hold")

    governed = sorted(reg.SPLIT_IDENTITY_TABLES)
    covered = []
    for relpath in governed:
        report = store._validated.get(relpath)
        if report is None:
            raise SplitIdentityError(
                f"{relpath} carries the split registry's eligible-site "
                "identities and never passed the write-time identity "
                "comparison; it is not sealable")
        recorded = dict(report.get("split_identity") or {})
        if not recorded:
            raise SplitIdentityError(
                f"{relpath} was validated without an eligible-site identity "
                "record; the comparison is not optional for a governed table")
        spec = reg.split_identity_spec(relpath)
        expected = tuple(
            (int(n), tuple(sorted(reg.expected_identity_sites(
                spec, int(n), fresh.sites_at(int(n))))))
            for n in sorted(spec["n_values"]))
        if tuple(recorded["sites_by_n"]) != expected:
            raise SplitIdentityError(
                f"{relpath}: the identities it was written against are not the "
                "ones the re-verified split registry admits "
                f"({recorded['sites_by_n']} vs {expected})")
        covered.append(relpath)

    return {"registry_path": fresh.registry_path,
            "registry_sha256": fresh.registry_sha256,
            "registry_rows_verified": fresh.n_rows,
            "registry_cells_verified": fresh.n_cells_verified,
            "eligible_sites_by_n": {str(n): list(v) for n, v in fresh.sites_by_n},
            "n_identity_bound_artifacts": len(covered),
            "identity_bound_artifacts": covered,
            "rederived_at_barrier_without_cache": True}


def _assert_cross_bindings_complete(state: dict) -> int:
    """Every cross-file group actually had all its members offered."""
    proved = 0
    for bucket, members in sorted(CROSS_ARTIFACT_GROUPS.items()):
        offered = set(state.get(bucket, {}))
        missing = [m for m in members if m not in offered]
        if missing:
            raise CrossArtifactError(
                f"the {bucket} consistency group was never proved: "
                f"{missing} were not offered through the write barrier")
        proved += 1
    return proved


@dataclass
class _StageFRecordStore:
    """Registered artifacts only, written once, digested as they are written."""

    root: Path
    run_kind: str
    provenance: dict = field(default_factory=dict)
    _written: dict = field(default_factory=dict, repr=False)
    _validated: dict = field(default_factory=dict, repr=False)
    #: cross-file projections of the artifacts offered so far (§H.4.3, §H.3)
    _cross: dict = field(default_factory=dict, repr=False)
    #: the run's own `stage_f_inputs.VerifiedSplitIdentities`, or `None`. A
    #: store without one cannot persist any site x `N` identity (§D.9).
    _splits: object = None

    # -- writers ------------------------------------------------------------
    def _guard(self, relpath: str) -> Path:
        reg.assert_registered_artifact(relpath)
        if relpath == reg.COMPLETION_FILE:
            raise StageFStoreError(
                "COMPLETION.json is written by the barrier's terminal step "
                "alone, never through the ordinary writer (§I.1 step 7)")
        if relpath == reg.CHECKSUM_FILE:
            raise StageFStoreError(
                f"{reg.CHECKSUM_FILE} is written by the barrier's own ledger "
                "step alone, over the artifacts actually persisted; it is never "
                "a caller-supplied artifact (§I.1 step 2)")
        if relpath in self._written:
            raise StageFStoreError(f"{relpath} was already written this run")
        return self.root / relpath

    def write_table(self, relpath: str, rows, columns=None) -> str:
        """Persist one registered table through the single validation barrier.

        All three write-time conditions must hold before any byte reaches disk:
        the **exact** registered schema, **unique** registered logical keys and
        coherent **status / reason / degeneracy** semantics — on top of the
        frozen cardinality. There is no second write path, so no module can
        persist a table that skipped them.
        """
        path = self._guard(relpath)
        expected = reg.expected_rows(relpath)
        rows = list(rows)
        if expected is not None and len(rows) != expected:
            raise StageFStoreError(
                f"{relpath} holds {len(rows)} rows, the frozen registry "
                f"cardinality is {expected}")
        frame = pd.DataFrame(rows)
        schema = reg.table_columns(relpath)
        if columns is not None and schema is not None \
                and tuple(columns) != tuple(schema):
            raise SchemaViolationError(
                f"{relpath}: the caller offered a column order that is not the "
                "frozen registry order; the registry is the only authority")
        if schema is not None and not frame.empty:
            # Reorder to the frozen order when — and only when — the table
            # presents exactly the frozen column set; a mismatch is rejected by
            # `_validate_schema` rather than repaired by projection.
            if set(frame.columns) == set(schema):
                frame = frame[list(schema)]
        # Every condition is proved *before* a byte is written, and the artifact
        # is recorded as validated only once all of them have passed. Writing
        # bytes is never itself evidence of validation (§I.1).
        report = {
            "kind": "csv",
            "columns": _validate_schema(relpath, frame),
            "logical_key": _validate_logical_key(relpath, frame),
            "numeric_cells_checked": _validate_numeric(relpath, frame),
            "closed_domain_cells_checked": _validate_closed_domains(relpath,
                                                                    frame),
            "site_identifiers_checked": _validate_site_identifiers(relpath, frame),
            "structural_cells_checked": _validate_structural_cells(relpath, frame),
            "split_identity": _validate_split_identity(self._splits, relpath,
                                                       frame),
            "status_rows_checked": _validate_status(relpath, frame),
            "n_rows": int(len(frame)),
            "validated": True,
        }
        # Cross-file agreement is proved **before** the bytes are written, so a
        # table that disagrees with a sibling already on disk never reaches it.
        if _cross_bucket(relpath) is not None:
            _record_cross(self._cross, relpath, _project_cross(relpath, frame))
        digest = _atomic_write(path, frame.to_csv(index=False).encode("utf-8"),
                               fsync_up_to=self.root)
        self._written[relpath] = digest
        self._validated[relpath] = report
        return digest

    def write_json(self, relpath: str, payload) -> str:
        """Persist one registered JSON artifact through the same barrier.

        A JSON file is not a weaker artifact than a table: a registered
        status-carrying payload passes an exact schema, genuinely numeric
        scientific values and the same §G status semantics before any byte is
        written, and the serialiser refuses to stringify an arbitrary object or
        emit a non-finite number in *any* payload.
        """
        path = self._guard(relpath)
        checked = _validate_json_payload(relpath, payload)
        if _cross_bucket(relpath) is not None:
            _record_cross(self._cross, relpath, _project_cross(relpath, payload))
        blob = canonical_json(payload)     # NaN / Infinity / objects refused
        digest = _atomic_write(path, blob, fsync_up_to=self.root)
        self._written[relpath] = digest
        self._validated[relpath] = {"kind": "json", "columns": (),
                                    "logical_key": (),
                                    "numeric_cells_checked": checked,
                                    "closed_domain_cells_checked": 0,
                                    "site_identifiers_checked": 0,
                                    "structural_cells_checked": 0,
                                    "split_identity": {},
                                    "status_rows_checked": 0, "n_rows": None,
                                    "validated": True}
        return digest

    # -- inspection ---------------------------------------------------------
    @property
    def n_artifacts_written(self) -> int:
        """How many registered artifacts this run has persisted so far."""
        return len(self._written)

    def relpaths_on_disk(self) -> tuple:
        if not self.root.exists():
            return ()
        return tuple(sorted(
            str(p.relative_to(self.root)) for p in self.root.rglob("*")
            if p.is_file()))

    def write_checksum_ledger(self) -> str:
        """Step 2 — `RESULT_CHECKSUMS.sha256` over every artifact written.

        The ledger is an artifact like any other: its bytes are fsynced and its
        directory entry is fsynced before anything rereads it. An in-memory
        ledger is never treated as written.
        """
        lines = [f"{digest}  {relpath}"
                 for relpath, digest in sorted(self._written.items())
                 if relpath not in reg.LEDGER_EXCLUDED]
        path = self.root / reg.CHECKSUM_FILE
        digest = _atomic_write(path, ("\n".join(lines) + "\n").encode("utf-8"),
                               fsync_up_to=self.root)
        self._written[reg.CHECKSUM_FILE] = digest
        return digest

    def read_checksum_ledger(self, *, require_exact: bool = True) -> dict:
        """Reread the ledger **from disk**, parse it strictly, verify it fully.

        The parser is adversarial by design. A checksum ledger is the store's own
        statement about itself, so a line that is merely *parseable* is not
        enough: an absolute path, a `../` component, a repeated entry, a
        self-entry, a path that is not exactly one registered relative path, or a
        digest that is not 64 hexadecimal characters is a corrupt ledger, and a
        corrupt ledger may never carry a run to completion.

        The two frozen exclusions are structural, not stylistic: the ledger
        cannot checksum itself (its own bytes are not yet final while it is being
        written), and `COMPLETION.json` does not exist at step 2 and must not be
        pre-announced. Every remaining registered artifact must appear exactly
        once, and each is re-checked against a digest recomputed from disk.
        """
        path = self.root / reg.CHECKSUM_FILE
        if not path.exists():
            raise CompletionBarrierError(
                f"{reg.CHECKSUM_FILE} is not on disk; the ledger must be "
                "written and fsynced before it is reread (§I.1 steps 2-4)")
        expected = set(reg.expected_checksum_paths())
        root = self.root.resolve()
        hexdigits = set("0123456789abcdef")
        recorded: dict = {}
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            digest, separator, name = line.partition("  ")
            if not separator or not name or not digest:
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} is malformed: {line!r}")
            if len(digest) != 64 or not set(digest) <= hexdigits:
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} carries {digest!r}, "
                    "which is not a sha256 digest")
            if name != name.strip() or name != name.replace("\\", ""):
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} names {name!r}, which "
                    "is not a plain relative path")
            if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} names the absolute path "
                    f"{name!r}; the ledger records store-relative paths only")
            if any(part in ("", ".", "..") for part in name.split("/")):
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} names {name!r}, which "
                    "carries an empty or traversing path component")
            if name == reg.CHECKSUM_FILE:
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} checksums the ledger "
                    "itself; the ledger is never self-referential "
                    f"(reg.LEDGER_COVERS_ITSELF={reg.LEDGER_COVERS_ITSELF})")
            if name == reg.COMPLETION_FILE:
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} names "
                    f"{reg.COMPLETION_FILE}, which is terminal and written "
                    "after the ledger is sealed "
                    f"(reg.LEDGER_COVERS_COMPLETION="
                    f"{reg.LEDGER_COVERS_COMPLETION})")
            if name in recorded:
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} repeats {name!r}; an "
                    "artifact is ledgered exactly once")
            if name not in expected:
                raise CompletionBarrierError(
                    f"{reg.CHECKSUM_FILE} line {number} names {name!r}, which "
                    f"is not one of the {len(expected)} registered "
                    "checksum-covered artifacts (§H.1)")
            recorded[name] = digest

        if require_exact and set(recorded) != expected:
            missing = sorted(expected - set(recorded))
            raise CompletionBarrierError(
                f"the ledger covers {len(recorded)} of {len(expected)} "
                f"registered artifacts; missing {missing[:5]}")

        disagreements = []
        for relpath, digest in sorted(recorded.items()):
            target = self.root / relpath
            if target.is_symlink():
                raise CompletionBarrierError(
                    f"{relpath} is a symbolic link; a ledgered artifact is a "
                    "real file inside the store, never an alias")
            if not target.exists():
                disagreements.append((relpath, "absent"))
                continue
            if root not in target.resolve().parents:
                raise CompletionBarrierError(
                    f"{relpath} resolves to {target.resolve()}, outside the "
                    f"store root {root}")
            if reg.sha256_file(target) != digest:
                disagreements.append((relpath, "digest"))
        if disagreements:
            raise CompletionBarrierError(
                f"the reread ledger disagrees with the store on "
                f"{len(disagreements)} artifacts, e.g. {disagreements[:5]}")
        return {"entries": recorded, "ledger_sha256": reg.sha256_file(path),
                "n_entries": len(recorded)}

    def content_digest(self) -> str:
        """`sha256` over sorted `(relpath, sha256(bytes))` — §I.3."""
        import hashlib

        h = hashlib.sha256()
        for relpath in sorted(self.relpaths_on_disk()):
            if relpath == reg.COMPLETION_FILE:
                continue
            h.update(relpath.encode("utf-8"))
            h.update(reg.sha256_file(self.root / relpath).encode("utf-8"))
        return h.hexdigest()


# ---------------------------------------------------------------------------
# the smoke store — structurally unable to complete a run
# ---------------------------------------------------------------------------
@dataclass
class SmokeStageFStore(_StageFRecordStore):
    """Non-authoritative. Refuses the authoritative root; has no `complete_run`."""

    @classmethod
    def create(cls, root, provenance: dict | None = None,
               split_identities=None) -> "SmokeStageFStore":
        resolved = Path(root).resolve()
        lexical = Path(authoritative_root())
        # Both forms: a smoke run may not write to the authoritative root, nor
        # to whatever that root would resolve to if it were ever redirected.
        forbidden = {lexical, lexical.resolve()}
        if resolved in forbidden \
                or any(a in resolved.parents for a in forbidden) \
                or any(resolved in a.parents for a in forbidden):
            raise StageFStoreError(
                f"{resolved} is (or contains) the authoritative store "
                f"{lexical}; a smoke run never writes there")
        resolved.mkdir(parents=True, exist_ok=True)
        # A rehearsal binds the same verified identities a Formal run does. It
        # may be handed a different mapping — that is what makes "verified for
        # one registry, persisted from another" testable — but it can never be
        # handed *no* mapping and still write a site x `N` artifact.
        import stage_f_inputs as inputs

        if split_identities is None:
            split_identities = inputs.derive_verified_split_identities()
        store = cls(root=resolved, run_kind=reg.RUN_KIND_SMOKE,
                    provenance=provenance or build_provenance(reg.RUN_KIND_SMOKE),
                    _splits=split_identities)
        store.write_json(reg.PROVENANCE_FILE, store.provenance)
        return store

    def seal_smoke(self) -> dict:
        """Freeze a ledger and stop. No completion marker exists on this path."""
        ledger = self.write_checksum_ledger()
        return {"authoritative": False, "run_kind": reg.RUN_KIND_SMOKE,
                "completion_marker": None, "checksum_ledger_sha256": ledger,
                "n_artifacts": len(self._written),
                "content_digest": self.content_digest()}


# ---------------------------------------------------------------------------
# the formal store — hard-bound root, stale-root fail-closed, seven-step barrier
# ---------------------------------------------------------------------------
@dataclass
class FormalStageFStore(_StageFRecordStore):
    """The one authoritative Stage-F store. Its root is `results/stage_f_v2/`.

    The direct constructor is refused **unconditionally**: no root, no
    `run_kind` and no attribute assignment produces an instance this way, so
    there is nothing for a caller to configure into authority. The one
    authorized instance is assembled inside the authority closure by
    `create_authoritative`, which binds it by object identity; completion
    re-derives that binding, the class, the run kind and the repository's own
    `results/stage_f_v2/` from scratch.
    """

    def __post_init__(self) -> None:
        raise FormalAuthorityError(
            "FormalStageFStore(...) is not a construction path: the one "
            "authoritative store is built by "
            "FormalStageFStore.create_authoritative(), which binds it to the "
            "repository's own results/stage_f_v2/ by object identity (§I.2)")

    @classmethod
    def create_authoritative(cls, provenance: dict | None = None
                             ) -> "FormalStageFStore":
        """Takes **no root argument** (§I.2) and refuses a non-empty root.

        There is no merge, no resume, no repair, no back-fill and no overwrite.
        Archiving or removing a previous attempt is an explicit operator action
        taken before the run, never something this class does.
        """
        return _create_authoritative_store(cls, provenance)

    # -- §I.1, the seven steps, in this exact order -------------------------
    def complete_run(self) -> dict:
        # Re-established here, not inherited from construction: attributes are
        # mutable, so the barrier proves authority again at the moment it is
        # used — and proves first that the mechanism itself still holds no
        # insertable authorisation state.
        assert_no_authorization_container()
        _verify_formal_authority(self)

        unvalidated = sorted(set(self._written) - set(self._validated)
                             - {reg.CHECKSUM_FILE})
        if unvalidated:
            raise CompletionBarrierError(
                f"{len(unvalidated)} artifacts did not pass the write-time "
                f"schema/key/status barrier: {unvalidated[:5]}")
        # Every duplicated scientific fact was actually compared against its
        # sibling, rather than each file having been checked alone.
        try:
            cross_groups = _assert_cross_bindings_complete(self._cross)
        except CrossArtifactError as exc:
            raise CompletionBarrierError(str(exc)) from exc
        # …and the eligible-site identities every one of those artifacts is
        # keyed on are re-derived from the frozen split registry here, not
        # inherited from construction.
        try:
            split_binding = _assert_split_identity_binding(self)
        except SplitIdentityError as exc:
            raise CompletionBarrierError(str(exc)) from exc

        completion_path = self.root / reg.COMPLETION_FILE
        if completion_path.exists():
            raise CompletionBarrierError(
                "COMPLETION.json is present before step 7 (§J.3)")

        steps = []
        steps.append(("write_artifacts", len(self._written)))          # 1
        assert cross_groups == len(CROSS_ARTIFACT_GROUPS)

        ledger_digest = self.write_checksum_ledger()                   # 2
        steps.append(("write_checksum_ledger", ledger_digest))

        # 3 — every artifact was fsynced as it was written; this syncs the
        # directory entries of the tree itself, deepest first.
        synced = 0
        for directory in sorted({p.parent for p in self.root.rglob("*")
                                 if p.is_file()},
                                key=lambda p: len(p.parts), reverse=True):
            synced += _fsync_directory_chain(directory, self.root)
        steps.append(("flush_and_fsync", synced))

        ledger = self.read_checksum_ledger()                           # 4
        recorded = ledger["entries"]
        if ledger["ledger_sha256"] != ledger_digest:
            raise CompletionBarrierError(
                "the ledger on disk does not match the ledger just written")
        steps.append(("reread_from_disk", len(recorded)))

        mismatched = []                                                # 5
        for relpath, digest in sorted(self._written.items()):
            if relpath == reg.CHECKSUM_FILE:
                continue
            path = self.root / relpath
            if not path.exists():
                mismatched.append((relpath, "absent"))
                continue
            on_disk = reg.sha256_file(path)
            if on_disk != digest or recorded.get(relpath) != digest:
                mismatched.append((relpath, on_disk[:12]))
        if mismatched:
            raise CompletionBarrierError(
                f"digest mismatch at barrier step 5: {mismatched[:5]}")
        if self.provenance != json.loads(
                (self.root / reg.PROVENANCE_FILE).read_bytes()):
            raise CompletionBarrierError(
                "the persisted provenance differs from the run's provenance")
        steps.append(("recompute_and_compare_digests", len(self._written)))

        expected = set(reg.registered_artifact_paths()) - {reg.COMPLETION_FILE}
        observed = set(self.relpaths_on_disk())
        missing, unregistered = sorted(expected - observed), sorted(observed - expected)
        if missing or unregistered:
            raise CompletionBarrierError(
                f"the artifact set is not exact: {len(missing)} registered "
                f"artifacts missing {missing[:5]}, {len(unregistered)} "
                f"unregistered files present {unregistered[:5]} (§I.1 step 6)")

        # …and the four books must agree with one another exactly, not merely
        # each with the tree. The registry says what must exist; `_written` says
        # what this run persisted; `_validated` says what passed the write-time
        # barrier; the ledger says what was sealed. A subset relation anywhere
        # among them is a hole — an artifact written but never validated, or
        # validated but never sealed, or sealed but never registered — and each
        # of those is a way for an unchecked quantity to enter an authoritative
        # store. Only exact equality closes them (§I.1 step 6).
        registered = set(reg.expected_checksum_paths())
        books = {
            "registered": registered,
            "written": set(self._written) - {reg.CHECKSUM_FILE},
            "validated": set(self._validated) - {reg.CHECKSUM_FILE},
            "ledgered": set(recorded),
        }
        disagreeing = {name: (sorted(paths - registered)[:5],
                              sorted(registered - paths)[:5])
                       for name, paths in books.items() if paths != registered}
        if disagreeing:
            raise CompletionBarrierError(
                "the registered, written, validated and ledgered artifact sets "
                f"are not identical ({len(registered)} registered): "
                f"{disagreeing} — shown as (extra, missing) per book (§I.1)")
        steps.append(("verify_artifact_set_exact", len(observed)))

        # The last thing checked before the marker is the first thing checked at
        # the top: the mechanism holds nothing insertable, and this object's
        # authority — including the lexical symlink question — re-derived once
        # more against the live repository.
        n_inspected = assert_no_authorization_container()
        authority = _verify_formal_authority(self)
        completion = {                                                 # 7
            "stage": "F",
            "run_kind": reg.RUN_KIND_FORMAL,
            "authoritative": True,
            "authority": authority,
            "protocol_version": reg.PROTOCOL_VERSION,
            "protocol_tag": reg.PROTOCOL_TAG,
            "protocol_commit": reg.PROTOCOL_COMMIT,
            "implementation_commit": self.provenance.get("implementation_commit"),
            "checksum_ledger_sha256": ledger_digest,
            "content_digest": self.content_digest(),
            "n_artifacts": len(observed),
            "n_artifacts_validated": len(self._validated),
            "artifact_sets_identical": sorted(books),
            "n_checksum_covered_artifacts": len(registered),
            "authority_mechanism": {
                "authorization_container_present": False,
                "n_authority_references_inspected": n_inspected,
                "root_symlink_rejected_before_resolution": True,
            },
            "split_registry_identity_binding": split_binding,
            "write_time_validation": {
                "schema_enforced": True, "logical_keys_unique": True,
                "status_semantics_enforced": True,
                "numeric_semantics_enforced": True,
                "column_projection_permitted": reg.COLUMN_PROJECTION_PERMITTED,
                "duplicate_logical_keys_permitted":
                    reg.DUPLICATE_LOGICAL_KEYS_PERMITTED,
                "numeric_strings_coerced": reg.NUMERIC_STRINGS_COERCED,
                "closed_domain_semantics_enforced": True,
                "closed_domain_strings_coerced":
                    reg.CLOSED_DOMAIN_STRINGS_COERCED,
                "json_serialisation_coerces_objects":
                    reg.JSON_SERIALISATION_COERCES_OBJECTS,
                "nonfinite_permitted_anywhere": reg.NONFINITE_PERMITTED_ANYWHERE,
                "n_status_rows_checked": sum(
                    r["status_rows_checked"] for r in self._validated.values()),
                "n_numeric_cells_checked": sum(
                    r["numeric_cells_checked"] for r in self._validated.values()),
                "n_closed_domain_cells_checked": sum(
                    r["closed_domain_cells_checked"]
                    for r in self._validated.values()),
            },
            "durability": {"per_file_fsync": True,
                           "directory_entries_fsynced": synced,
                           "ledger_reread_and_verified": True,
                           "ledger_covers_itself": reg.LEDGER_COVERS_ITSELF,
                           "ledger_covers_completion":
                               reg.LEDGER_COVERS_COMPLETION},
            "barrier_steps": list(reg.BARRIER_STEPS),
            "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        marker = _atomic_write(completion_path, canonical_json(completion),
                               fsync_up_to=self.root)
        steps.append(("write_completion_marker_atomically", marker))

        if [name for name, _ in steps] != list(reg.BARRIER_STEPS):
            raise CompletionBarrierError(
                f"the barrier ran {[n for n, _ in steps]}, the frozen order is "
                f"{list(reg.BARRIER_STEPS)}")
        return {**completion, "completion_marker_sha256": marker,
                "steps": [name for name, _ in steps]}


# ---------------------------------------------------------------------------
# persisting one run's results
# ---------------------------------------------------------------------------
def persist_rq4(store, rq4: dict) -> dict:
    digests = {
        "rq4/per_site_metrics.csv": store.write_table(
            "rq4/per_site_metrics.csv", rq4["per_site_metrics"],
            reg.RQ4_PER_SITE_COLUMNS),
        "rq4/per_repeat_target_local.csv": store.write_table(
            "rq4/per_repeat_target_local.csv", rq4["per_repeat_target_local"],
            reg.RQ4_PER_REPEAT_COLUMNS),
        "rq4/site_macro_metrics.csv": store.write_table(
            "rq4/site_macro_metrics.csv", rq4["site_macro_metrics"],
            reg.RQ4_SITE_MACRO_COLUMNS),
        "rq4/marginal_metrics.csv": store.write_table(
            "rq4/marginal_metrics.csv", rq4["marginal_metrics"],
            reg.RQ4_MARGINAL_COLUMNS),
        "rq4/site_bootstrap_intervals.csv": store.write_table(
            "rq4/site_bootstrap_intervals.csv", rq4["site_bootstrap_intervals"],
            reg.RQ4_INTERVAL_COLUMNS),
        "rq4/degeneracy.csv": store.write_table(
            "rq4/degeneracy.csv", rq4["degeneracy"], reg.RQ4_DEGENERACY_COLUMNS),
        "rq4/conformal_config.json": store.write_json(
            "rq4/conformal_config.json", rq4["conformal_config"]),
    }
    return digests


def persist_svm(store, svm: dict) -> dict:
    return {
        "svm/per_site_bacc.csv": store.write_table(
            "svm/per_site_bacc.csv", svm["per_site_bacc"],
            reg.SVM_PER_SITE_COLUMNS),
        "svm/site_macro_bacc.json": store.write_json(
            "svm/site_macro_bacc.json", svm["site_macro_bacc"]),
        "svm/fold_selection.csv": store.write_table(
            "svm/fold_selection.csv", svm["fold_selection"],
            reg.SVM_FOLD_SELECTION_COLUMNS),
        "svm/svm_config.json": store.write_json("svm/svm_config.json",
                                                svm["svm_config"]),
    }


def persist_sensitivity(store, label: str, tables: dict) -> dict:
    """One sensitivity's registered tables plus its `manifest.json`."""
    registered = reg.SENSITIVITY_ARTIFACTS[label]
    missing = sorted(set(registered) - set(tables))
    extra = sorted(set(tables) - set(registered))
    if missing or extra:
        raise StageFStoreError(
            f"{label}: {len(missing)} registered tables missing {missing}, "
            f"{len(extra)} unregistered tables offered {extra}")
    digests = {}
    for name in sorted(registered):
        relpath = f"{reg.NAMESPACE_SENSITIVITIES}/{label}/{name}"
        payload = tables[name]
        digests[relpath] = (store.write_json(relpath, payload)
                            if name.endswith(".json")
                            else store.write_table(relpath, payload))
    return digests
