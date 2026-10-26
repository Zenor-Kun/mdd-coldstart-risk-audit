"""Synthetic RQ3 fixtures — for tests and the non-authoritative smoke only.

Nothing here reads the manifest, the S0 confidence table, the frozen thresholds
or any real `R_i`. Every frame it returns is generated from a seeded RNG, so a
quantity computed on it is a property of the generator, never an RQ3 result.

Two uses:

* **substantive numerical tests** — `simulate_glmm` draws from the §9.1 model
  itself, so parameter recovery, quadrature convergence and the four-start rule
  can be checked against a known truth;
* **the mechanical smoke** — `synthetic_population` returns frames with the
  frozen site counts (23 and 9) and the frozen covariate columns, so the whole
  persistence and 40/60 wiring can be exercised end to end at a size that runs
  in seconds.

The site counts match the real ones because `SITE_BALANCED_DENOMINATOR` and the
LOSO fold counts are frozen at 23 and 9; the *row* counts deliberately do not,
so a synthetic frame can never be mistaken for a frozen complete-case subset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rq3_data as data
import rq3_protocol as proto

#: The site counts of the two frozen populations (§5.4, §15.1).
SYNTHETIC_SITE_COUNT = {data.FULL_SAMPLE: 23, data.MDD_ONLY: 9}

#: The covariates each population's models need (§6, §7).
SYNTHETIC_COLUMNS = {
    data.FULL_SAMPLE: ("age", "sex", "mean_fd", "diagnosis"),
    data.MDD_ONLY: ("mean_fd", "hamd", "first_episode", "on_medication",
                    "illness_duration_months"),
}


def simulate_glmm(n_sites: int = 12, n_per_site: int = 40, beta=(-0.4, 0.8, -0.5),
                  sigma_site: float = 0.7, seed: int = 0) -> dict:
    """Draw from the §9.1 model with known `beta` and `sigma_site`.

    `beta[0]` is the intercept. Returns the design matrix (intercept included),
    the binary outcome, the site labels in contiguous blocks, and the truth.
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    n_slopes = beta.size - 1
    n = n_sites * n_per_site
    sites = np.repeat([f"S{i + 1}" for i in range(n_sites)], n_per_site)
    slopes = rng.normal(size=(n, n_slopes)) if n_slopes else np.empty((n, 0))
    X = np.column_stack([np.ones(n), slopes])
    u = np.repeat(rng.normal(0.0, sigma_site, n_sites), n_per_site)
    probability = 1.0 / (1.0 + np.exp(-(X @ beta + u)))
    R = (rng.random(n) < probability).astype(float)
    return {
        "X": X, "R": R, "sites": sites,
        "beta": beta, "sigma_site": float(sigma_site),
        "u": u, "n_sites": n_sites, "n_per_site": n_per_site,
    }


def synthetic_population(name: str, n_per_site: int = 6, seed: int = 0) -> data.Population:
    """A frame with the frozen site count and the frozen covariate columns.

    Values are drawn inside the domains the §8 coding maps accept, so the
    production coding path is exercised rather than bypassed.
    """
    if name not in SYNTHETIC_SITE_COUNT:
        raise ValueError(f"unknown population {name!r}")
    rng = np.random.default_rng(seed)
    n_sites = SYNTHETIC_SITE_COUNT[name]
    sites = [f"S{i + 1}" for i in range(n_sites)]
    rows = []
    for site_index, site in enumerate(sites):
        for row_index in range(n_per_site):
            row = {
                "canonical_id": f"{site}-9-{row_index:04d}",
                "feature_source_id": f"{site}-9-{row_index:04d}",
                "site": site,
                "mean_fd": float(abs(rng.normal(0.15, 0.06)) + 0.01),
            }
            if name == data.FULL_SAMPLE:
                row["age"] = float(rng.uniform(18.0, 65.0))
                row["sex"] = float(rng.integers(1, 3))              # 1 or 2
                row["diagnosis"] = float((site_index + row_index) % 2)
            else:
                row["diagnosis"] = 1.0
                row["hamd"] = float(rng.uniform(5.0, 30.0))
                row["first_episode"] = float(rng.choice([-1.0, 1.0]))
                row["on_medication"] = float(rng.choice([-1.0, 1.0]))
                row["illness_duration_months"] = float(rng.uniform(0.5, 120.0))
            rows.append(row)
    frame = pd.DataFrame(rows)
    for column in SYNTHETIC_COLUMNS[name]:
        if column not in frame.columns:
            raise AssertionError(f"synthetic {name} frame is missing {column}")
    return data.Population(name=name, frame=frame)


def synthetic_outcome(population: data.Population, operating_point: str,
                      method: str, seed: int = 0,
                      site_sd: float = 0.6) -> data.CellOutcome:
    """A binary outcome with site-level structure, drawn independently of `X`.

    The generator is deliberately covariate-free: the smoke must not produce
    anything resembling an association between a real covariate and abstention.
    Prevalence is kept away from 0 and 1 within every site so no fold is
    degenerate for mechanical reasons.
    """
    if (operating_point, method) not in proto.CELL_REGISTRY:
        raise ValueError(f"unregistered cell {(operating_point, method)!r}")
    cell_index = proto.CELL_REGISTRY.index((operating_point, method))
    rng = np.random.default_rng(seed * 1000 + cell_index)
    site_effect = {site: rng.normal(0.0, site_sd) for site in population.sites}
    logit = population.frame.site.map(site_effect).to_numpy(dtype=float)
    probability = 1.0 / (1.0 + np.exp(-logit))
    R = (rng.random(population.n) < probability).astype(float)

    # keep every site non-degenerate: a site that is all-0 or all-1 makes the
    # mechanics uninteresting rather than wrong, and the real full-sample subset
    # has no degenerate site in any cell (§2.4)
    frame = population.frame.reset_index(drop=True)
    for site in population.sites:
        mask = (frame.site == site).to_numpy()
        if mask.sum() < 2:
            continue
        indices = np.flatnonzero(mask)
        if R[indices].sum() == 0:
            R[indices[0]] = 1.0
        elif R[indices].sum() == indices.size:
            R[indices[0]] = 0.0
    return data.CellOutcome(population.name, operating_point, method, R)


def synthetic_populations(n_per_site: int = 6, seed: int = 0) -> dict:
    return {
        name: synthetic_population(name, n_per_site=n_per_site, seed=seed + index)
        for index, name in enumerate(data.POPULATIONS)
    }


def synthetic_outcome_builder(seed: int = 0):
    """An `outcome_builder` for `rq3_execution.execute`, on synthetic data."""
    def build(population: data.Population, operating_point: str, method: str):
        return synthetic_outcome(population, operating_point, method, seed=seed)
    return build
