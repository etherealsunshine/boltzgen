"""Residue-level introspective acceptance utilities.

This module is intentionally lightweight and CPU-only. It isolates the
I-DLM-style acceptance rule we would want to plug into a BoltzGen proposal path:

    z ~ q
    accept z with min(1, p(z) / q(z))
    otherwise resample z ~ normalize(max(0, p - q))

Here `q` is a proposal distribution, for example a BoltzGen residue proposal
head, and `p` is an anchor/verifier distribution, for example an inverse-folding
or future BoltzGen anchor-mode head.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from typing import Mapping


AMINO_ACIDS: tuple[str, ...] = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
)


@dataclass(frozen=True)
class AcceptanceResult:
    """One introspective acceptance decision."""

    final: str
    accepted: bool
    proposal: str
    accept_probability: float


@dataclass(frozen=True)
class SimulationResult:
    """Summary from repeated introspective acceptance samples."""

    acceptance_rate: float
    l1_to_anchor: float
    empirical_distribution: dict[str, float]
    common_corrections: list[tuple[tuple[str, str], int]]


def normalize(distribution: Mapping[str, float]) -> dict[str, float]:
    """Return a normalized copy of a non-negative categorical distribution."""

    total = sum(distribution.values())
    if total <= 0:
        msg = "distribution must have positive total mass"
        raise ValueError(msg)

    normalized = {}
    for key, value in distribution.items():
        if value < 0:
            msg = f"distribution has negative mass for {key!r}: {value}"
            raise ValueError(msg)
        normalized[key] = value / total
    return normalized


def ensure_support(
    p: Mapping[str, float],
    q: Mapping[str, float],
    support: tuple[str, ...] = AMINO_ACIDS,
) -> tuple[dict[str, float], dict[str, float]]:
    """Normalize p/q and ensure all support keys are present."""

    missing_p = set(support) - set(p)
    missing_q = set(support) - set(q)
    if missing_p or missing_q:
        msg = f"missing support keys: p={sorted(missing_p)}, q={sorted(missing_q)}"
        raise ValueError(msg)

    return normalize({key: p[key] for key in support}), normalize(
        {key: q[key] for key in support}
    )


def sample_categorical(
    distribution: Mapping[str, float],
    rng: random.Random | None = None,
) -> str:
    """Sample one key from a categorical distribution."""

    rng = rng or random
    draw = rng.random()
    cumulative = 0.0
    last_key = None
    for key, probability in distribution.items():
        cumulative += probability
        last_key = key
        if draw <= cumulative:
            return key

    if last_key is None:
        msg = "cannot sample from empty distribution"
        raise ValueError(msg)
    return last_key


def corrected_distribution(
    p: Mapping[str, float],
    q: Mapping[str, float],
    support: tuple[str, ...] = AMINO_ACIDS,
) -> dict[str, float]:
    """Distribution used after rejecting a proposal: normalize(max(0, p - q))."""

    p_norm, q_norm = ensure_support(p, q, support)
    residual = {key: max(0.0, p_norm[key] - q_norm[key]) for key in support}
    if sum(residual.values()) == 0:
        return p_norm
    return normalize(residual)


def acceptance_rate(
    p: Mapping[str, float],
    q: Mapping[str, float],
    support: tuple[str, ...] = AMINO_ACIDS,
) -> float:
    """Exact expected acceptance rate, equal to distribution overlap."""

    p_norm, q_norm = ensure_support(p, q, support)
    return sum(min(p_norm[key], q_norm[key]) for key in support)


def exact_final_distribution(
    p: Mapping[str, float],
    q: Mapping[str, float],
    support: tuple[str, ...] = AMINO_ACIDS,
) -> dict[str, float]:
    """Analytic final distribution after accept/correct.

    This should equal the anchor distribution p up to floating-point error.
    Keeping the function explicit makes the POC testable without relying on
    Monte Carlo noise.
    """

    p_norm, q_norm = ensure_support(p, q, support)
    correction = corrected_distribution(p_norm, q_norm, support)
    reject_mass = 1.0 - acceptance_rate(p_norm, q_norm, support)

    final = {}
    for key in support:
        accepted_mass = min(p_norm[key], q_norm[key])
        final[key] = accepted_mass + reject_mass * correction[key]
    return final


def introspective_accept(
    p: Mapping[str, float],
    q: Mapping[str, float],
    rng: random.Random | None = None,
    support: tuple[str, ...] = AMINO_ACIDS,
) -> AcceptanceResult:
    """Sample a proposal from q and accept/correct it against anchor p."""

    rng = rng or random
    p_norm, q_norm = ensure_support(p, q, support)
    proposal = sample_categorical(q_norm, rng)
    accept_probability = min(1.0, p_norm[proposal] / max(q_norm[proposal], 1e-12))

    if rng.random() < accept_probability:
        return AcceptanceResult(
            final=proposal,
            accepted=True,
            proposal=proposal,
            accept_probability=accept_probability,
        )

    correction = corrected_distribution(p_norm, q_norm, support)
    final = sample_categorical(correction, rng)
    return AcceptanceResult(
        final=final,
        accepted=False,
        proposal=proposal,
        accept_probability=accept_probability,
    )


def l1_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
    support: tuple[str, ...] = AMINO_ACIDS,
) -> float:
    """L1 distance between two categorical distributions."""

    return sum(abs(left[key] - right[key]) for key in support)


def run_simulation(
    p: Mapping[str, float],
    q: Mapping[str, float],
    *,
    n_samples: int = 100_000,
    seed: int = 7,
    support: tuple[str, ...] = AMINO_ACIDS,
) -> SimulationResult:
    """Run a deterministic Monte Carlo sanity check for the acceptance rule."""

    rng = random.Random(seed)
    p_norm, q_norm = ensure_support(p, q, support)
    final_counts: Counter[str] = Counter()
    corrections: Counter[tuple[str, str]] = Counter()
    accepted = 0

    for _ in range(n_samples):
        result = introspective_accept(p_norm, q_norm, rng, support)
        final_counts[result.final] += 1
        accepted += int(result.accepted)
        if not result.accepted:
            corrections[(result.proposal, result.final)] += 1

    empirical = {key: final_counts[key] / n_samples for key in support}
    return SimulationResult(
        acceptance_rate=accepted / n_samples,
        l1_to_anchor=l1_distance(empirical, p_norm, support),
        empirical_distribution=empirical,
        common_corrections=corrections.most_common(5),
    )


def toy_pocket_distributions() -> tuple[dict[str, float], dict[str, float]]:
    """Return anchor p and proposal q for a toy charged-pocket residue."""

    q = normalize(
        dict(
            zip(
                AMINO_ACIDS,
                (
                    0.02,
                    0.22,
                    0.04,
                    0.01,
                    0.01,
                    0.14,
                    0.02,
                    0.01,
                    0.04,
                    0.02,
                    0.03,
                    0.32,
                    0.01,
                    0.03,
                    0.01,
                    0.03,
                    0.02,
                    0.01,
                    0.06,
                    0.02,
                ),
            )
        )
    )
    p = normalize(
        dict(
            zip(
                AMINO_ACIDS,
                (
                    0.02,
                    0.41,
                    0.03,
                    0.01,
                    0.01,
                    0.21,
                    0.01,
                    0.01,
                    0.05,
                    0.02,
                    0.02,
                    0.08,
                    0.01,
                    0.02,
                    0.01,
                    0.03,
                    0.02,
                    0.01,
                    0.07,
                    0.02,
                ),
            )
        )
    )
    return p, q

