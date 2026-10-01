"""Lazy, certified action selection shared by U2.2 wave adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from fgsp_ch.nonlocal_waves.adaptivity.prospective_causal_operator import (
    ACTIONS,
    analytical_safe_actions,
)
from fgsp_ch.nonlocal_waves.adaptivity.trait_responsive_controller import (
    TraitResponseDecision,
    TraitResponsiveState,
)


@dataclass(frozen=True, slots=True)
class ResponsiveSelection:
    chosen: object
    committed_error: float
    certificate_state: str
    neural_authority: bool
    revoked: bool
    probe_count: int
    probe_work: float
    raw_effectivity: float
    certified_upper: float
    candidate_work: float
    model_candidate_evaluations: int
    response: TraitResponseDecision
    analytically_safe: bool


def select_responsive_action(
    *, arm: str, build: Callable, probe: Callable,
    response_vector: Callable, response_state: TraitResponsiveState,
    mechanism, certificate, ledger, reserve_fraction: float,
    probe_safety: float, local_budget: float,
) -> ResponsiveSelection:
    """Select an action while evaluating alternatives only when diagnostics demand it."""
    if arm not in {"trait_responsive", "always_probe", "analytical_control"}:
        raise ValueError(f"unknown U2.2 arm: {arm}")
    hold = build(("hold",))[0]
    indicator = hold.indicators
    response = response_state.observe(
        response_vector(hold),
        hold_error=hold.envelope.deterministic_error,
        local_budget=local_budget,
        tail_ratio=indicator.tail_ratio,
        aliasing_defect=indicator.aliasing_defect,
    )
    hold_safe = bool(analytical_safe_actions(
        (hold.envelope,), reserve_fraction=reserve_fraction,
    ))
    # Laziness is an efficiency policy, never a safety policy. If the cheap
    # hold candidate is inadmissible, corrective actions must be constructed.
    expand_alternatives = bool(response.evaluate_alternatives or not hold_safe)
    if arm == "trait_responsive" and not expand_alternatives:
        options = (hold,)
    else:
        alternatives = build(tuple(action for action in ACTIONS if action != "hold"))
        by_action = {row.action: row for row in (hold, *alternatives)}
        options = tuple(by_action[action] for action in ACTIONS)
    envelopes = tuple(row.envelope for row in options)
    safe = analytical_safe_actions(envelopes, reserve_fraction=reserve_fraction)
    if not safe:
        raise RuntimeError("U2.2 has no analytically safe candidate")
    safe_options = [row for row in options if row.envelope in safe]
    candidate_work = float(sum(row.envelope.work for row in options))
    probe_count, probe_work = 0, 0.0
    authority, revoked = False, False
    status, certified, raw = "analytical_fallback", np.nan, np.nan

    if arm == "trait_responsive":
        proposal = mechanism.raw_proposal(
            np.stack([row.features for row in options]), envelopes,
            reserve_fraction=reserve_fraction,
        )
        proposed = next(
            (row for row in options if row.action == proposal.action_id), None
        )
        raw = proposal.upper_effectivity
        measured = None
        if proposed is not None:
            support = proposal.support_distances.get("universal", np.nan)
            preliminary = certificate.decide(
                action_id=proposed.action, raw_effectivity=raw,
                deterministic_error=proposed.envelope.deterministic_error,
                remaining_budget=ledger.remaining, support_distance=support,
                reserve_fraction=reserve_fraction,
                force_probe=response.force_probe,
            )
            decision = preliminary
            if preliminary.probe_required:
                measured, work = probe(proposed)
                probe_count, probe_work = 1, float(work)
                decision = certificate.decide(
                    action_id=proposed.action, raw_effectivity=raw,
                    deterministic_error=proposed.envelope.deterministic_error,
                    remaining_budget=ledger.remaining, support_distance=support,
                    probe_error=measured, reserve_fraction=reserve_fraction,
                    force_probe=response.force_probe,
                )
            authority, revoked = decision.authority, decision.revoked
            certified, status = decision.certified_upper, decision.state
            if authority:
                chosen, committed = proposed, certified
            elif measured is not None and probe_safety * measured <= (
                1.0 - reserve_fraction
            ) * ledger.remaining:
                chosen, committed = proposed, probe_safety * measured
                status = "probed_commit"
            else:
                chosen = min(
                    safe_options,
                    key=lambda row: (
                        row.envelope.work, row.envelope.deterministic_error,
                        row.action,
                    ),
                )
                committed = chosen.envelope.deterministic_error
        else:
            chosen = min(safe_options, key=lambda row: row.envelope.work)
            committed = chosen.envelope.deterministic_error
    elif arm == "always_probe":
        measured_options = []
        for option in safe_options:
            error, work = probe(option)
            measured_options.append((option, error))
            probe_count += 1
            probe_work += float(work)
        admitted = [
            pair for pair in measured_options
            if probe_safety * pair[1]
            <= (1.0 - reserve_fraction) * ledger.remaining
        ]
        admitted_ids = {id(pair[0]) for pair in admitted}
        chosen, measured = min(
            admitted or measured_options,
            key=lambda pair: (
                0 if id(pair[0]) in admitted_ids else 1,
                pair[0].envelope.work, pair[1],
            ),
        )
        committed = (
            probe_safety * measured if id(chosen) in admitted_ids
            else chosen.envelope.deterministic_error
        )
        status = "always_probe"
    else:
        chosen = min(
            safe_options,
            key=lambda row: (
                row.envelope.work, row.envelope.deterministic_error, row.action
            ),
        )
        committed = chosen.envelope.deterministic_error

    if not ledger.admissible(committed):
        chosen = min(safe_options, key=lambda row: row.envelope.deterministic_error)
        committed = chosen.envelope.deterministic_error
        authority, status = False, "budget_fallback"
    return ResponsiveSelection(
        chosen, float(committed), status, bool(authority), bool(revoked),
        probe_count, float(probe_work), float(raw), float(certified),
        candidate_work, len(options) if arm == "trait_responsive" else 0,
        response, chosen.envelope in safe,
    )
