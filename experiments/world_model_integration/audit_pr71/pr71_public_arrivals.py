"""PR71 original-rule-safe PUBLIC arrival boundary for predicted business worlds.

Conditioned experiment ONLY: at current public tick t, ingest *new* stage1
objects visible in the REAL public observation at t, never from future state
or hidden job table. No factual stage2 is injected. This adapter is not an
autonomous new-job arrival prediction model. Preserve immutable original
world-state legality and recompute station/resource observations.
"""
from __future__ import annotations

from dataclasses import replace

from business_transport.observation import attach_prediction, capacity_view
from business_transport.scenario import stage_id
from lab_transport.state_v2 import rebuild


def apply_current_public_stage1(predicted, actual_current, public_previous, rules):
    """Only reveal newly publicly visible stage1 at the current tick.

    The caller MUST supply an actual public observation for the same tick.
    Earlier previous actual observation is used ONLY to identify first public
    arrival. A stage1 that should already have been visible but disappeared
    from predicted world is a failure, not silently re-injected.
    """
    if predicted.tick != actual_current.tick:
        raise ValueError("arrival stream must be the public CURRENT tick, never t+1")
    if public_previous is not None and public_previous.tick + 1 != actual_current.tick:
        raise ValueError("public arrival observation stream is not chronological")
    known = {s.id: s for s in predicted.samples}
    previous_ids = ({s.id for s in public_previous.samples
                     if getattr(s, 'stage_index', None) == 0}
                    if public_previous is not None else set())
    if previous_ids - set(known):
        raise ValueError("already public entity disappeared from predicted world")
    arriving = []
    for sample in actual_current.samples:
        if (getattr(sample, "stage_index", None) != 0
            or sample.id != stage_id(sample.job_id, 0)
            or sample.id in previous_ids):
            continue
        if sample.id in known:
            continue
        if (sample.status != "waiting" or sample.robot_id is not None
            or sample.release_tick > actual_current.tick):
            raise ValueError("newly visible stage1 must be a public waiting task")
        # The original current observation contains this complete *public*
        # task object. Never copy a future (t+1) observation or a stage2.
        arriving.append(sample)
    if not arriving:
        return predicted, ()
    staged = replace(predicted, samples=tuple(sorted(
        (*predicted.samples, *arriving), key=lambda s: s.id
    )))
    # Mirror the ORIGINAL release_successors recomputation. Merely appending
    # a sample without rebuilding the station demand leaves inconsistent
    # waiting_pickups/approaching_pickups and resource observations.
    # rebuild() returns plain StationV2 without the BusinessStationV2
    # device_available field. Preserve the actual *current* device bits BEFORE
    # that conversion; never infer a device event from station.available.
    devices = {s.id: bool(getattr(s, "device_available", s.available))
               for s in predicted.stations}
    rebuilt = attach_prediction(staged, rebuild(staged, staged.robots))
    staged = replace(staged, samples=rebuilt.samples, stations=rebuilt.stations)
    staged = capacity_view(staged, (), rules, device_status=devices)[0]
    if any(s.id.endswith("::stage2") for s in arriving):
        raise AssertionError("factual successor stage2 leaked into prediction")
    return staged, tuple(s.id for s in arriving)
