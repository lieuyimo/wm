"""PR77 verified PR61 relational GNN -> EXISTING lawful full-world forecast API.

This is not a new algorithm: reuse the original PR59/61 pretrained
PublicMapTemporalRelationalWorld (GRU/RGCN/SAGEConv), PR67
NeuralTaskServiceRule, business WorkflowDynamics safety checks, and
RHWM ForecastBatch/ForecastParticle/_distribution from hybrid.py.

The archived PR61 original weights are local-only. Production or research
inference can be created ONLY from the original SHA/manifest-admitted archive.
An explicitly named private test-fixture entrypoint checks structural wiring
with UNTRAINED predictors but cannot be mistaken for trained research output.

This LAB-ONLY adapter is deliberately not a rail traffic forecasting model.
Rail must implement domain-specific observation encoder, prediction heads
and legitimate track/route/signal legal projection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch

from business_transport.workflow_transition import check_business_transition
from experiments.world_model_integration.neural_pr67.pr67_neural_world import (
    NeuralTaskServiceRule,
)
from experiments.world_model_integration.recovery_pr68.pr68_checkpoint import (
    ARCHIVE_SHA256, load_original_models,
)
from lab_transport.state_v2 import ObservationV2
from lab_transport.transition_schema import Assignment

from .contracts import (
    EventForecast, ForecastBatch, ForecastParticle, ForecastStep,
)
from .hybrid import _distribution, _seed


class RelationalGNNSystemWorldModel:
    """Existing relational graph model is the primary predictive backbone.

    Event probabilities come from PR61's actual TASK NEXT STATUS output;
    PR67 ensures legal service successor/robot/resource state via original
    ProgressAwareRule + WorkflowDynamics. Future unseen arrivals and faults
    are still explicitly unknown.
    """

    MAX_HORIZON = 20
    MAX_PARTICLES = 8

    def __init__(self, geometry, rules, predictor, *, model_version, _fixture=False):
        if not _fixture and not model_version.startswith("pr61-sha-verified-"):
            raise ValueError("PR61 relational model requires original verified checkpoint")
        if not callable(getattr(predictor, "forward", None)):
            raise TypeError("PR61-compatible PyG predictor must be supplied")
        self.geometry, self.rules = geometry, rules
        self.predictor = predictor.cpu().eval()
        self.model_version = model_version
        self.untrained_test_fixture = _fixture

    @classmethod
    def from_original_archive(
        cls, geometry, rules, archive: Path, manifest: Path, *,
        arm: str = "map_on", seed: int = 1810,
        original_validation: Path | None = None,
    ):
        # Admission runs BEFORE the predictor can be constructed or called.
        models, report = load_original_models(
            Path(archive), Path(manifest), original_validation)
        if (not report.get("weights_loaded_from_original_pr61")
                or report.get("archive_sha256") != ARCHIVE_SHA256):
            raise ValueError("PR61 original archive attestation missing")
        if (arm, seed) not in models:
            raise ValueError("unregistered PR61 arm/seed")
        version = f"pr61-sha-verified-{arm}-seed{seed}"
        return cls(geometry, rules, models[(arm, seed)],
                   model_version=version)

    @classmethod
    def _for_untrained_contract_test_only(cls, geometry, rules, predictor):
        return cls(
            geometry, rules, predictor,
            model_version="UNTRAINED_PR77_INTERFACE_TEST_NOT_A_MODEL_RESULT",
            _fixture=True,
        )

    def _core(self, seed: int):
        return NeuralTaskServiceRule(
            self.geometry, self.rules, self.predictor, seed=seed)

    def _step(
        self, core: NeuralTaskServiceRule,
        state: ObservationV2,
        action: Sequence[Assignment],
    ):
        if not isinstance(state, ObservationV2):
            raise ValueError("PR61 model requires real public LAB ObservationV2")
        proposed=tuple(action)
        predicted=core.predict_next(state, proposed)
        # Original LAB workflow/robot/resource legality is never bypassed.
        check_business_transition(
            state, predicted, self.geometry, self.rules, proposed)
        if predicted.tick != state.tick + 1:
            raise AssertionError("PR61 world step must advance exactly one tick")
        events=tuple(EventForecast(
            robot_id=e.robot_id, sample_id=e.sample_id,
            phase=e.phase, station_id=e.station_id,
            p_finish_next=e.probability_complete,
            sampled_finished=e.sampled_complete,
        ) for e in core.last_neural_events)
        return ForecastStep(state, proposed, predicted,
                            events, self.model_version)

    def predict_next(self, public_state: ObservationV2,
                     legal_assignments: Sequence[Assignment], *,
                     sampling_seed: int = 1810) -> ForecastStep:
        if type(sampling_seed) is not int:
            raise ValueError("sampling_seed must be integer")
        # Single step has no invented pre-t public history. A multistep
        # forecast uses rollout(), which preserves the PR54 chronological
        # 4-frame history on each particle (no reset per step).
        return self._step(self._core(sampling_seed),
                          public_state, legal_assignments)

    def rollout(
        self, public_state: ObservationV2,
        first_action: Sequence[Assignment], *,
        continuation_policy: Callable[[ObservationV2], Sequence[Assignment]],
        horizon_ticks: int = 5,
        particles: int = 3,
        sampling_seed: int = 1810,
        continuation_policy_name: str = "public_predicted_state",
    ) -> ForecastBatch:
        if not isinstance(public_state, ObservationV2):
            raise ValueError("cannot forecast a rail state using LAB PR61 encoder")
        if not callable(continuation_policy):
            raise ValueError("continuation policy required for future model states")
        if (type(horizon_ticks) is not int
                or not 1 <= horizon_ticks <= self.MAX_HORIZON
                or type(particles) is not int
                or not 1 <= particles <= self.MAX_PARTICLES):
            raise ValueError("unsupported horizon/particle count")
        if type(sampling_seed) is not int or not continuation_policy_name:
            raise ValueError("missing seed or named public continuation policy")
        initial=tuple(first_action)
        traces=[]
        for p in range(particles):
            core=self._core(_seed(sampling_seed, p, public_state.tick))
            state=public_state
            steps=[]
            for idx in range(horizon_ticks):
                # Existing PR67 history mutates chronologically inside core.
                # Do NOT call reset_seed per tick: that clears temporal state.
                action=(initial if idx==0
                        else tuple(continuation_policy(state)))
                current=self._step(core, state, action)
                steps.append(current)
                state=current.next_state
            traces.append(ForecastParticle(p,tuple(steps)))
        result=tuple(traces)
        return ForecastBatch(
            self.model_version, public_state, initial,
            continuation_policy_name, result,
            tuple(_distribution(result,k) for k in range(horizon_ticks)),
        )
