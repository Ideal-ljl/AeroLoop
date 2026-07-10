from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from .metrics import EpisodeMetrics, MetricConfig
from .protocols import EnvironmentAdapter, PolicyAdapter
from .types import (
    CanonicalAction,
    EpisodeResult,
    EpisodeSpec,
    PolicyInput,
    Observation,
    StepRecord,
    TerminationReason,
)


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 200
    execution_horizon: int | None = 1
    stop_threshold: float = 0.5
    terminate_on_collision: bool = True
    execute_motion_on_stop: bool = False
    include_steps_in_result: bool = True

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.execution_horizon is not None and self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive or null for native chunk execution")


class RolloutRunner:
    def __init__(
        self,
        environment: EnvironmentAdapter,
        policy: PolicyAdapter,
        rollout: RolloutConfig | None = None,
        metrics: MetricConfig | None = None,
    ):
        self.environment = environment
        self.policy = policy
        self.rollout = rollout or RolloutConfig()
        self.metric_config = metrics or MetricConfig()

    def run_episode(self, episode: EpisodeSpec) -> EpisodeResult:
        records: list[StepRecord] = []
        image_history = []
        action_history: list[CanonicalAction] = []
        state_history = []
        queue: deque[tuple[CanonicalAction, int, int, float | None, dict]] = deque()
        inference_call = 0
        termination = TerminationReason.MAX_STEPS
        error = None

        tracker = EpisodeMetrics(episode, self.metric_config)
        observation = Observation(
            rgb=None,
            pose=episode.start_pose,
            relative_state=(0.0, 0.0, 0.0, 0.0),
            step_index=0,
        )

        try:
            self.policy.reset(episode)
            observation = self.environment.reset(episode)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            metrics = tracker.finalize(TerminationReason.ERROR, observation.pose, 0)
            return EpisodeResult(
                episode_id=episode.episode_id,
                env_name=episode.env_name,
                termination_reason=TerminationReason.ERROR,
                metrics=metrics,
                steps=(),
                error=error,
            )

        try:
            for step in range(self.rollout.max_steps):
                if not queue:
                    policy_input = PolicyInput(
                        episode=episode,
                        observation=observation,
                        image_history=tuple(image_history),
                        action_history=tuple(action_history),
                        state_history=tuple(state_history),
                    )
                    started = time.perf_counter()
                    chunk = self.policy.predict(policy_input)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    tracker.add_inference(elapsed_ms)
                    inference_call += 1
                    actions = chunk.actions
                    if self.rollout.execution_horizon is not None:
                        actions = actions[: self.rollout.execution_horizon]
                    for action_index, action in enumerate(actions):
                        queue.append((action, inference_call, action_index, elapsed_ms if action_index == 0 else None, dict(chunk.metadata)))

                action, call_index, action_index, inference_ms, policy_metadata = queue.popleft()
                stop_requested = action.stop >= self.rollout.stop_threshold
                effective_action = action
                if stop_requested and not self.rollout.execute_motion_on_stop:
                    effective_action = action.zero_motion()

                before_observation = observation
                before = before_observation.pose
                transition = self.environment.execute(effective_action)
                observation = transition.observation
                distances = tracker.update(before, observation.pose, collision=transition.collision, info=transition.info)
                self.policy.on_action_executed(effective_action, transition)

                records.append(
                    StepRecord(
                        step=step,
                        inference_call=call_index,
                        action_index=action_index,
                        action=action,
                        pose_before=before,
                        pose_after=observation.pose,
                        collision=transition.collision,
                        inference_ms=inference_ms,
                        distances=distances,
                        policy_metadata=policy_metadata,
                    )
                )
                image_history.append(before_observation.rgb)
                action_history.append(effective_action)
                state_history.append(before_observation.relative_state)

                if transition.collision and self.rollout.terminate_on_collision:
                    termination = TerminationReason.COLLISION
                    break
                if stop_requested:
                    termination = TerminationReason.STOP
                    break
            else:
                termination = TerminationReason.MAX_STEPS
        except Exception as exc:
            termination = TerminationReason.ERROR
            error = f"{type(exc).__name__}: {exc}"

        metrics = tracker.finalize(termination, observation.pose, len(records))
        return EpisodeResult(
            episode_id=episode.episode_id,
            env_name=episode.env_name,
            termination_reason=termination,
            metrics=metrics,
            steps=tuple(records) if self.rollout.include_steps_in_result else (),
            error=error,
        )
