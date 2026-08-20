from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from .metrics import Metric, MetricConfig, MetricSuite
from .protocols import PolicyAdapter, SimulatorAdapter
from .observers import RolloutObserver
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
    terminate_on_success: bool = False
    execute_motion_on_stop: bool = False
    include_steps_in_result: bool = True
    fail_on_observer_error: bool = False

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.execution_horizon is not None and self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive or null for native chunk execution")


class RolloutRunner:
    def __init__(
        self,
        simulator: SimulatorAdapter | None = None,
        policy: PolicyAdapter | None = None,
        rollout: RolloutConfig | None = None,
        metrics: MetricConfig | None = None,
        observers: Sequence[RolloutObserver] | None = None,
        custom_metrics: Sequence[Metric] | None = None,
        *,
        environment: SimulatorAdapter | None = None,
    ):
        if simulator is not None and environment is not None:
            raise ValueError("provide simulator or the legacy environment argument, not both")
        self.simulator = simulator or environment
        if self.simulator is None:
            raise ValueError("a simulator adapter is required")
        if policy is None:
            raise ValueError("a policy adapter is required")
        self.environment = self.simulator  # AeroLoop 0.1 compatibility attribute.
        self.policy = policy
        self.rollout = rollout or RolloutConfig()
        self.metric_config = metrics or MetricConfig()
        self.observers = list(observers or [])
        self.custom_metrics = list(custom_metrics or [])

    def _observer_call(self, method: str, *args):
        values = []
        for observer in self.observers:
            try:
                values.append(getattr(observer, method)(*args))
            except Exception as exc:
                if self.rollout.fail_on_observer_error:
                    raise
                print(f"[observer warning] {type(observer).__name__}.{method}: {type(exc).__name__}: {exc}")
        return values

    def run_episode(self, episode: EpisodeSpec) -> EpisodeResult:
        records: list[StepRecord] = []
        image_history = []
        view_history = []
        action_history: list[CanonicalAction] = []
        state_history = []
        queue: deque[tuple[CanonicalAction, int, int, float | None, dict]] = deque()
        inference_call = 0
        termination = TerminationReason.MAX_STEPS
        error = None

        tracker = MetricSuite(episode, self.metric_config, self.custom_metrics)
        observation = Observation(
            rgb=None,
            pose=episode.start_pose,
            relative_state=(0.0, 0.0, 0.0, 0.0),
            step_index=0,
        )

        try:
            tracker.reset()
            self.policy.reset(episode)
            observation = self.simulator.reset(episode)
            self._observer_call("on_episode_start", episode, observation)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            metrics = tracker.finalize_builtin(TerminationReason.ERROR, observation.pose, 0)
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
                        view_history=tuple(view_history),
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
                        queue.append(
                            (
                                action,
                                inference_call,
                                action_index,
                                elapsed_ms if action_index == 0 else None,
                                dict(chunk.metadata),
                            )
                        )

                action, call_index, action_index, inference_ms, policy_metadata = queue.popleft()
                stop_requested = action.stop >= self.rollout.stop_threshold
                effective_action = action
                if stop_requested and not self.rollout.execute_motion_on_stop:
                    effective_action = action.zero_motion()

                before_observation = observation
                before = before_observation.pose
                transition = self.simulator.execute(effective_action)
                observation = transition.observation
                distances = tracker.update(
                    before,
                    observation.pose,
                    action=effective_action,
                    collision=transition.collision,
                    info=transition.info,
                )
                self.policy.on_action_executed(effective_action, transition)

                record = StepRecord(
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
                records.append(record)
                image_history.append(before_observation.rgb)
                view_history.append(dict(before_observation.images))
                action_history.append(effective_action)
                state_history.append(before_observation.relative_state)

                observer_values = self._observer_call("on_step", episode, observation, record)
                if any(value is False for value in observer_values):
                    termination = TerminationReason.USER_ABORT
                    break

                if transition.collision and self.rollout.terminate_on_collision:
                    termination = TerminationReason.COLLISION
                    break
                if stop_requested:
                    termination = TerminationReason.STOP
                    break
                if (
                    self.rollout.terminate_on_success
                    and distances[self.metric_config.distance_mode] < self.metric_config.success_distance
                ):
                    termination = TerminationReason.SUCCESS
                    break
            else:
                termination = TerminationReason.MAX_STEPS
        except Exception as exc:
            termination = TerminationReason.ERROR
            error = f"{type(exc).__name__}: {exc}"

        try:
            metrics = tracker.finalize(termination, observation.pose, len(records))
        except Exception as exc:
            termination = TerminationReason.ERROR
            metric_error = f"metric finalize failed: {type(exc).__name__}: {exc}"
            error = f"{error}; {metric_error}" if error else metric_error
            metrics = tracker.finalize_builtin(termination, observation.pose, len(records))
        result = EpisodeResult(
            episode_id=episode.episode_id,
            env_name=episode.env_name,
            termination_reason=termination,
            metrics=metrics,
            steps=tuple(records) if self.rollout.include_steps_in_result else (),
            error=error,
        )
        self._observer_call("on_episode_end", episode, observation, result)
        artifacts = {}
        for observer in self.observers:
            try:
                artifacts.update(observer.artifacts())
            except Exception:
                pass
        if artifacts:
            result = EpisodeResult(
                episode_id=result.episode_id,
                env_name=result.env_name,
                termination_reason=result.termination_reason,
                metrics=result.metrics,
                steps=result.steps,
                error=result.error,
                artifacts=artifacts,
            )
        return result
