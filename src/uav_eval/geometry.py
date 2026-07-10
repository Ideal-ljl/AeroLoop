from __future__ import annotations

import math
from typing import Sequence

from .types import CanonicalAction, Pose


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def apply_body_action(pose: Pose, action: CanonicalAction) -> Pose:
    """Apply body-frame translation without treating strafing as yaw."""
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    return Pose(
        x=pose.x + action.dx * cos_yaw - action.dy * sin_yaw,
        y=pose.y + action.dx * sin_yaw + action.dy * cos_yaw,
        z=pose.z + action.dz,
        yaw=wrap_angle(pose.yaw + action.d_yaw),
    )


def distance(a: Sequence[float], b: Sequence[float], dimensions: int = 3) -> float:
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(dimensions)))


def relative_state(pose: Pose, origin: Pose) -> tuple[float, float, float, float]:
    dx = pose.x - origin.x
    dy = pose.y - origin.y
    cos_yaw = math.cos(origin.yaw)
    sin_yaw = math.sin(origin.yaw)
    return (
        dx * cos_yaw + dy * sin_yaw,
        -dx * sin_yaw + dy * cos_yaw,
        pose.z - origin.z,
        wrap_angle(pose.yaw - origin.yaw),
    )
