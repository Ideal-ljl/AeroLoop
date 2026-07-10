import math
import unittest

from uav_eval.geometry import apply_body_action, relative_state
from uav_eval.types import CanonicalAction, Pose


class GeometryTest(unittest.TestCase):
    def test_strafe_does_not_change_yaw(self):
        pose = Pose(0, 0, 0, 0.25)
        result = apply_body_action(pose, CanonicalAction(0, 2, 0, 0, 0))
        self.assertAlmostEqual(result.yaw, 0.25)
        self.assertAlmostEqual(result.x, -2 * math.sin(0.25))
        self.assertAlmostEqual(result.y, 2 * math.cos(0.25))

    def test_explicit_yaw_delta(self):
        result = apply_body_action(Pose(0, 0, 0, math.pi - 0.1), CanonicalAction(0, 0, 0, 0.2, 0))
        self.assertAlmostEqual(result.yaw, -math.pi + 0.1)

    def test_relative_state_uses_start_frame(self):
        origin = Pose(10, 20, 3, math.pi / 2)
        pose = Pose(10, 22, 5, math.pi)
        state = relative_state(pose, origin)
        self.assertAlmostEqual(state[0], 2)
        self.assertAlmostEqual(state[1], 0, places=6)
        self.assertAlmostEqual(state[2], 2)
        self.assertAlmostEqual(state[3], math.pi / 2)


if __name__ == "__main__":
    unittest.main()
