# cranebrain/utils/gamepad_control.py

import numpy as np
from inputs import get_gamepad


class GamepadController:
    """
    Reads Xbox controller events and produces a 7D normalized action vector
    for the RobocraneEnv.
    
    Mapping:
        Left stick X → cartesian v_x  (left-right)
        Left stick Y → cartesian v_y  (forward-back)
        Right trigger  → v_z up
        Left trigger   → v_z down

    Output:
        action (7,) in [-1,1]  (your env handles denormalizing)
    """

    def __init__(self, deadzone=0.12):
        self.deadzone = deadzone

        # Store last values
        self.lx = 0.0   # left stick X
        self.ly = 0.0   # left stick Y
        self.rt = 0.0   # right trigger
        self.lt = 0.0   # left trigger

        # For optional modes
        self.slow = 1.0

    # -------------------------
    # Internal helpers
    # -------------------------
    def _apply_deadzone(self, v):
        if abs(v) < self.deadzone:
            return 0.0
        return v

    def _normalize_axis(self, value):
        # Convert [-32768, 32767] to [-1, 1]
        return float(value) / 32767.0

    # -------------------------
    # Poll for updates
    # -------------------------
    def poll(self):
        """
        Poll gamepad events. Non-blocking, safe if controller unplugged.
        """
        try:
            events = get_gamepad()
        except Exception:
            # No controller connected or transient USB error
            return

        for event in events:
            code = event.code
            val = event.state

            # Left stick XY
            if code == "ABS_X":
                self.lx = self._normalize_axis(val)
                self.lx = self._apply_deadzone(self.lx)

            elif code == "ABS_Y":
                self.ly = -self._normalize_axis(val)  # invert Y
                self.ly = self._apply_deadzone(self.ly)

            # Triggers (0–255 range)
            elif code == "ABS_RZ":
                self.rt = val / 255.0    # up velocity
            elif code == "ABS_Z":
                self.lt = val / 255.0    # down velocity

            # Buttons for optional scaling
            elif code == "BTN_SOUTH":   # A button = slow mode
                self.slow = 0.4
            elif code == "BTN_EAST":    # B button = normal mode
                self.slow = 1.0

    # -------------------------
    # Public: Get action
    # -------------------------
    def get_action(self):
        """
        Returns a 7D action vector for RobocraneEnv in [-1,1].

        Cartesian velocities:
            vx = lx
            vy = ly
            vz = rt - lt

        The remaining joints are zero.
        """
        self.poll()

        vx = self.lx * self.slow
        vy = self.ly * self.slow
        vz = (self.rt - self.lt) * self.slow

        # Convert into joint velocity command space:
        # your robocrane_env expects a 7D vector 
        # in normalized form (will be denormalized inside env).
        #
        # For debugging we leave the remaining joints zero.
        action = np.zeros(3, dtype=np.float32)

        # You can remap this later via IK:
        action[0] = vx
        action[1] = vy
        action[2] = vz

        return action
