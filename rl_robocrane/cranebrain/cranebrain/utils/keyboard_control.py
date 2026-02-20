# cranebrain/utils/keyboard_control.py

import numpy as np
import pygame


class KeyboardController:
    """
    Opens a tiny pygame window that reliably captures keyboard input.

    Keys:
        W/S → +x / -x
        A/D → +y / -y
        E/Q → +z / -z

        R → slow mode
        N → normal mode
        T → turbo mode

    Output:
        action (3,) in [-1, 1] float32
    """

    def __init__(self, slow=0.3, normal=1.0, turbo=2.0):
        pygame.init()
        pygame.display.set_caption("Robocrane Teleop")

        # Small UI window
        self.screen = pygame.display.set_mode((420, 260))
        self.font = pygame.font.SysFont("monospace", 18)

        # speed multipliers
        self.slow = slow
        self.normal = normal
        self.turbo = turbo
        self.scale = normal

        self.running = True
        self._draw_help()

    # ------------------------------------------------------------
    # UI overlay with key assignments
    # ------------------------------------------------------------
    def _draw_help(self):
        self.screen.fill((20, 20, 20))

        lines = [
            "Robocrane Keyboard Teleop",
            "--------------------------------",
            "W / S  : +X / -X",
            "A / D  : +Y / -Y",
            "E / Q  : +Z / -Z",
            "",
            "R      : slow mode",
            "N      : normal mode",
            "T      : turbo mode",
            "",
            f"Speed scale: {self.scale:.2f}",
            "",
            "Focus this window to control robot!"
        ]

        y = 15
        for line in lines:
            text = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(text, (20, y))
            y += 22

        pygame.display.flip()

    # ------------------------------------------------------------
    # Print key assignments (terminal version)
    # ------------------------------------------------------------
    def print_key_assignments(self):
        print("\n📌  Keyboard Teleoperation Active (Pygame Window)")
        print("===================================")
        print(" Movement:")
        print("   W / S  →  +X / -X (forward / back)")
        print("   A / D  →  +Y / -Y (left / right)")
        print("   E / Q  →  +Z / -Z (up / down)")
        print()
        print(" Speed modes:")
        print("   R      →  Slow mode")
        print("   N      →  Normal mode")
        print("   T      →  Turbo mode")
        print()
        print(" Other:")
        print("   Close window → Stop teleop")
        print("===================================\n")

    # ------------------------------------------------------------
    # Main action generator
    # ------------------------------------------------------------
    def get_action(self):
        """
        Returns a (3,) float32 velocity command in [-1, 1].
        """

        if not self.running:
            return np.zeros(3, dtype=np.float32)

        # process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()

        vx = vy = vz = 0.0

        # movement
        if keys[pygame.K_w]:
            vx = +1.0
        if keys[pygame.K_s]:
            vx = -1.0
        if keys[pygame.K_a]:
            vy = +1.0
        if keys[pygame.K_d]:
            vy = -1.0
        if keys[pygame.K_e]:
            vz = +1.0
        if keys[pygame.K_q]:
            vz = -1.0

        # speed modes
        if keys[pygame.K_r]:
            self.scale = self.slow
        if keys[pygame.K_n]:
            self.scale = self.normal
        if keys[pygame.K_t]:
            self.scale = self.turbo

        # update UI
        self._draw_help()

        # return action
        return np.array([vx, vy, vz], dtype=np.float32) * self.scale
