"""
Stanley Lateral Controller and Tier4 Velocity Profiler
Optimized for hardware control via ESP32 PWM on NVIDIA Jetson Orin Nano.

Implements:
  1. Stanley Steering Control Law (from Autonomous-Racing/stanley_avoidance)
     delta = psi + arctan( (k_E * e_y) / (v + k_v) )
  2. Slew-rate limiter & EMA filter to prevent servo jitter
  3. Curvature-adaptive throttle profiler (from Autoware Tier4 Velocity Planner)
"""

import numpy as np


class StanleyController:
    """
    Computes smooth steering angle (0-180 deg, 90=center) and throttle (-100 to +100).
    """

    def __init__(self, k_E=1.8, k_v=0.8, k_heading=1.2, max_steer_rate=14.0):
        # Stanley gains
        self.k_E = k_E              # Cross-track error gain
        self.k_v = k_v              # Velocity softening constant
        self.k_heading = k_heading  # Heading error gain
        self.max_steer_rate = max_steer_rate  # Max degrees delta allowed per control cycle

        # State memory
        self.prev_steering_angle = 90.0
        self.prev_throttle = 0.0

        # Hardware calibration for ESP32
        self.CENTER_STEER = 90.0
        self.MIN_STEER = 15.0
        self.MAX_STEER = 165.0

    def compute_steering(self, target_lateral_offset_m, target_heading_deg, current_speed_ms=1.5):
        """
        Calculates Stanley steering angle.

        Args:
            target_lateral_offset_m: lateral error e_y (+right, -left) in meters
            target_heading_deg: target heading angle psi in degrees
            current_speed_ms: vehicle forward velocity in m/s

        Returns:
            servo_angle: integer [0, 180], 90 is center
        """
        # 1. Heading error term (radians)
        psi_rad = np.radians(target_heading_deg) * self.k_heading

        # 2. Cross-track error term (radians)
        # In Stanley law: delta_ct = arctan( (k_E * e_y) / (v + k_v) )
        e_y = target_lateral_offset_m
        crosstrack_rad = np.arctan2(self.k_E * e_y, current_speed_ms + self.k_v)

        # 3. Total steering correction
        total_delta_rad = psi_rad + crosstrack_rad
        total_delta_deg = np.degrees(total_delta_rad)

        # Convert to servo angle: 90 is straight, <90 is left, >90 is right
        raw_servo_angle = self.CENTER_STEER + total_delta_deg

        # 4. Slew-rate limiting to prevent violent oscillations
        delta_change = raw_servo_angle - self.prev_steering_angle
        clamped_change = np.clip(delta_change, -self.max_steer_rate, self.max_steer_rate)
        filtered_angle = self.prev_steering_angle + clamped_change

        # 5. Exponential Moving Average for ultra-smooth output
        alpha = 0.65
        final_angle = alpha * filtered_angle + (1.0 - alpha) * self.prev_steering_angle
        final_angle = np.clip(final_angle, self.MIN_STEER, self.MAX_STEER)

        self.prev_steering_angle = final_angle
        return int(round(final_angle))

    def compute_throttle(self, driving_state, steering_angle, base_cruise_throttle=55):
        """
        Curvature-adaptive throttle profiling (inspired by Autoware Tier4).

        Args:
            driving_state: active state machine state
            steering_angle: current steering angle (0-180)
            base_cruise_throttle: default cruise power

        Returns:
            throttle: integer (-100 to 100)
        """
        steer_deviation = abs(steering_angle - self.CENTER_STEER)

        if driving_state in ("RED LIGHT", "STOP SIGN"):
            target_thr = 0.0

        elif "EMERGENCY" in driving_state or "AEB" in driving_state:
            # Instantaneous emergency brake cutoff & reverse pulse
            self.prev_throttle = -25.0
            return -25

        elif "BRAKING" in driving_state:
            target_thr = 0.0

        elif driving_state == "REVERSE":
            target_thr = -25.0

        elif "YIELD_CROSSING" in driving_state:
            target_thr = 22.0  # Slow creep for crossing pedestrian/cattle

        elif "DODGE" in driving_state or "AVOID" in driving_state:
            # Scale throttle down on aggressive steering evasions
            curvature_factor = max(0.45, 1.0 - (steer_deviation / 80.0))
            target_thr = 35.0 * curvature_factor

        elif driving_state == "SLOW":
            target_thr = 30.0

        else:  # CRUISING
            # Gently reduce throttle if steering slightly into a curve
            speed_curve = max(0.65, 1.0 - (steer_deviation / 100.0) * 0.40)
            target_thr = base_cruise_throttle * speed_curve

        # Smooth throttle transitions
        thr_alpha = 0.50
        smooth_thr = thr_alpha * target_thr + (1.0 - thr_alpha) * self.prev_throttle
        self.prev_throttle = smooth_thr
        return int(round(smooth_thr))
