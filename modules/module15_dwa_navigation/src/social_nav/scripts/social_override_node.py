#!/usr/bin/env python3
"""
Module 15 — Social Override Node
===================================
Intercepts raw DWA velocity commands and applies social constraints
based on human proximity before publishing to the actual robot.

Pipeline:
  Nav2/DWA  →  /cmd_vel_raw  →  [THIS NODE]  →  /cmd_vel  →  Robot
                                      ↑
                           /human_positions
                           /social_costmap_raw

How it works:
  1. DWA planner handles obstacle avoidance → publishes /cmd_vel_raw
  2. This node reads human positions and social costmap
  3. Applies 3 layers of social behaviour:

     Layer A — Speed Scaling (proxemics zones)
       Intimate  < 0.45m → STOP (multiplier=0.0)
       Personal  0.45–1.2m → 20% speed
       Social    1.2–3.6m  → 60% speed
       Public    > 3.6m    → 100% speed

     Layer B — Lookahead Cost Check
       Predicts robot position 0.5s ahead using current velocity
       Looks up social costmap cost at predicted position
       Further reduces speed if lookahead position is high-cost

     Layer C — Lateral Avoidance Bias
       If a human is within personal zone AND ahead of robot
       Adds angular velocity bias to steer the robot to pass
       BEHIND the human (not through their front/personal space)

  4. Applies smooth transitions (no sudden speed jumps)
  5. Publishes final /cmd_vel

Safety:
  - If /human_positions goes stale > 2s → revert to full speed
    (prevents robot freezing if tracker dies)
  - Emergency stop if human < INTIMATE_RADIUS and closing fast

Topics:
  SUB  /cmd_vel_raw          geometry_msgs/Twist  (from Nav2/DWA)
  SUB  /human_positions      std_msgs/String      (JSON from tracker)
  SUB  /social_costmap_raw   std_msgs/String      (JSON from costmap)
  PUB  /cmd_vel              geometry_msgs/Twist  (to robot)
  PUB  /social_status        std_msgs/String      (debug info JSON)

Architecture note:
  In the launch file, remap Nav2's default /cmd_vel output to /cmd_vel_raw
  so this node sits between Nav2 and the robot base.

Standalone test:
  python3 social_override_node.py --test

ROS2:
  python3 social_override_node.py --ros
"""

import os
import sys
import json
import math
import time
import argparse
import threading
import numpy as np

# Optional ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# Proxemics zone radii (metres)
INTIMATE_RADIUS  = 0.45
PERSONAL_RADIUS  = 1.20
SOCIAL_RADIUS    = 3.60

# Speed multipliers per zone
SPEED_MULT = {
    "intimate":  0.0,    # full stop
    "personal":  0.20,   # 20% of planned speed
    "social":    0.60,   # 60% of planned speed
    "public":    1.00,   # full speed
}

# Costmap-based speed reduction
# If predicted position cost > threshold, reduce speed
COSTMAP_HIGH_THRESHOLD  = 70    # cost 70–100 → slow down significantly
COSTMAP_MED_THRESHOLD   = 40    # cost 40–70 → slow down moderately
COSTMAP_HIGH_MULT       = 0.15  # speed multiplier when cost > HIGH_THRESHOLD
COSTMAP_MED_MULT        = 0.45  # speed multiplier when cost > MED_THRESHOLD

# Lookahead time for predicted position check
LOOKAHEAD_TIME_S = 0.5

# Lateral avoidance
LATERAL_GAIN       = 0.8    # how strongly to bias angular velocity
MAX_LATERAL_BIAS   = 0.6    # max angular vel bias (rad/s)
LATERAL_ZONE       = PERSONAL_RADIUS   # only apply lateral avoidance in personal zone

# Smoothing
ALPHA_SMOOTHING    = 0.35   # low-pass filter on speed multiplier
                             # 0 = no smoothing, 1 = never changes

# Safety watchdog
STALE_TIMEOUT_S    = 2.0    # revert to full speed if no updates for this long

# Max velocities (TurtleBot3 Burger limits)
MAX_LINEAR_VEL     = 0.22   # m/s
MAX_ANGULAR_VEL    = 2.84   # rad/s

# Topics
<<<<<<< HEAD
TOPIC_CMD_RAW      = "/cmd_vel_nav"
=======
TOPIC_CMD_RAW      = "/cmd_vel_raw"
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
TOPIC_CMD_OUT      = "/cmd_vel"
TOPIC_HUMANS       = "/human_positions"
TOPIC_COSTMAP_RAW  = "/social_costmap_raw"
TOPIC_STATUS       = "/social_status"


# ─────────────────────────────────────────────────────────────
# Costmap Lookup Helper
# ─────────────────────────────────────────────────────────────

class CostmapLookup:
    """
    Fast lookup of social costmap cost at a world (x, y) position.
    Parses the JSON costmap from /social_costmap_raw once and
    caches the numpy grid for O(1) lookups.
    """

    def __init__(self):
        self._grid       = None
        self._resolution = 0.1
        self._origin_x   = -5.0
        self._origin_y   = -5.0
        self._width      = 100
        self._height     = 100
        self._lock       = threading.Lock()
        self._last_update = 0.0

    def update_from_json(self, json_str: str):
        """Parse and cache new costmap data."""
        try:
            d = json.loads(json_str)
            flat = np.array(d["data"], dtype=np.float32)
            grid = flat.reshape(d["height"], d["width"])
            with self._lock:
                self._grid       = grid
                self._resolution = d["resolution"]
                self._origin_x   = d["origin_x"]
                self._origin_y   = d["origin_y"]
                self._width      = d["width"]
                self._height     = d["height"]
                self._last_update = time.time()
        except Exception as e:
            pass  # keep previous grid on parse error

    def get_cost(self, wx: float, wy: float) -> int:
        """Return cost [0–100] at world position (wx, wy)."""
        with self._lock:
            if self._grid is None:
                return 0
            col = int((wx - self._origin_x) / self._resolution)
            row = int((wy - self._origin_y) / self._resolution)
            if 0 <= col < self._width and 0 <= row < self._height:
                return int(self._grid[row, col])
        return 0

    @property
    def is_stale(self) -> bool:
        return (time.time() - self._last_update) > STALE_TIMEOUT_S


# ─────────────────────────────────────────────────────────────
# Social Velocity Controller
# ─────────────────────────────────────────────────────────────

class SocialVelocityController:
    """
    Core logic for computing socially-aware velocity from:
      - Raw DWA velocity command
      - List of human states
      - Social costmap

    Completely decoupled from ROS2 for clean testing.
    """

    def __init__(self):
        self._smooth_mult  = 1.0     # smoothed speed multiplier
        self._costmap      = CostmapLookup()
        self._last_humans  = []
        self._last_human_t = 0.0
        self._lock         = threading.Lock()

    # ── State updates ──

    def update_humans(self, json_str: str):
        try:
            humans = json.loads(json_str)
            with self._lock:
                self._last_humans  = humans
                self._last_human_t = time.time()
        except Exception:
            pass

    def update_costmap(self, json_str: str):
        self._costmap.update_from_json(json_str)

    # ── Zone classification ──

    @staticmethod
    def classify_zone(dist: float) -> str:
        if dist < INTIMATE_RADIUS:
            return "intimate"
        elif dist < PERSONAL_RADIUS:
            return "personal"
        elif dist < SOCIAL_RADIUS:
            return "social"
        return "public"

    # ── Nearest human ──

    def _nearest_human(self, robot_x: float, robot_y: float):
        """Return (distance, human_dict) of the nearest human. None if no humans."""
        with self._lock:
            humans = list(self._last_humans)
        if not humans:
            return None, None
        best_dist, best_human = float("inf"), None
        for h in humans:
            if not h.get("is_human", True):
                continue  # skip dynamic obstacles for zone logic
            dx = h["x"] - robot_x
            dy = h["y"] - robot_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist, best_human = dist, h
        return best_dist, best_human

    # ── Layer A: speed multiplier from nearest human zone ──

    def _zone_speed_mult(self, dist: float) -> float:
        zone = self.classify_zone(dist)
        return SPEED_MULT[zone]

    # ── Layer B: costmap lookahead speed mult ──

    def _costmap_speed_mult(self, robot_x: float, robot_y: float,
                             vx: float, vy: float) -> float:
        """
        Predict robot position LOOKAHEAD_TIME_S ahead.
        Look up social cost there.
        Return speed multiplier based on cost.
        """
        pred_x = robot_x + vx * LOOKAHEAD_TIME_S
        pred_y = robot_y + vy * LOOKAHEAD_TIME_S
        cost   = self._costmap.get_cost(pred_x, pred_y)

        if cost >= COSTMAP_HIGH_THRESHOLD:
            return COSTMAP_HIGH_MULT
        elif cost >= COSTMAP_MED_THRESHOLD:
            return COSTMAP_MED_MULT
        return 1.0

    # ── Layer C: lateral avoidance bias ──

    def _lateral_bias(self, robot_x: float, robot_y: float,
                       robot_yaw: float, raw_linear: float) -> float:
        """
        Compute angular velocity bias to steer robot away from
        the front-space of nearby humans.

        Returns angular velocity bias (rad/s). Positive=left, negative=right.
        """
        if raw_linear < 0.01:
            return 0.0  # not moving, no point steering

        with self._lock:
            humans = list(self._last_humans)

        total_bias = 0.0
        for h in humans:
            if not h.get("is_human", True):
                continue

            dx   = h["x"] - robot_x
            dy   = h["y"] - robot_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > LATERAL_ZONE or dist < 0.01:
                continue

            # Only bias if human is roughly ahead (within ±90° of robot heading)
            heading_to_human = math.atan2(dy, dx)
            angle_diff = abs(
                math.atan2(math.sin(heading_to_human - robot_yaw),
                           math.cos(heading_to_human - robot_yaw))
            )
            if angle_diff > math.pi / 2:
                continue   # human is behind robot

            # Which side of human's path is the robot on?
            h_yaw  = h.get("yaw", 0.0)
            side_x = -math.sin(h_yaw)
            side_y =  math.cos(h_yaw)
            side_proj = dx * side_x + dy * side_y

            # Bias: steer away from human's front zone
            # If robot is to left of human (side_proj > 0) → keep going left (positive omega)
            # If robot is to right (side_proj < 0) → keep going right (negative omega)
            bias_strength = LATERAL_GAIN * (1.0 - dist / LATERAL_ZONE)
            bias = -side_proj / (dist + 1e-6) * bias_strength
            total_bias += bias

        return float(np.clip(total_bias, -MAX_LATERAL_BIAS, MAX_LATERAL_BIAS))

    # ── Master compute function ──

    def compute(self,
                raw_vx: float, raw_omega: float,
                robot_x: float = 0.0,
                robot_y: float = 0.0,
                robot_yaw: float = 0.0) -> dict:
        """
        Compute socially-aware velocity from raw DWA command.

        Args:
            raw_vx    : linear velocity from DWA (m/s)
            raw_omega : angular velocity from DWA (rad/s)
            robot_x/y : current robot world position (from /odom)
            robot_yaw : current robot heading (radians)

        Returns:
            {
              "vx"           : float  (final linear velocity)
              "omega"        : float  (final angular velocity)
              "multiplier"   : float  (applied speed multiplier)
              "zone"         : str    (current proxemics zone)
              "nearest_dist" : float  (distance to nearest human)
              "layer_a_mult" : float  (zone-based mult)
              "layer_b_mult" : float  (costmap lookahead mult)
              "lateral_bias" : float  (angular vel bias)
            }
        """
        # Check human data freshness
        data_age = time.time() - self._last_human_t
        if data_age > STALE_TIMEOUT_S:
            # No fresh human data → safe to run at full speed
            return {
                "vx": float(np.clip(raw_vx,    -MAX_LINEAR_VEL, MAX_LINEAR_VEL)),
                "omega": float(np.clip(raw_omega, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)),
                "multiplier": 1.0,
                "zone": "public",
                "nearest_dist": 999.0,
                "layer_a_mult": 1.0,
                "layer_b_mult": 1.0,
                "lateral_bias": 0.0,
            }

        # ── Layer A: zone-based speed ──
        nearest_dist, nearest_human = self._nearest_human(robot_x, robot_y)
        if nearest_dist is None:
            nearest_dist = 999.0
        zone       = self.classify_zone(nearest_dist)
        mult_a     = self._zone_speed_mult(nearest_dist)

        # ── Layer B: costmap lookahead ──
        # Use robot's current heading to estimate velocity direction
        vx_world = raw_vx * math.cos(robot_yaw)
        vy_world = raw_vx * math.sin(robot_yaw)
        mult_b   = self._costmap_speed_mult(robot_x, robot_y, vx_world, vy_world)

        # ── Combine A and B: take the more restrictive ──
        target_mult = min(mult_a, mult_b)

        # ── Smooth multiplier to avoid jerky speed changes ──
        self._smooth_mult = (ALPHA_SMOOTHING * self._smooth_mult
                             + (1 - ALPHA_SMOOTHING) * target_mult)

        final_mult = self._smooth_mult

        # ── Apply multiplier to linear velocity ──
        final_vx = raw_vx * final_mult

        # ── Layer C: lateral bias ──
        bias       = self._lateral_bias(robot_x, robot_y, robot_yaw, abs(raw_vx))
        final_omega = raw_omega + bias

        # ── Hard clamp on intimate zone: full stop ──
        if zone == "intimate":
            final_vx    = 0.0
            final_omega = 0.0  # also stop rotation to avoid confusing the human

        # ── Clamp to robot hardware limits ──
        final_vx    = float(np.clip(final_vx,    -MAX_LINEAR_VEL,  MAX_LINEAR_VEL))
        final_omega = float(np.clip(final_omega, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL))

        return {
            "vx":            final_vx,
            "omega":         final_omega,
            "multiplier":    round(final_mult, 3),
            "zone":          zone,
            "nearest_dist":  round(nearest_dist, 3),
            "layer_a_mult":  round(mult_a, 3),
            "layer_b_mult":  round(mult_b, 3),
            "lateral_bias":  round(bias, 3),
        }


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class SocialOverrideNode(Node):
    """
    ROS2 social override node.

    Intercepts /cmd_vel_raw from Nav2/DWA planner,
    applies social constraints, publishes to /cmd_vel.

    LAUNCH FILE NOTE:
    Remap Nav2 local planner output:
      remappings=[('/cmd_vel', '/cmd_vel_raw')]
    This node then reads /cmd_vel_raw and publishes /cmd_vel.
    """

    def __init__(self):
        super().__init__("social_override_node")

        # Declare parameters
        self.declare_parameter("intimate_radius",  INTIMATE_RADIUS)
        self.declare_parameter("personal_radius",  PERSONAL_RADIUS)
        self.declare_parameter("social_radius",    SOCIAL_RADIUS)
        self.declare_parameter("lookahead_time",   LOOKAHEAD_TIME_S)
        self.declare_parameter("lateral_gain",     LATERAL_GAIN)

        self.controller   = SocialVelocityController()
        self._robot_x     = 0.0
        self._robot_y     = 0.0
        self._robot_yaw   = 0.0
        self._odom_lock   = threading.Lock()

        # Publishers
        self.pub_cmd    = self.create_publisher(Twist,  TOPIC_CMD_OUT, 10)
        self.pub_status = self.create_publisher(String, TOPIC_STATUS,  10)

        # Subscribers
        self.create_subscription(Twist,  TOPIC_CMD_RAW,    self._cmd_raw_cb,   10)
        self.create_subscription(String, TOPIC_HUMANS,     self._humans_cb,    10)
        self.create_subscription(String, TOPIC_COSTMAP_RAW,self._costmap_cb,   10)

        # Odom subscriber for robot pose (needed for lookahead and lateral bias)
        try:
            from nav_msgs.msg import Odometry
            self.create_subscription(
                Odometry, "/odom", self._odom_cb, 10
            )
            self.get_logger().info("Subscribed to /odom for robot pose")
        except Exception:
            self.get_logger().warning("nav_msgs not available — using (0,0,0) for robot pose")

        self.get_logger().info(
            f"Social override node started\n"
            f"  SUB  : {TOPIC_CMD_RAW}\n"
            f"  SUB  : {TOPIC_HUMANS}\n"
            f"  SUB  : {TOPIC_COSTMAP_RAW}\n"
            f"  PUB  : {TOPIC_CMD_OUT}\n"
            f"  PUB  : {TOPIC_STATUS}\n"
            f"  Zones: intimate<{INTIMATE_RADIUS}m  "
            f"personal<{PERSONAL_RADIUS}m  social<{SOCIAL_RADIUS}m"
        )

    def _odom_cb(self, msg):
        """Extract robot pose from odometry."""
        from nav_msgs.msg import Odometry
        import math
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        # Extract yaw from quaternion
        q  = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._odom_lock:
            self._robot_x   = px
            self._robot_y   = py
            self._robot_yaw = yaw

    def _humans_cb(self, msg: String):
        self.controller.update_humans(msg.data)

    def _costmap_cb(self, msg: String):
        self.controller.update_costmap(msg.data)

    def _cmd_raw_cb(self, msg: Twist):
        """
        Main callback — called every time DWA publishes a velocity command.
        Applies social constraints and publishes final /cmd_vel.
        """
        with self._odom_lock:
            rx, ry, ryaw = self._robot_x, self._robot_y, self._robot_yaw

        result = self.controller.compute(
            raw_vx    = msg.linear.x,
            raw_omega = msg.angular.z,
            robot_x   = rx,
            robot_y   = ry,
            robot_yaw = ryaw,
        )

        # Publish final cmd_vel
        out             = Twist()
        out.linear.x    = result["vx"]
        out.angular.z   = result["omega"]
        self.pub_cmd.publish(out)

        # Publish debug status
        status_msg      = String()
        status_msg.data = json.dumps({
            "zone":          result["zone"],
            "nearest_dist":  result["nearest_dist"],
            "multiplier":    result["multiplier"],
            "layer_a":       result["layer_a_mult"],
            "layer_b":       result["layer_b_mult"],
            "lateral_bias":  result["lateral_bias"],
            "raw_vx":        round(msg.linear.x, 3),
            "final_vx":      round(result["vx"], 3),
        })
        self.pub_status.publish(status_msg)

        # Log only when zone changes or speed limited
        if result["zone"] != "public" or result["multiplier"] < 0.95:
            self.get_logger().info(
                f"[{result['zone']:8s}] dist={result['nearest_dist']:.2f}m  "
                f"mult={result['multiplier']:.2f}  "
                f"vx: {msg.linear.x:.2f}→{result['vx']:.2f}  "
                f"bias={result['lateral_bias']:.2f}"
            )

    def destroy_node(self):
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# Standalone Tests
# ─────────────────────────────────────────────────────────────

def run_test():
    print("=" * 60)
    print("  Social Override Node — Standalone Tests")
    print("=" * 60)

    ctrl = SocialVelocityController()

    # Feed human positions
    def feed_human(hx, hy, yaw=0.0, speed=0.0, name="human_1"):
        ctrl.update_humans(json.dumps([{
            "id": name, "x": hx, "y": hy,
            "vx": speed*math.cos(yaw), "vy": speed*math.sin(yaw),
            "yaw": yaw, "speed": speed, "is_human": True
        }]))

    # ── T1: No humans → full speed ──
    print("\nT1 No humans → full speed")
    ctrl.update_humans(json.dumps([]))
    ctrl._last_human_t = time.time()  # mark as fresh but empty
    r = ctrl.compute(0.2, 0.0, robot_x=0.0, robot_y=0.0)
    print(f"  vx={r['vx']:.3f}  mult={r['multiplier']:.2f}  zone={r['zone']}")
    assert r["zone"] == "public"
    assert r["vx"] > 0.15
    print("  ✅")

    # ── T2: Human in public zone → full speed ──
    print("\nT2 Human at 4.0m (public zone)")
    feed_human(4.0, 0.0)
    r = ctrl.compute(0.2, 0.0, robot_x=0.0, robot_y=0.0)
    print(f"  dist={r['nearest_dist']:.2f}m  zone={r['zone']}  vx={r['vx']:.3f}")
    assert r["zone"] == "public"
    assert r["vx"] > 0.15
    print("  ✅")

    # ── T3: Human in social zone → 60% speed ──
    print("\nT3 Human at 2.0m (social zone)")
    feed_human(2.0, 0.0)
    # Run several steps to let smoother converge
    for _ in range(20):
        r = ctrl.compute(0.22, 0.0, robot_x=0.0, robot_y=0.0)
    print(f"  dist={r['nearest_dist']:.2f}m  zone={r['zone']}  "
          f"mult={r['multiplier']:.2f}  vx={r['vx']:.3f}")
    assert r["zone"] == "social"
    assert r["vx"] < 0.18   # reduced from 0.22
    print("  ✅")

    # ── T4: Human in personal zone → 20% speed ──
    print("\nT4 Human at 0.8m (personal zone)")
    feed_human(0.8, 0.0)
    for _ in range(20):
        r = ctrl.compute(0.22, 0.0, robot_x=0.0, robot_y=0.0)
    print(f"  dist={r['nearest_dist']:.2f}m  zone={r['zone']}  "
          f"mult={r['multiplier']:.2f}  vx={r['vx']:.3f}")
    assert r["zone"] == "personal"
    assert r["vx"] < 0.10
    print("  ✅")

    # ── T5: Human in intimate zone → STOP ──
    print("\nT5 Human at 0.3m (intimate zone) → FULL STOP")
    feed_human(0.3, 0.0)
    r = ctrl.compute(0.22, 0.5, robot_x=0.0, robot_y=0.0)
    print(f"  dist={r['nearest_dist']:.2f}m  zone={r['zone']}  "
          f"vx={r['vx']:.3f}  omega={r['omega']:.3f}")
    assert r["zone"] == "intimate"
    assert r["vx"] == 0.0
    assert r["omega"] == 0.0
    print("  ✅")

    # ── T6: Lateral avoidance bias ──
    print("\nT6 Lateral avoidance — human ahead-left, robot should steer left")
    ctrl2 = SocialVelocityController()
    ctrl2.update_humans(json.dumps([{
        "id": "h", "x": 0.9, "y": 0.2,  # slightly left of robot's path
        "vx": 1.0, "vy": 0.0,            # human walking +x
        "yaw": 0.0, "speed": 1.0, "is_human": True
    }]))
    ctrl2._last_human_t = time.time()
    r6 = ctrl2.compute(0.15, 0.0, robot_x=0.0, robot_y=0.0, robot_yaw=0.0)
    print(f"  lateral_bias={r6['lateral_bias']:.3f}  omega={r6['omega']:.3f}")
    print(f"  (bias should be non-zero to steer around human)")
    assert abs(r6["lateral_bias"]) > 0.0
    print("  ✅")

    # ── T7: Stale data fallback ──
    print("\nT7 Stale human data → full speed fallback")
    ctrl3 = SocialVelocityController()
    ctrl3.update_humans(json.dumps([{"id":"h","x":0.5,"y":0,"vx":0,"vy":0,"yaw":0,"speed":0,"is_human":True}]))
    ctrl3._last_human_t = time.time() - (STALE_TIMEOUT_S + 1.0)  # simulate stale
    r7 = ctrl3.compute(0.22, 0.0)
    print(f"  multiplier={r7['multiplier']:.2f}  zone={r7['zone']}  vx={r7['vx']:.3f}")
    assert r7["multiplier"] == 1.0
    assert r7["zone"] == "public"
    print("  ✅")

    # ── T8: Smooth transitions ──
    print("\nT8 Smooth transition — no sudden speed jumps")
    ctrl4 = SocialVelocityController()
    ctrl4.update_humans(json.dumps([{"id":"h","x":0.8,"y":0,"vx":0,"vy":0,"yaw":0,"speed":0,"is_human":True}]))
    ctrl4._last_human_t = time.time()
    speeds = []
    for _ in range(10):
        r8 = ctrl4.compute(0.22, 0.0)
        speeds.append(r8["vx"])
    max_jump = max(abs(speeds[i]-speeds[i-1]) for i in range(1, len(speeds)))
    print(f"  Speed sequence: {[round(s,3) for s in speeds]}")
    print(f"  Max step jump : {max_jump:.4f} m/s")
    assert max_jump < 0.10, f"Jump too large: {max_jump:.3f}"
    print("  ✅")

    # ── T9: Max velocity clamping ──
    print("\nT9 Hardware velocity clamping")
    ctrl5 = SocialVelocityController()
    ctrl5.update_humans(json.dumps([]))
    ctrl5._last_human_t = time.time()
    r9 = ctrl5.compute(5.0, 10.0)   # unrealistic values
    print(f"  Input vx=5.0 → output vx={r9['vx']:.3f} (max {MAX_LINEAR_VEL})")
    print(f"  Input omega=10.0 → output omega={r9['omega']:.3f} (max {MAX_ANGULAR_VEL})")
    assert abs(r9["vx"])    <= MAX_LINEAR_VEL
    assert abs(r9["omega"]) <= MAX_ANGULAR_VEL
    print("  ✅")

    # ── T10: Zone classification boundaries ──
    print("\nT10 Zone boundary classification")
    controller_static = SocialVelocityController()
    tests = [
        (0.0,   "intimate"),
        (0.44,  "intimate"),
        (0.45,  "personal"),
        (1.19,  "personal"),
        (1.20,  "social"),
        (3.59,  "social"),
        (3.60,  "public"),
        (10.0,  "public"),
    ]
    for dist, expected in tests:
        got = controller_static.classify_zone(dist)
        status = "✅" if got == expected else "❌"
        print(f"  {dist:.2f}m → {got:8s} (expect {expected:8s}) {status}")
        assert got == expected

    print()
    print("=" * 60)
    print("  ✅ All 10 Social Override tests passed")
    print("=" * 60)
    print()
    print("  Place at:")
    print("  modules/module15_dwa_navigation/src/social_nav/scripts/social_override_node.py")
    print()
    print("  In launch file — remap Nav2 output:")
    print("  remappings=[('/cmd_vel', '/cmd_vel_raw')]")
    print()
    print("  Monitor in real-time:")
    print("  ros2 topic echo /social_status")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Social override node")
    p.add_argument("--ros",  action="store_true", help="Run as ROS2 node")
    p.add_argument("--test", action="store_true", help="Run standalone tests")
    return p.parse_args()


def main():
    args = parse_args()

    if args.test:
        run_test()
        return

    if args.ros:
        if not ROS_AVAILABLE:
            print("❌ rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = SocialOverrideNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down social override node")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("Usage:")
        print("  python3 social_override_node.py --test")
        print("  python3 social_override_node.py --ros")


if __name__ == "__main__":
    main()
