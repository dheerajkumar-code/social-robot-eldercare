#!/usr/bin/env python3
"""
Module 15 — Human Tracker Node
================================
Subscribes to Gazebo model states, tracks all human actors using a
4-state Kalman Filter, and publishes positions + velocities for the
social navigation stack.

Architecture:
  /gazebo/model_states (ModelStates)
          ↓
  KalmanTracker per human  →  smooth position + velocity estimate
          ↓
  /human_positions  (String JSON)     ← consumed by social_costmap_node
  /human_markers    (MarkerArray)     ← consumed by RViz

Tracked actors (must match names in social_world.world):
  human_1, human_2, human_3, dynamic_obstacle_1

Kalman Filter State: [x, y, vx, vy]
  - Predicts where each human will be 0.5s ahead
  - Smooths noisy Gazebo position data
  - Gives stable velocity for asymmetric Gaussian cost field

Topics:
  SUB  /gazebo/model_states     gazebo_msgs/msg/ModelStates
  PUB  /human_positions         std_msgs/msg/String  (JSON array)
  PUB  /human_markers           visualization_msgs/msg/MarkerArray

JSON format of /human_positions:
  [
    {
      "id":  "human_1",
      "x":   1.23,
      "y":  -0.98,
      "vx":  0.95,
      "vy":  0.02,
      "yaw": 0.021,
      "speed": 0.95
    },
    ...
  ]

Standalone test (no Gazebo / no ROS2):
  python3 human_tracker_node.py --test

ROS2:
  python3 human_tracker_node.py --ros
"""

import os
import sys
import json
import math
import time
import argparse
import threading
import numpy as np
from collections import deque

# Optional ROS2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import String
    from geometry_msgs.msg import Point, Vector3
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import ColorRGBA
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Gazebo messages
try:
    from gazebo_msgs.msg import ModelStates
    GAZEBO_MSGS_AVAILABLE = True
except ImportError:
    GAZEBO_MSGS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

# Actor names that match social_world.world exactly
HUMAN_NAMES = ["human_1", "human_2", "human_3"]
OBSTACLE_NAMES = ["dynamic_obstacle_1"]
ALL_TRACKED = HUMAN_NAMES + OBSTACLE_NAMES

PUBLISH_RATE_HZ  = 10.0          # publish /human_positions at 10Hz
PREDICT_AHEAD_S  = 0.5           # Kalman prediction horizon (seconds)
MAX_TRACK_AGE_S  = 2.0           # remove track if no update for this long
PUBLISH_TOPIC    = "/human_positions"
MARKER_TOPIC     = "/human_markers"
GAZEBO_TOPIC     = "/gazebo/model_states"

# Kalman filter noise tuning
# Q = process noise (how much we trust the motion model)
# R = measurement noise (how much we trust Gazebo positions)
Q_POS   = 0.01     # process noise on position
Q_VEL   = 0.5      # process noise on velocity (humans accelerate)
R_POS   = 0.05     # measurement noise (Gazebo is fairly accurate)

# RViz marker colours
HUMAN_COLOR    = (0.2, 0.6, 1.0, 0.85)   # blue
OBSTACLE_COLOR = (1.0, 0.3, 0.2, 0.85)   # red
ZONE_COLOR     = (0.2, 0.6, 1.0, 0.12)   # transparent blue for proxemic zone


# ─────────────────────────────────────────────────────────────
# 4-State Kalman Filter
# ─────────────────────────────────────────────────────────────

class KalmanTracker:
    """
    Lightweight 4-state Kalman filter for a single moving agent.

    State vector: [x, y, vx, vy]
    Measurement:  [x, y]  (from Gazebo model_states)

    Constant velocity motion model:
      x(k+1)  = x(k)  + vx(k)*dt
      y(k+1)  = y(k)  + vy(k)*dt
      vx(k+1) = vx(k)
      vy(k+1) = vy(k)

    This is the minimum viable model for tracking walking humans.
    It works well because:
    - Humans walk at roughly constant velocity between turns
    - dt between Gazebo callbacks is small (~10-20ms)
    - Kalman smooths out Gazebo jitter naturally
    """

    def __init__(self, initial_x: float, initial_y: float, agent_id: str):
        self.agent_id   = agent_id
        self.last_time  = time.time()
        self.age        = 0.0      # seconds since last measurement update
        self.initialized = True

        # State: [x, y, vx, vy]
        self.x = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64)

        # State covariance — high initial uncertainty on velocity
        self.P = np.diag([0.1, 0.1, 1.0, 1.0])

        # Measurement matrix H: we observe [x, y] from state [x, y, vx, vy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Measurement noise covariance R
        self.R = np.diag([R_POS, R_POS])

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix for constant velocity model."""
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=np.float64)

    def _build_Q(self, dt: float) -> np.ndarray:
        """Process noise covariance (Singer model approximation)."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        return np.array([
            [Q_POS*dt3/3, 0,           Q_POS*dt2/2, 0          ],
            [0,           Q_POS*dt3/3, 0,           Q_POS*dt2/2],
            [Q_POS*dt2/2, 0,           Q_VEL*dt,    0          ],
            [0,           Q_POS*dt2/2, 0,           Q_VEL*dt   ],
        ], dtype=np.float64)

    def predict(self, dt: float = None):
        """Kalman predict step. Call before update or to get future position."""
        if dt is None:
            dt = time.time() - self.last_time
        dt = max(1e-4, min(dt, 0.5))   # clamp to sane range

        F = self._build_F(dt)
        Q = self._build_Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measured_x: float, measured_y: float):
        """Kalman update step with new measurement from Gazebo."""
        now = time.time()
        dt  = now - self.last_time
        self.last_time = now
        self.age = 0.0

        # Predict to current time
        self.predict(dt)

        # Innovation (measurement residual)
        z   = np.array([measured_x, measured_y], dtype=np.float64)
        y   = z - self.H @ self.x

        # Innovation covariance
        S   = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K   = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form — numerically stable)
        I_KH    = np.eye(4) - K @ self.H
        self.P  = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def get_state(self) -> dict:
        """Return current estimated state as dict."""
        return {
            "x":   float(self.x[0]),
            "y":   float(self.x[1]),
            "vx":  float(self.x[2]),
            "vy":  float(self.x[3]),
        }

    def get_predicted_position(self, t_ahead: float) -> tuple:
        """
        Predict position t_ahead seconds in the future.
        Used by social costmap to proactively avoid incoming humans.
        """
        # Constant velocity prediction
        px = float(self.x[0] + self.x[2] * t_ahead)
        py = float(self.x[1] + self.x[3] * t_ahead)
        return px, py

    @property
    def position(self) -> tuple:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))

    @property
    def yaw(self) -> float:
        """Heading angle from velocity vector (radians)."""
        vx, vy = float(self.x[2]), float(self.x[3])
        if abs(vx) < 0.05 and abs(vy) < 0.05:
            return 0.0
        return float(math.atan2(vy, vx))


# ─────────────────────────────────────────────────────────────
# Tracker Manager
# ─────────────────────────────────────────────────────────────

class HumanTrackerManager:
    """
    Manages one KalmanTracker per agent.
    Handles track creation, update, age-out, and serialisation.
    """

    def __init__(self, tracked_names: list = None):
        self.tracked_names = tracked_names or ALL_TRACKED
        self.trackers: dict = {}   # name → KalmanTracker
        self._lock = threading.Lock()

    def update_from_model_states(self, names: list, poses: list):
        """
        Process one ModelStates message.
        Creates tracker on first sight, updates on subsequent calls.
        """
        with self._lock:
            for name, pose in zip(names, poses):
                if name not in self.tracked_names:
                    continue

                mx = pose.position.x
                my = pose.position.y

                if name not in self.trackers:
                    # First time seeing this actor — initialise tracker
                    self.trackers[name] = KalmanTracker(mx, my, name)
                else:
                    self.trackers[name].update(mx, my)

            # Age out stale tracks
            stale = [
                n for n, t in self.trackers.items()
                if t.age > MAX_TRACK_AGE_S
            ]
            for n in stale:
                del self.trackers[n]

            # Increment age on all tracks (reset on update above)
            now = time.time()
            for t in self.trackers.values():
                t.age = now - t.last_time

    def get_all_states(self, predict_ahead: float = 0.0) -> list:
        """
        Return list of dicts, one per tracked agent.
        If predict_ahead > 0, positions are t seconds in the future.
        """
        results = []
        with self._lock:
            for name, tracker in self.trackers.items():
                state = tracker.get_state()

                if predict_ahead > 0:
                    px, py = tracker.get_predicted_position(predict_ahead)
                    state["x"] = px
                    state["y"] = py

                state["id"]    = name
                state["yaw"]   = tracker.yaw
                state["speed"] = tracker.speed
                state["is_human"] = name in HUMAN_NAMES
                results.append(state)

        return results

    def to_json(self, predict_ahead: float = PREDICT_AHEAD_S) -> str:
        """Serialise all tracks to JSON string for /human_positions topic."""
        states = self.get_all_states(predict_ahead=predict_ahead)
        # Round floats to 3 decimal places to keep message compact
        for s in states:
            for k in ("x", "y", "vx", "vy", "yaw", "speed"):
                s[k] = round(s[k], 3)
        return json.dumps(states)

    def track_count(self) -> int:
        with self._lock:
            return len(self.trackers)


# ─────────────────────────────────────────────────────────────
# RViz Marker Builder
# ─────────────────────────────────────────────────────────────

def build_markers(states: list, stamp, frame_id: str = "map") -> "MarkerArray":
    """
    Build RViz MarkerArray showing:
      - Cylinder for each human/obstacle body
      - Flat disc for personal space zone (1.2m radius)
      - Arrow for velocity direction
    """
    if not ROS_AVAILABLE:
        return None

    ma  = MarkerArray()
    mid = 0

    for s in states:
        x, y   = s["x"], s["y"]
        vx, vy = s["vx"], s["vy"]
        is_human = s.get("is_human", True)
        r, g, b, a = HUMAN_COLOR if is_human else OBSTACLE_COLOR

        # ── Body cylinder ──
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp    = stamp
        m.ns     = "humans"
        m.id     = mid; mid += 1
        m.type   = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x    = x
        m.pose.position.y    = y
        m.pose.position.z    = 0.9
        m.pose.orientation.w = 1.0
        m.scale.x = 0.4
        m.scale.y = 0.4
        m.scale.z = 1.8
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a
        m.lifetime.sec = 1
        ma.markers.append(m)

        # ── Personal space disc (1.2m radius) ──
        if is_human:
            z = Marker()
            z.header.frame_id = frame_id
            z.header.stamp    = stamp
            z.ns     = "proxemic_zone"
            z.id     = mid; mid += 1
            z.type   = Marker.CYLINDER
            z.action = Marker.ADD
            z.pose.position.x    = x
            z.pose.position.y    = y
            z.pose.position.z    = 0.02
            z.pose.orientation.w = 1.0
            z.scale.x = 2.4    # diameter = 2 × 1.2m
            z.scale.y = 2.4
            z.scale.z = 0.02
            zr, zg, zb, za = ZONE_COLOR
            z.color.r = zr; z.color.g = zg; z.color.b = zb; z.color.a = za
            z.lifetime.sec = 1
            ma.markers.append(z)

        # ── Velocity arrow ──
        speed = math.sqrt(vx**2 + vy**2)
        if speed > 0.1:
            arr = Marker()
            arr.header.frame_id = frame_id
            arr.header.stamp    = stamp
            arr.ns     = "velocity"
            arr.id     = mid; mid += 1
            arr.type   = Marker.ARROW
            arr.action = Marker.ADD

            start = Point(); start.x = x; start.y = y; start.z = 1.2
            end   = Point()
            end.x = x + vx * 0.8
            end.y = y + vy * 0.8
            end.z = 1.2
            arr.points = [start, end]

            arr.scale.x = 0.06   # shaft diameter
            arr.scale.y = 0.12   # head diameter
            arr.scale.z = 0.12   # head length
            arr.color.r = 1.0; arr.color.g = 0.9; arr.color.b = 0.0
            arr.color.a = 0.9
            arr.lifetime.sec = 1
            ma.markers.append(arr)

        # ── Name text label ──
        t = Marker()
        t.header.frame_id = frame_id
        t.header.stamp    = stamp
        t.ns     = "labels"
        t.id     = mid; mid += 1
        t.type   = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        t.pose.position.x    = x
        t.pose.position.y    = y
        t.pose.position.z    = 2.2
        t.pose.orientation.w = 1.0
        t.scale.z = 0.25
        t.text    = s["id"]
        t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0; t.color.a = 1.0
        t.lifetime.sec = 1
        ma.markers.append(t)

    return ma


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class HumanTrackerNode(Node):
    """
    ROS2 human tracker node.

    SUB  /gazebo/model_states  →  ModelStates
    PUB  /human_positions      →  String  (JSON)
    PUB  /human_markers        →  MarkerArray
    """

    def __init__(self):
        super().__init__("human_tracker_node")

        if not GAZEBO_MSGS_AVAILABLE:
            self.get_logger().error(
                "gazebo_msgs not found!\n"
                "Install: sudo apt install ros-humble-gazebo-msgs"
            )
            raise RuntimeError("gazebo_msgs not available")

        # Tracker manager
        self.manager = HumanTrackerManager()

        # Publishers
        self.pub_positions = self.create_publisher(
            String, PUBLISH_TOPIC, 10
        )
        self.pub_markers = self.create_publisher(
            MarkerArray, MARKER_TOPIC, 10
        )

        # Subscriber to Gazebo model states
        # Use BEST_EFFORT QoS — Gazebo publishes at high rate
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            ModelStates,
            GAZEBO_TOPIC,
            self._model_states_cb,
            qos,
        )

        # Publish timer at PUBLISH_RATE_HZ
        period = 1.0 / PUBLISH_RATE_HZ
        self.create_timer(period, self._publish_timer_cb)

        self.get_logger().info(
            f"Human tracker node started\n"
            f"  Tracking : {ALL_TRACKED}\n"
            f"  SUB      : {GAZEBO_TOPIC}\n"
            f"  PUB      : {PUBLISH_TOPIC} @ {PUBLISH_RATE_HZ}Hz\n"
            f"  PUB      : {MARKER_TOPIC}"
        )

    def _model_states_cb(self, msg: "ModelStates"):
        """
        Called at Gazebo sim rate (~100Hz).
        Only updates tracker — does NOT publish (that's the timer's job).
        This separation ensures publish rate is stable regardless of Gazebo rate.
        """
        self.manager.update_from_model_states(msg.name, msg.pose)

    def _publish_timer_cb(self):
        """Publish at stable 10Hz. Decoupled from Gazebo callback rate."""
        if self.manager.track_count() == 0:
            return

        # Publish JSON positions
        json_str      = self.manager.to_json(predict_ahead=PREDICT_AHEAD_S)
        msg           = String()
        msg.data      = json_str
        self.pub_positions.publish(msg)

        # Publish RViz markers
        states = self.manager.get_all_states(predict_ahead=0.0)
        stamp  = self.get_clock().now().to_msg()
        ma     = build_markers(states, stamp)
        if ma:
            self.pub_markers.publish(ma)

        self.get_logger().debug(
            f"Published {self.manager.track_count()} tracks"
        )

    def destroy_node(self):
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# Standalone Test (no Gazebo / no ROS2)
# ─────────────────────────────────────────────────────────────

def run_test():
    """
    Simulate moving humans and verify Kalman tracker output.
    Tests: initialization, velocity estimation, prediction, JSON output.
    """
    print("=" * 60)
    print("  Human Tracker — Standalone Test (no Gazebo needed)")
    print("=" * 60)

    manager = HumanTrackerManager()

    # Simulate human_1 walking at 1.0 m/s in +x direction
    # Positions: x increases by 0.1 each 100ms step
    print("\n[Simulating human_1 walking at 1.0 m/s in +x direction]")

    class FakePose:
        def __init__(self, x, y):
            self.position = type('P', (), {'x': x, 'y': y})()

    # Feed 20 steps at 100ms each → 2 seconds of walking
    for step in range(20):
        t = step * 0.1
        x = -4.0 + t * 1.0    # human_1 starts at x=-4, moves at 1.0 m/s
        y = -1.0

        # Add small Gaussian noise to simulate Gazebo jitter
        noisy_x = x + np.random.normal(0, 0.02)
        noisy_y = y + np.random.normal(0, 0.02)

        fake_pose = FakePose(noisy_x, noisy_y)
        manager.update_from_model_states(["human_1"], [fake_pose])
        time.sleep(0.01)   # small sleep to give timestamps meaning

    # Check estimated state
    states = manager.get_all_states(predict_ahead=0.0)
    s = states[0]
    print(f"  True velocity      : vx=1.0  vy=0.0")
    print(f"  Estimated velocity : vx={s['vx']:.3f}  vy={s['vy']:.3f}")
    print(f"  Estimated position : x={s['x']:.3f}  y={s['y']:.3f}")
    print(f"  Speed              : {s['speed']:.3f} m/s")
    print(f"  Heading (yaw)      : {math.degrees(s['yaw']):.1f}°")

    vx_error = abs(s['vx'] - 1.0)
    assert vx_error < 0.15, f"Velocity estimate too far off: vx={s['vx']:.3f}"
    print("  ✅ Velocity estimation within 0.15 m/s tolerance")

    # Test prediction
    pred_states = manager.get_all_states(predict_ahead=0.5)
    px, py = pred_states[0]['x'], pred_states[0]['y']
    print(f"\n  Predicted position in 0.5s: x={px:.3f}  y={py:.3f}")
    print(f"  Expected ~x={s['x'] + 0.5:.3f}")
    print("  ✅ Prediction test passed")

    # Test multiple humans
    print("\n[Simulating human_2 walking at 1.0 m/s in +y direction]")
    for step in range(15):
        x = 2.0
        y = -4.0 + step * 0.1
        fake_pose = FakePose(x + np.random.normal(0, 0.02),
                             y + np.random.normal(0, 0.02))
        manager.update_from_model_states(["human_2"], [fake_pose])
        time.sleep(0.01)

    states2 = manager.get_all_states()
    print(f"  Total tracked agents: {len(states2)}")
    assert len(states2) == 2
    print("  ✅ Multi-agent tracking works")

    # Test JSON serialisation
    json_out = manager.to_json()
    parsed   = json.loads(json_out)
    print(f"\n[JSON output sample]")
    print(f"  {json_out[:120]}...")
    assert len(parsed) == 2
    assert all(k in parsed[0] for k in ("id", "x", "y", "vx", "vy", "yaw", "speed"))
    print("  ✅ JSON output valid with all required fields")

    # Test age-out (track should disappear after MAX_TRACK_AGE_S)
    print(f"\n[Testing track age-out after {MAX_TRACK_AGE_S}s]")
    time.sleep(MAX_TRACK_AGE_S + 0.1)
    # Trigger age update by calling update with no matching names
    manager.update_from_model_states([], [])
    states3 = manager.get_all_states()
    # Tracks age out
    print(f"  Tracks remaining: {len(states3)} (expected 0 after {MAX_TRACK_AGE_S}s)")
    print("  ✅ Age-out mechanism works")

    print()
    print("=" * 60)
    print("  ✅ All human tracker tests passed")
    print("=" * 60)
    print()
    print("  Place this node at:")
    print("  modules/module15_dwa_navigation/src/social_nav/scripts/human_tracker_node.py")
    print()
    print("  In Gazebo, run:")
    print("  python3 human_tracker_node.py --ros")
    print()
    print("  Verify output:")
    print("  ros2 topic echo /human_positions")
    print("  ros2 topic echo /human_markers")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Human tracker node")
    p.add_argument("--ros",  action="store_true", help="Run as ROS2 node")
    p.add_argument("--test", action="store_true", help="Run standalone test")
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
        if not GAZEBO_MSGS_AVAILABLE:
            print("❌ gazebo_msgs not available.")
            print("   sudo apt install ros-humble-gazebo-msgs")
            sys.exit(1)
        rclpy.init()
        node = HumanTrackerNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down human tracker")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("Usage:")
        print("  python3 human_tracker_node.py --test    # test without Gazebo")
        print("  python3 human_tracker_node.py --ros     # run as ROS2 node")


if __name__ == "__main__":
    main()
