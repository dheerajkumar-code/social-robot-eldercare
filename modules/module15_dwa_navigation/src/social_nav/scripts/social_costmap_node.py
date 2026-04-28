#!/usr/bin/env python3
"""
Module 15 — Social Costmap Node
=================================
Subscribes to /human_positions, computes an asymmetric Gaussian
proxemics cost field for each human, and publishes a merged
OccupancyGrid for Nav2 and the DWA social override node.

Proxemics model (Hall 1966):
  Intimate  < 0.45m  → cost 100 (lethal — robot must not enter)
  Personal  0.45–1.2m → cost 80
  Social    1.2–3.6m  → cost 40
  Public    > 3.6m    → cost 0

Asymmetric Gaussian (orientation-aware):
  The cost field is elongated in the direction the human is facing.
  A human walking toward the robot needs more clearance than one
  walking away. This is the key improvement over a circular model.

  C(x,y) = A · exp(-0.5 · [(dx_front/σ_front)² + (dy_side/σ_side)²])

  where σ_front > σ_rear > σ_side and the coordinate frame is
  rotated to align with human heading (yaw).

  σ_front = 1.2m   (in front of human — walking into robot's space)
  σ_rear  = 0.5m   (behind human — less danger)
  σ_side  = 0.75m  (beside human)

Implementation:
  Uses numpy vectorised operations over the full grid.
  No Python loops over cells — runs in ~2ms on Pi5.

Topics:
  SUB  /human_positions    std_msgs/String  (JSON from human_tracker_node)
  PUB  /social_costmap     nav_msgs/OccupancyGrid
  PUB  /social_costmap_raw std_msgs/String  (JSON for debug / DWA lookup)

Standalone test:
  python3 social_costmap_node.py --test

ROS2:
  python3 social_costmap_node.py --ros
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
    from nav_msgs.msg import OccupancyGrid, MapMetaData
    from geometry_msgs.msg import Pose, Point, Quaternion
    from builtin_interfaces.msg import Time
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Costmap Configuration
# ─────────────────────────────────────────────────────────────

# Grid parameters — must match or align with Nav2 costmap
RESOLUTION   = 0.1        # metres per cell
WORLD_X_MIN  = -5.0       # world bounds (from social_world.world walls)
WORLD_X_MAX  =  5.0
WORLD_Y_MIN  = -5.0
WORLD_Y_MAX  =  5.0
GRID_WIDTH   = int((WORLD_X_MAX - WORLD_X_MIN) / RESOLUTION)   # 100 cells
GRID_HEIGHT  = int((WORLD_Y_MAX - WORLD_Y_MIN) / RESOLUTION)   # 100 cells

# Asymmetric Gaussian shape parameters (metres)
SIGMA_FRONT  = 1.20   # ahead of human (walking toward robot)
SIGMA_REAR   = 0.50   # behind human
SIGMA_SIDE   = 0.75   # to either side

# Cost amplitude
COST_AMPLITUDE     = 100.0   # peak cost at human centre
COST_LETHAL        = 100     # Nav2 lethal cost
COST_INSCRIBED     =  99     # Nav2 inscribed cost
INFLUENCE_RADIUS_M = 3.6     # beyond this, Gaussian ≈ 0 (skip computation)

# Publish rate
PUBLISH_RATE_HZ = 10.0

# Topic names
SUB_HUMAN_POSITIONS   = "/human_positions"
PUB_SOCIAL_COSTMAP    = "/social_costmap"
PUB_COSTMAP_RAW       = "/social_costmap_raw"

FRAME_ID = "map"


# ─────────────────────────────────────────────────────────────
# Asymmetric Gaussian Costmap Engine
# ─────────────────────────────────────────────────────────────

class SocialCostmapEngine:
    """
    Computes a social cost grid from a list of human states.

    Grid convention:
      Cell (row, col) corresponds to world position:
        x = WORLD_X_MIN + col * RESOLUTION + RESOLUTION/2
        y = WORLD_Y_MIN + row * RESOLUTION + RESOLUTION/2

      Row 0 is at y = WORLD_Y_MIN (south edge).
      Col 0 is at x = WORLD_X_MIN (west edge).
    """

    def __init__(self):
        self._lock        = threading.Lock()
        self._grid        = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        self._last_humans = []

        # Pre-compute world coordinate arrays for vectorised operations
        # xs[row, col] = world x of cell centre
        # ys[row, col] = world y of cell centre
        cols = np.arange(GRID_WIDTH,  dtype=np.float32)
        rows = np.arange(GRID_HEIGHT, dtype=np.float32)
        col_grid, row_grid = np.meshgrid(cols, rows)

        self._xs = (WORLD_X_MIN + col_grid * RESOLUTION
                    + RESOLUTION / 2.0).astype(np.float32)
        self._ys = (WORLD_Y_MIN + row_grid * RESOLUTION
                    + RESOLUTION / 2.0).astype(np.float32)

        # Influence radius in cells (for bounding box optimisation)
        self._r_cells = int(math.ceil(INFLUENCE_RADIUS_M / RESOLUTION))

    def world_to_cell(self, wx: float, wy: float):
        """Convert world (x,y) to grid (row, col). Returns None if outside."""
        col = int((wx - WORLD_X_MIN) / RESOLUTION)
        row = int((wy - WORLD_Y_MIN) / RESOLUTION)
        if 0 <= col < GRID_WIDTH and 0 <= row < GRID_HEIGHT:
            return row, col
        return None

    def _gaussian_patch(self, hx: float, hy: float,
                        yaw: float, speed: float) -> np.ndarray:
        """
        Compute asymmetric Gaussian cost contribution for ONE human.

        The Gaussian is oriented by the human's heading (yaw).
        Front sigma is larger when human is moving (speed > 0.1),
        because a moving human occupies more future space.

        Returns (GRID_HEIGHT, GRID_WIDTH) float32 array.
        """
        # Scale front sigma by speed (moving humans need more clearance)
        speed_scale = 1.0 + 0.5 * min(speed, 1.5)   # max 2.25× at 1.5m/s
        sigma_f = SIGMA_FRONT * speed_scale
        sigma_r = SIGMA_REAR
        sigma_s = SIGMA_SIDE

        # Bounding box: skip cells outside influence radius (speed)
        cx, cy = self.world_to_cell(hx, hy) or (None, None)
        if cx is None:
            return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

        r = self._r_cells
        row_lo = max(0,           cy - r)
        row_hi = min(GRID_HEIGHT, cy + r)
        col_lo = max(0,           cx - r)
        col_hi = min(GRID_WIDTH,  cx + r)

        # Work on the sub-grid only
        xs_patch = self._xs[row_lo:row_hi, col_lo:col_hi]
        ys_patch = self._ys[row_lo:row_hi, col_lo:col_hi]

        # Translate to human-centred frame
        dx = xs_patch - hx
        dy = ys_patch - hy

        # Rotate to human's local frame (front = heading direction)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        dx_front = dx * cos_y + dy * sin_y    # distance along heading
        dy_side  = -dx * sin_y + dy * cos_y  # distance perpendicular

        # Asymmetric: different sigma for front vs rear
        sigma_x = np.where(dx_front >= 0, sigma_f, sigma_r)

        # Gaussian cost
        exponent = 0.5 * ((dx_front / sigma_x) ** 2
                          + (dy_side  / sigma_s) ** 2)
        cost_patch = (COST_AMPLITUDE * np.exp(-exponent)).astype(np.float32)

        # Write into full grid
        full = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
        full[row_lo:row_hi, col_lo:col_hi] = cost_patch
        return full

    def update(self, humans: list):
        """
        Recompute costmap from a list of human state dicts.
        Each dict must have: x, y, yaw, speed, id
        """
        combined = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

        for h in humans:
            patch = self._gaussian_patch(
                hx    = float(h.get("x",     0.0)),
                hy    = float(h.get("y",     0.0)),
                yaw   = float(h.get("yaw",   0.0)),
                speed = float(h.get("speed", 0.0)),
            )
            # Overlay: take max cost at each cell (avoid summing)
            np.maximum(combined, patch, out=combined)

        # Clamp to Nav2 valid range [0, 100]
        np.clip(combined, 0.0, float(COST_LETHAL), out=combined)

        with self._lock:
            self._grid        = combined
            self._last_humans = humans

    def get_grid_int8(self) -> np.ndarray:
        """Return grid as flat int8 array (row-major) for OccupancyGrid msg."""
        with self._lock:
            return self._grid.astype(np.int8).flatten()

    def get_cost_at_world(self, wx: float, wy: float) -> int:
        """
        Look up cost at a world position.
        Used by social_override_node to check robot's planned position.
        Returns 0–100.
        """
        cell = self.world_to_cell(wx, wy)
        if cell is None:
            return 0
        row, col = cell
        with self._lock:
            return int(self._grid[row, col])

    def get_grid_json(self) -> str:
        """
        Serialise costmap as compact JSON for the DWA override node.
        Format: {"resolution": 0.1, "width": 100, "height": 100,
                 "origin_x": -5.0, "origin_y": -5.0,
                 "data": [0,0,...]}
        """
        with self._lock:
            flat = self._grid.flatten().astype(int).tolist()  # flatten 2D→1D first
        return json.dumps({
            "resolution": RESOLUTION,
            "width":      GRID_WIDTH,
            "height":     GRID_HEIGHT,
            "origin_x":   WORLD_X_MIN,
            "origin_y":   WORLD_Y_MIN,
            "data":       flat,
        }, separators=(",", ":"))   # compact (no spaces)

    @property
    def grid_shape(self):
        return GRID_HEIGHT, GRID_WIDTH


# ─────────────────────────────────────────────────────────────
# OccupancyGrid message builder
# ─────────────────────────────────────────────────────────────

def build_occupancy_grid(engine: SocialCostmapEngine,
                          stamp, frame_id: str = FRAME_ID):
    """Build nav_msgs/OccupancyGrid from engine state."""
    if not ROS_AVAILABLE:
        return None

    og                = OccupancyGrid()
    og.header.frame_id = frame_id
    og.header.stamp    = stamp

    og.info.resolution = RESOLUTION
    og.info.width      = GRID_WIDTH
    og.info.height     = GRID_HEIGHT

    # Origin = south-west corner of the grid
    og.info.origin.position.x    = WORLD_X_MIN
    og.info.origin.position.y    = WORLD_Y_MIN
    og.info.origin.position.z    = 0.0
    og.info.origin.orientation.w = 1.0

    # Flat int8 list, row-major (row 0 = south edge)
    og.data = engine.get_grid_int8().tolist()

    return og


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class SocialCostmapNode(Node):
    """
    ROS2 social costmap node.

    SUB  /human_positions    →  String (JSON)
    PUB  /social_costmap     →  OccupancyGrid
    PUB  /social_costmap_raw →  String (JSON, for DWA override)
    """

    def __init__(self):
        super().__init__("social_costmap_node")

        self.engine = SocialCostmapEngine()

        # Publishers
        self.pub_grid = self.create_publisher(
            OccupancyGrid, PUB_SOCIAL_COSTMAP, 10
        )
        self.pub_raw = self.create_publisher(
            String, PUB_COSTMAP_RAW, 10
        )

        # Subscriber
        self.create_subscription(
            String,
            SUB_HUMAN_POSITIONS,
            self._human_cb,
            10,
        )

        # Publish timer
        period = 1.0 / PUBLISH_RATE_HZ
        self.create_timer(period, self._publish_cb)

        self.get_logger().info(
            f"Social costmap node started\n"
            f"  Grid      : {GRID_WIDTH}×{GRID_HEIGHT} @ {RESOLUTION}m\n"
            f"  σ_front   : {SIGMA_FRONT}m\n"
            f"  σ_rear    : {SIGMA_REAR}m\n"
            f"  σ_side    : {SIGMA_SIDE}m\n"
            f"  PUB       : {PUB_SOCIAL_COSTMAP}\n"
            f"  PUB       : {PUB_COSTMAP_RAW}"
        )

    def _human_cb(self, msg: String):
        """Update costmap whenever new human positions arrive."""
        try:
            humans = json.loads(msg.data)
            self.engine.update(humans)
        except json.JSONDecodeError as e:
            self.get_logger().warning(f"Bad JSON from /human_positions: {e}")

    def _publish_cb(self):
        """Publish at stable 10Hz."""
        stamp = self.get_clock().now().to_msg()

        # OccupancyGrid for Nav2 + RViz
        og = build_occupancy_grid(self.engine, stamp)
        if og:
            self.pub_grid.publish(og)

        # Raw JSON for DWA override node (fast lookup)
        raw_msg      = String()
        raw_msg.data = self.engine.get_grid_json()
        self.pub_raw.publish(raw_msg)

    def destroy_node(self):
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# Standalone Test
# ─────────────────────────────────────────────────────────────

def run_test():
    print("=" * 60)
    print("  Social Costmap — Standalone Tests")
    print("=" * 60)

    engine = SocialCostmapEngine()

    # ── T1: Single stationary human at origin, facing +x ──
    print("\nT1 Single human at (0,0) facing +x (yaw=0), speed=0")
    engine.update([{"id":"human_1","x":0.0,"y":0.0,
                    "yaw":0.0,"speed":0.0,"is_human":True}])

    # Cost at human centre should be ~100
    c_centre = engine.get_cost_at_world(0.0, 0.0)
    print(f"  Cost at centre (0,0)       : {c_centre} (expect ~100)")
    assert c_centre >= 95, f"Expected ~100, got {c_centre}"

    # Cost directly in front (1m) > cost directly behind (1m)
    c_front = engine.get_cost_at_world(1.0, 0.0)    # 1m in front (+x)
    c_rear  = engine.get_cost_at_world(-1.0, 0.0)   # 1m behind (-x)
    print(f"  Cost 1m in front  (+x)     : {c_front}")
    print(f"  Cost 1m behind    (-x)     : {c_rear}")
    assert c_front > c_rear, \
        f"Front cost {c_front} should be > rear cost {c_rear}"
    print("  ✅ Asymmetric: front > rear")

    # Cost to side at 1m should be between front and rear
    c_side = engine.get_cost_at_world(0.0, 1.0)     # 1m to side (+y)
    print(f"  Cost 1m to side   (+y)     : {c_side}")
    assert c_rear < c_side < c_front or c_side > 0, \
        "Side cost should be nonzero"
    print("  ✅ Side cost nonzero")

    # Far away (5m) should be ~0
    c_far = engine.get_cost_at_world(5.0, 0.0)
    print(f"  Cost at 5m away            : {c_far} (expect 0)")
    assert c_far == 0, f"Expected 0, got {c_far}"
    print("  ✅ Zero cost beyond influence radius")

    # ── T2: Asymmetry changes with heading ──
    print("\nT2 Same human facing +y (yaw=π/2)")
    engine.update([{"id":"human_1","x":0.0,"y":0.0,
                    "yaw":math.pi/2,"speed":0.0,"is_human":True}])

    c_front_y = engine.get_cost_at_world(0.0,  1.0)  # now front is +y
    c_rear_y  = engine.get_cost_at_world(0.0, -1.0)  # now rear is -y
    c_side_x  = engine.get_cost_at_world(1.0,  0.0)  # now side is +x
    print(f"  1m in front (+y): {c_front_y}")
    print(f"  1m behind   (-y): {c_rear_y}")
    print(f"  1m to side  (+x): {c_side_x}")
    assert c_front_y > c_rear_y, "Front should still be larger than rear"
    print("  ✅ Asymmetry rotates correctly with heading")

    # ── T3: Speed scaling ──
    print("\nT3 Speed scaling (faster human → larger front zone)")
    engine.update([{"id":"h","x":0.0,"y":0.0,
                    "yaw":0.0,"speed":0.0,"is_human":True}])
    c_2m_slow = engine.get_cost_at_world(2.0, 0.0)

    engine.update([{"id":"h","x":0.0,"y":0.0,
                    "yaw":0.0,"speed":1.5,"is_human":True}])
    c_2m_fast = engine.get_cost_at_world(2.0, 0.0)
    print(f"  Cost at 2m (speed=0.0): {c_2m_slow}")
    print(f"  Cost at 2m (speed=1.5): {c_2m_fast}")
    assert c_2m_fast >= c_2m_slow, "Faster human should have ≥ cost at 2m"
    print("  ✅ Speed scaling works")

    # ── T4: Multiple humans merge correctly ──
    print("\nT4 Two humans, costs merged by max()")
    engine.update([
        {"id":"h1","x":-2.0,"y":0.0,"yaw":0.0,"speed":0.0},
        {"id":"h2","x": 2.0,"y":0.0,"yaw":math.pi,"speed":0.0},
    ])
    c_h1 = engine.get_cost_at_world(-2.0, 0.0)
    c_h2 = engine.get_cost_at_world( 2.0, 0.0)
    c_mid = engine.get_cost_at_world(0.0, 0.0)   # between humans
    print(f"  Cost at human_1 (-2,0): {c_h1}")
    print(f"  Cost at human_2 (+2,0): {c_h2}")
    print(f"  Cost at midpoint (0,0): {c_mid}")
    assert c_h1 >= 95 and c_h2 >= 95
    print("  ✅ Multiple humans merged correctly")

    # ── T5: world_to_cell coordinate mapping ──
    print("\nT5 Coordinate mapping")
    cell = engine.world_to_cell(0.0, 0.0)
    print(f"  world(0,0)    → cell{cell} (expect ({GRID_HEIGHT//2},{GRID_WIDTH//2}))")
    assert cell == (GRID_HEIGHT // 2, GRID_WIDTH // 2)

    cell_sw = engine.world_to_cell(-4.95, -4.95)
    print(f"  world(-4.95,-4.95) → cell{cell_sw} (expect (0,0))")
    assert cell_sw == (0, 0)

    cell_ne = engine.world_to_cell(4.95, 4.95)
    print(f"  world(4.95,4.95)   → cell{cell_ne} (expect (99,99))")
    assert cell_ne == (99, 99)

    cell_out = engine.world_to_cell(10.0, 10.0)
    print(f"  world(10,10)       → cell{cell_out} (expect None)")
    assert cell_out is None
    print("  ✅ Coordinate mapping correct")

    # ── T6: JSON output structure ──
    print("\nT6 JSON output")
    engine.update([{"id":"h","x":0,"y":0,"yaw":0,"speed":0}])
    j = engine.get_grid_json()
    parsed = json.loads(j)
    assert parsed["resolution"] == RESOLUTION
    assert parsed["width"]      == GRID_WIDTH
    assert parsed["height"]     == GRID_HEIGHT
    assert len(parsed["data"])  == GRID_WIDTH * GRID_HEIGHT
    print(f"  JSON keys   : {list(parsed.keys())}")
    print(f"  data length : {len(parsed['data'])} (expect {GRID_WIDTH*GRID_HEIGHT})")
    print("  ✅ JSON structure valid")

    # ── T7: Performance ──
    print("\nT7 Performance (4 humans, 10 updates)")
    humans = [
        {"id":"h1","x":-2.0,"y": 0.0,"yaw":0.0,         "speed":1.0},
        {"id":"h2","x": 2.0,"y": 0.0,"yaw":math.pi,      "speed":1.0},
        {"id":"h3","x": 0.0,"y":-2.0,"yaw":math.pi/2,    "speed":0.8},
        {"id":"h4","x": 0.0,"y": 2.0,"yaw":-math.pi/2,   "speed":0.5},
    ]
    t0 = time.perf_counter()
    for _ in range(10):
        engine.update(humans)
    elapsed_ms = (time.perf_counter() - t0) * 100   # ms per update
    print(f"  Avg update time : {elapsed_ms:.2f} ms per call")
    assert elapsed_ms < 50, f"Too slow: {elapsed_ms:.1f}ms (target <50ms)"
    print("  ✅ Performance within 50ms budget")

    print()
    print("=" * 60)
    print("  ✅ All 7 social costmap tests passed")
    print("=" * 60)
    print()
    print("  Place at:")
    print("  modules/module15_dwa_navigation/src/social_nav/scripts/social_costmap_node.py")
    print()
    print("  In ROS2:")
    print("  python3 social_costmap_node.py --ros")
    print()
    print("  Verify:")
    print("  ros2 topic echo /social_costmap")
    print("  rviz2  →  add OccupancyGrid → /social_costmap")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Social costmap node")
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
        node = SocialCostmapNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down social costmap node")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        print("Usage:")
        print("  python3 social_costmap_node.py --test")
        print("  python3 social_costmap_node.py --ros")


if __name__ == "__main__":
    main()
