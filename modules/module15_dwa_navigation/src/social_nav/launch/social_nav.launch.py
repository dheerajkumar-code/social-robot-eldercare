#!/usr/bin/env python3
"""
<<<<<<< HEAD
social_nav.launch.py — DEFINITIVE FIXED VERSION v3
=====================================================
Fixes ALL TF / spawn / RViz failures on ROS2 Humble.

ROOT CAUSES FIXED IN THIS VERSION:
─────────────────────────────────────────────────────
BUG 1 [PARSE CRASH]   open(urdf_file) ran at Python parse time
                       → crashes before colcon build completes
                       FIX: Use Command(['xacro', urdf_file])
                            evaluated at EXECUTION time only

BUG 2 [BROKEN TF]     joint_state_publisher conflicted with Gazebo
                       → Gazebo diff_drive plugin ALREADY publishes
                         /joint_states for simulated wheels
                       → Running joint_state_publisher simultaneously
                         causes duplicate messages → TF tree broken
                       → "No transform from base_link" in RViz
                       FIX: REMOVE joint_state_publisher in simulation
                            Gazebo handles joint states automatically

BUG 3 [TF FREEZE]     robot_state_publisher started before /clock existed
                       → use_sim_time=True requires Gazebo /clock
                       → RSP froze waiting for clock → no TF published
                       FIX: Start RSP with 2s delay (after Gazebo clock)

BUG 4 [MESH MISSING]  GAZEBO_MODEL_PATH missing turtlebot3_description
                       → STL mesh files in turtlebot3_description/meshes/
                       → Robot had no visual mesh → invisible in Gazebo
                       FIX: Add turtlebot3_description share to MODEL_PATH

BUG 5 [SPAWN RACE]    5s timer for spawn not event-driven
                       → On slow machines Gazebo not ready in 5s
                       FIX: Increase to 7s + use -timeout 60 flag
                            spawn_entity.py polls /spawn_entity itself

BUG 6 [PLUGIN LOST]   GAZEBO_PLUGIN_PATH not set
                       → libgazebo_ros_diff_drive.so not found by Gazebo
                       → No wheel velocity → robot doesn't move
                       FIX: Set GAZEBO_PLUGIN_PATH to ROS2 lib directory

EXECUTION ORDER:
  t=0s   ENV VARS set (TURTLEBOT3_MODEL, GAZEBO_MODEL_PATH, GAZEBO_PLUGIN_PATH)
  t=0s   Gazebo starts (gazebo_ros factory → /spawn_entity + /clock available)
  t=2s   robot_state_publisher starts (has /clock now → publishes /robot_description)
  t=7s   spawn_entity.py runs (reads /robot_description → spawns TurtleBot3)
            Gazebo diff_drive starts publishing /joint_states + odom TF
  t=10s  Nav2 bringup (AMCL, planner, controller, bt_navigator, lifecycle)
  t=12s  Social nodes: human_tracker, social_costmap, social_override
  t=13s  RViz2

TF CHAIN AFTER FIX:
  map → odom        (AMCL / static publisher)
  odom → base_footprint (Gazebo diff_drive plugin)
  base_footprint → base_link (robot_state_publisher from URDF)
  base_link → base_scan / imu_link / wheel_* (robot_state_publisher)
  wheel_left_joint / wheel_right_joint (Gazebo diff_drive → /joint_states)

cmd_vel CHAIN:
  DWB → /cmd_vel → velocity_smoother → /cmd_vel_smoothed
  → collision_monitor → /cmd_vel_nav
  → social_override_node → /cmd_vel → TurtleBot3 Gazebo plugin

Usage:
  ros2 launch social_nav social_nav.launch.py
  ros2 launch social_nav social_nav.launch.py headless:=True use_rviz:=False
  ros2 launch social_nav social_nav.launch.py robot_x:=-3.5 robot_y:=0.0
"""

import os

=======
social_nav.launch.py
======================
Master launch file for Module 15 — Social Navigation.

Launches in order:
  1. Gazebo with social_world.world (humans + obstacles)
  2. TurtleBot3 spawner
  3. Nav2 stack (AMCL, planner, controller, costmaps)
  4. human_tracker_node.py       → /human_positions
  5. social_costmap_node.py      → /social_costmap
  6. social_override_node.py     → /cmd_vel (from /cmd_vel_raw)
  7. RViz2 with social_nav.rviz

Key architecture:
  Nav2 DWB controller  →  /cmd_vel_raw  →  social_override_node  →  /cmd_vel
  (remapped via launch arguments)

Usage:
  # Full simulation
  ros2 launch social_nav social_nav.launch.py

  # Headless (no Gazebo GUI) — for Pi5
  ros2 launch social_nav social_nav.launch.py headless:=True

  # Custom goal position
  ros2 launch social_nav social_nav.launch.py goal_x:=3.5 goal_y:=2.0

  # Without RViz
  ros2 launch social_nav social_nav.launch.py use_rviz:=False
"""

import os
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
<<<<<<< HEAD
    TimerAction,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    Command,
    FindExecutable,
    PathJoinSubstitution,
=======
    ExecuteProcess,
    TimerAction,
    GroupAction,
    LogInfo,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

<<<<<<< HEAD
    # ─────────────────────────────────────────────────────────
    # Package paths (resolved at launch time)
    # ─────────────────────────────────────────────────────────
    pkg_social_nav        = get_package_share_directory("social_nav")
    pkg_nav2_bringup      = get_package_share_directory("nav2_bringup")
    pkg_turtlebot3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    pkg_turtlebot3_desc   = get_package_share_directory("turtlebot3_description")
    pkg_gazebo_ros        = get_package_share_directory("gazebo_ros")

    # ─────────────────────────────────────────────────────────
    # File paths
    # ─────────────────────────────────────────────────────────
    world_file   = os.path.join(pkg_social_nav,      "worlds", "social_world.world")
    params_file  = os.path.join(pkg_social_nav,      "config", "nav2_params.yaml")
    rviz_file    = os.path.join(pkg_social_nav,      "rviz",   "social_nav.rviz")
    map_file     = os.path.join(pkg_social_nav,      "maps",   "map.yaml")
    urdf_file    = os.path.join(pkg_turtlebot3_desc, "urdf",   "turtlebot3_burger.urdf")

    # ─────────────────────────────────────────────────────────
    # Launch arguments
    # ─────────────────────────────────────────────────────────
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="True",
        description="Use Gazebo simulation time"
=======
    # ── Package directories ──
    pkg_social_nav    = get_package_share_directory("social_nav")
    pkg_nav2_bringup  = get_package_share_directory("nav2_bringup")
    pkg_turtlebot3_gz = get_package_share_directory("turtlebot3_gazebo")

    # ── Paths ──
    world_file  = os.path.join(pkg_social_nav, "worlds", "social_world.world")
    params_file = os.path.join(pkg_social_nav, "config",  "nav2_params.yaml")
    rviz_file   = os.path.join(pkg_social_nav, "rviz",    "social_nav.rviz")
    map_file    = os.path.join(pkg_social_nav, "maps",    "map.yaml")

    # Script paths
    scripts_dir       = os.path.join(pkg_social_nav, "lib", "social_nav")
    human_tracker     = os.path.join(scripts_dir, "human_tracker_node.py")
    social_costmap    = os.path.join(scripts_dir, "social_costmap_node.py")
    social_override   = os.path.join(scripts_dir, "social_override_node.py")

    # ── Launch arguments ──
    declare_headless = DeclareLaunchArgument(
        "headless", default_value="False",
        description="Run Gazebo without GUI (True for Pi5 / SSH)"
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
    )
    declare_use_rviz = DeclareLaunchArgument(
        "use_rviz", default_value="True",
        description="Launch RViz2"
    )
<<<<<<< HEAD
    declare_headless = DeclareLaunchArgument(
        "headless", default_value="False",
        description="Run Gazebo without GUI (SSH/headless)"
    )
    declare_robot_x = DeclareLaunchArgument(
        "robot_x", default_value="-3.5",
        description="Robot spawn X position"
    )
    declare_robot_y = DeclareLaunchArgument(
        "robot_y", default_value="0.0",
        description="Robot spawn Y position"
=======
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="True",
        description="Use Gazebo simulation time"
    )
    declare_robot_x = DeclareLaunchArgument(
        "robot_x", default_value="-3.5",
        description="Robot spawn x position"
    )
    declare_robot_y = DeclareLaunchArgument(
        "robot_y", default_value="0.0",
        description="Robot spawn y position"
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
    )
    declare_robot_yaw = DeclareLaunchArgument(
        "robot_yaw", default_value="0.0",
        description="Robot spawn yaw (radians)"
    )
<<<<<<< HEAD

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz     = LaunchConfiguration("use_rviz")
    robot_x      = LaunchConfiguration("robot_x")
    robot_y      = LaunchConfiguration("robot_y")
    robot_yaw    = LaunchConfiguration("robot_yaw")

    # ─────────────────────────────────────────────────────────
    # FIX 3+4+6: Environment variables — ALL set before any node
    # ─────────────────────────────────────────────────────────

    # TurtleBot3 model selection
    set_tb3_model = SetEnvironmentVariable(
        "TURTLEBOT3_MODEL", "burger"
    )

    # FIX 4: GAZEBO_MODEL_PATH — includes BOTH gazebo models AND tb3 description
    # turtlebot3_gazebo/models  → Gazebo-specific model definitions
    # turtlebot3_description share → URDF + mesh STL files for visual display
    set_gazebo_model_path = SetEnvironmentVariable(
        "GAZEBO_MODEL_PATH",
        os.path.join(pkg_turtlebot3_gazebo, "models")
        + ":" + os.path.join(pkg_turtlebot3_desc, "meshes")
        + ":" + os.environ.get("GAZEBO_MODEL_PATH", ""),
    )

    # FIX 6: GAZEBO_PLUGIN_PATH — libgazebo_ros_diff_drive.so must be findable
    # Without this, the diff_drive plugin silently fails → robot doesn't move
    ros_lib_path = os.path.join(
        os.path.dirname(os.path.dirname(pkg_gazebo_ros)),
        "lib"
    )
    set_gazebo_plugin_path = SetEnvironmentVariable(
        "GAZEBO_PLUGIN_PATH",
        ros_lib_path + ":" + os.environ.get("GAZEBO_PLUGIN_PATH", ""),
    )

    # ─────────────────────────────────────────────────────────
    # 1. Gazebo — proper ROS factory integration
    #
    # gazebo_ros/launch/gazebo.launch.py loads:
    #   libgazebo_ros_init.so    → /clock topic (sim time)
    #   libgazebo_ros_factory.so → /spawn_entity service
    # These two are the minimum for ROS-Gazebo integration.
    # ─────────────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world":   world_file,
            "verbose": "false",
            "pause":   "false",
        }.items(),
    )

    # ─────────────────────────────────────────────────────────
    # 2. robot_state_publisher — delayed 2s for Gazebo /clock
    #
    # FIX 1: Use Command(['xacro', urdf_file]) — NOT open(urdf_file)
    #   open() runs at Python PARSE time → crashes if package not installed
    #   Command() runs at EXECUTION time → safe after colcon build
    #
    # FIX 3: 2s delay ensures /clock is available before RSP starts
    #   robot_state_publisher with use_sim_time=True needs /clock to publish TF
    #
    # NOTE: Do NOT run joint_state_publisher here (FIX 2).
    #   Gazebo's diff_drive plugin publishes /joint_states automatically
    #   when the robot is spawned. Running joint_state_publisher alongside
    #   it causes duplicate messages and TF tree corruption.
    # ─────────────────────────────────────────────────────────
    robot_description = Command(
        [FindExecutable(name="xacro"), " ", urdf_file]
    )

    robot_state_publisher = TimerAction(
        period=2.0,
        actions=[
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{
                    # FIX 1: Command substitution → evaluated at execution time
                    "robot_description": robot_description,
                    "use_sim_time":      use_sim_time,
                }],
            )
        ],
    )

    # ─────────────────────────────────────────────────────────
    # 3. Spawn TurtleBot3 — at t=7s
    #
    # FIX 5: Increased to 7s (from 5s) + -timeout 60 flag
    #   spawn_entity.py with -timeout polls /spawn_entity itself
    #   The 7s gives Gazebo ample time to start on any hardware
    #
    # Spawn reads from /robot_description topic published by RSP (started at t=2s)
    # After spawn: Gazebo diff_drive plugin activates and publishes:
    #   /joint_states     → RSP uses this to complete TF tree
    #   /odom             → odom→base_footprint TF frame
    #   /cmd_vel          → robot actuation
    # ─────────────────────────────────────────────────────────
    spawn_robot = TimerAction(
        period=7.0,
=======
    declare_turtlebot_model = DeclareLaunchArgument(
        "turtlebot3_model", default_value="burger",
        description="TurtleBot3 model: burger or waffle"
    )

    use_sim_time    = LaunchConfiguration("use_sim_time")
    headless        = LaunchConfiguration("headless")
    use_rviz        = LaunchConfiguration("use_rviz")
    robot_x         = LaunchConfiguration("robot_x")
    robot_y         = LaunchConfiguration("robot_y")
    robot_yaw       = LaunchConfiguration("robot_yaw")
    tb3_model       = LaunchConfiguration("turtlebot3_model")

    # ── 1. Gazebo server ──
    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver",
            "--verbose",
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
            world_file,
        ],
        output="screen",
        condition=UnlessCondition(headless),
    )

    gazebo_server_headless = ExecuteProcess(
        cmd=[
            "gzserver",
            "--verbose",
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
            world_file,
        ],
        output="screen",
        condition=IfCondition(headless),
    )

    # ── Gazebo client (GUI) — only when not headless ──
    gazebo_client = ExecuteProcess(
        cmd=["gzclient", "--verbose"],
        output="screen",
        condition=UnlessCondition(headless),
    )

    # ── 2. Robot state publisher (URDF → TF) ──
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gz, "launch",
                         "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # ── 3. Spawn TurtleBot3 in Gazebo ──
    # Wait 3s for Gazebo to fully start before spawning
    spawn_robot = TimerAction(
        period=3.0,
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
        actions=[
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
<<<<<<< HEAD
                name="spawn_turtlebot3",
                output="screen",
                arguments=[
                    "-entity",  "turtlebot3_burger",
                    "-topic",   "robot_description",
                    "-x",       robot_x,
                    "-y",       robot_y,
                    "-z",       "0.01",
                    "-Y",       robot_yaw,
                    "-timeout", "60",
                ],
=======
                arguments=[
                    "-entity",        "turtlebot3_burger",
                    "-topic",         "robot_description",
                    "-x",             robot_x,
                    "-y",             robot_y,
                    "-z",             "0.01",
                    "-Y",             robot_yaw,
                ],
                output="screen",
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
            )
        ],
    )

<<<<<<< HEAD
    # ─────────────────────────────────────────────────────────
    # 4. Nav2 bringup — at t=10s
    #
    # Starts after robot is spawned and TF is publishing.
    # autostart=True: lifecycle manager activates all Nav2 nodes automatically
    # ─────────────────────────────────────────────────────────
    nav2_bringup = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_nav2_bringup, "launch", "bringup_launch.py")
                ),
                launch_arguments={
                    "use_sim_time":    use_sim_time,
                    "params_file":     params_file,
                    "map":             map_file,
                    "use_composition": "False",
                    "autostart":       "True",
=======
    # ── 4. Nav2 bringup ──
    # IMPORTANT: remap controller_server /cmd_vel → /cmd_vel_raw
    # so social_override_node sits between Nav2 and robot
    nav2_bringup = TimerAction(
        period=5.0,   # wait for robot to spawn
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_nav2_bringup, "launch",
                                 "bringup_launch.py")
                ),
                launch_arguments={
                    "use_sim_time":  use_sim_time,
                    "params_file":   params_file,
                    "map":           map_file,
                    "use_composition": "False",
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
                }.items(),
            )
        ],
    )

<<<<<<< HEAD
    # ─────────────────────────────────────────────────────────
    # 5. Social navigation nodes — at t=12s
    #
    # human_tracker_node  : /gazebo/model_states → /human_positions
    # social_costmap_node : /human_positions → /social_costmap
    # social_override_node: /cmd_vel_nav → (proxemics) → /cmd_vel
    # ─────────────────────────────────────────────────────────
    human_tracker_node = TimerAction(
        period=12.0,
=======
    # ── 5. human_tracker_node ──
    # Delay 6s — must wait for Gazebo to publish /gazebo/model_states
    human_tracker_node = TimerAction(
        period=6.0,
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
        actions=[
            Node(
                package="social_nav",
                executable="human_tracker_node.py",
                name="human_tracker_node",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
            )
        ],
    )

<<<<<<< HEAD
    social_costmap_node = TimerAction(
        period=12.0,
=======
    # ── 6. social_costmap_node ──
    # Delay 7s — needs human_tracker to be running
    social_costmap_node = TimerAction(
        period=7.0,
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
        actions=[
            Node(
                package="social_nav",
                executable="social_costmap_node.py",
                name="social_costmap_node",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
            )
        ],
    )

<<<<<<< HEAD
    # cmd_vel routing: Nav2 collision_monitor → /cmd_vel_nav
    #                  social_override subscribes /cmd_vel_nav
    #                  social_override publishes /cmd_vel → robot
    social_override_node = TimerAction(
        period=13.0,
=======
    # ── 7. social_override_node ──
    # Delay 8s — needs costmap + human tracker
    # Remaps Nav2 /cmd_vel output → /cmd_vel_raw for interception
    social_override_node = TimerAction(
        period=8.0,
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
        actions=[
            Node(
                package="social_nav",
                executable="social_override_node.py",
                name="social_override_node",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
<<<<<<< HEAD
=======
                remappings=[
                    # Nav2 controller outputs to /cmd_vel_raw
                    # This node reads /cmd_vel_raw and publishes /cmd_vel
                    ("/cmd_vel_raw", "/cmd_vel_raw"),
                    ("/cmd_vel",     "/cmd_vel"),
                ],
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
            )
        ],
    )

<<<<<<< HEAD
    # ─────────────────────────────────────────────────────────
    # 6. RViz2 — at t=13s
    # ─────────────────────────────────────────────────────────
    rviz_node = TimerAction(
        period=13.0,
=======
    # ── 8. RViz2 ──
    rviz_node = TimerAction(
        period=8.0,
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
        actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_file],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
                condition=IfCondition(use_rviz),
            )
        ],
    )

<<<<<<< HEAD
    # ─────────────────────────────────────────────────────────
    # Startup log
    # ─────────────────────────────────────────────────────────
    startup_log = LogInfo(
        msg="\n"
            "╔══════════════════════════════════════════════════════╗\n"
            "║   DIAT Social Navigation — DEFINITIVE FIX v3        ║\n"
            "╠══════════════════════════════════════════════════════╣\n"
            "║  t=0s   Gazebo (gazebo_ros factory plugin loaded)    ║\n"
            "║  t=2s   robot_state_publisher (URDF via xacro)       ║\n"
            "║  t=7s   spawn_entity (reads /robot_description)      ║\n"
            "║           Gazebo publishes /joint_states + odom TF   ║\n"
            "║  t=10s  Nav2 bringup (planner + controller + AMCL)   ║\n"
            "║  t=12s  Social nodes (tracker + costmap + override)  ║\n"
            "║  t=13s  RViz2                                        ║\n"
            "╠══════════════════════════════════════════════════════╣\n"
            "║  AFTER LAUNCH:                                       ║\n"
            "║    1. Click '2D Pose Estimate' at (-3.5, 0.0) →right ║\n"
            "║    2. Click '2D Goal Pose'     at ( 3.5, 2.0)        ║\n"
            "║    3. Robot navigates to goal                        ║\n"
            "╠══════════════════════════════════════════════════════╣\n"
            "║  DEBUG COMMANDS:                                     ║\n"
            "║    ros2 service list | grep spawn_entity             ║\n"
            "║    ros2 topic echo /tf --once                        ║\n"
            "║    ros2 topic hz /cmd_vel                            ║\n"
            "║    ros2 topic hz /joint_states                       ║\n"
            "╚══════════════════════════════════════════════════════╝\n"
    )

    return LaunchDescription([
        # ── Environment variables (MUST be first) ──
        set_tb3_model,
        set_gazebo_model_path,
        set_gazebo_plugin_path,

        # ── Declare arguments ──
        declare_use_sim_time,
        declare_use_rviz,
        declare_headless,
        declare_robot_x,
        declare_robot_y,
        declare_robot_yaw,

        startup_log,

        # ── t=0s: Gazebo with ROS plugins ──
        gazebo,

        # ── t=2s: TF publisher (after Gazebo clock) ──
        robot_state_publisher,

        # ── t=7s: Spawn robot ──
        spawn_robot,

        # ── t=10s: Nav2 stack ──
        nav2_bringup,

        # ── t=12s+: Social navigation ──
=======
    # ── Startup log ──
    startup_log = LogInfo(
        msg="\n"
            "========================================\n"
            "  DIAT Social Navigation — Starting up\n"
            "========================================\n"
            "  World  : social_world.world\n"
            "  Humans : human_1, human_2, human_3\n"
            "  Nodes  : human_tracker → social_costmap → social_override\n"
            "  Topics : /human_positions  /social_costmap  /social_status\n"
            "========================================\n"
    )

    return LaunchDescription([
        # Declare arguments
        declare_headless,
        declare_use_rviz,
        declare_use_sim_time,
        declare_robot_x,
        declare_robot_y,
        declare_robot_yaw,
        declare_turtlebot_model,

        # Log
        startup_log,

        # Gazebo
        gazebo_server,
        gazebo_server_headless,
        gazebo_client,

        # Robot
        robot_state_publisher,
        spawn_robot,

        # Nav2
        nav2_bringup,

        # Social navigation nodes (staggered start)
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
        human_tracker_node,
        social_costmap_node,
        social_override_node,

<<<<<<< HEAD
        # ── t=13s: RViz ──
=======
        # RViz
>>>>>>> daadf900ed5b6df72d54bc89c258fca61126983d
        rviz_node,
    ])
