#!/usr/bin/env python3
"""
integration.launch.py
======================
Launches all Module 10 nodes together.

Nodes started:
  1. main_controller   — central FSM orchestrator
  2. head_controller   — EZ-Robot pan/tilt via USB serial
  3. health_bridge     — watchdog for all 9 modules

Usage:
  ros2 launch integration integration.launch.py
  ros2 launch integration integration.launch.py serial_port:=/dev/ttyUSB1
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    declare_serial = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyUSB0",
        description="USB serial port for EZ-Robot head (IoTiny)"
    )

    startup_log = LogInfo(
        msg=(
            "\n╔═══════════════════════════════════════════════╗"
            "\n║   DIAT Social Robot — Module 10 Integration  ║"
            "\n╠═══════════════════════════════════════════════╣"
            "\n║  main_controller  → FSM orchestrator          ║"
            "\n║  head_controller  → EZ-Robot head (USB)       ║"
            "\n║  health_bridge    → Module watchdog           ║"
            "\n╚═══════════════════════════════════════════════╝\n"
        )
    )

    main_controller = Node(
        package="integration",
        executable="main_controller",
        name="integration_controller",
        output="screen",
        emulate_tty=True,
    )

    head_controller = Node(
        package="integration",
        executable="head_controller",
        name="head_controller_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "serial_port": LaunchConfiguration("serial_port"),
        }],
    )

    health_bridge = Node(
        package="integration",
        executable="health_bridge",
        name="health_bridge_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        declare_serial,
        startup_log,
        main_controller,
        head_controller,
        health_bridge,
    ])
