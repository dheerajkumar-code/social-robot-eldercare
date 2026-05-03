from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='integration',
            executable='main_controller',
            name='integration_controller',
            output='screen',
            emulate_tty=True,
        )
    ])
