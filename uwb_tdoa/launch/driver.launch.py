import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("uwb_tdoa"), "config", "tdoa.yaml"
    )

    return LaunchDescription(
        [
            Node(
                package="uwb_tdoa",
                executable="driver",
                name="tdoa_driver",
                namespace="uwb/tdoa",
                output="screen",
                parameters=[config],
                # mimic ROS 1 required='true': bring the launch down if the node exits
                on_exit=Shutdown(),
            ),
        ]
    )
