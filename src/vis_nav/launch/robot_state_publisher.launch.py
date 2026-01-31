import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    # urdf_file_name = 'td_robot.urdf' /home/regmed/dregmed/vis_to_nav/src/vis_nav/description/urdf/robot_w.urdf
    urdf_file_name = 'robot_w.urdf'

    # print(' s  urdf_file_name : {}'.format(urdf_file_name))
    

    # urdf = os.path.join(
    #     get_package_share_directory('vis_nav'),
    #     'urdf',
    #     urdf_file_name)
    
    url = '/home/regmed/dregmed/vis_to_nav/install/vis_nav/share/vis_nav/description'
    urdf = os.path.join(url,
        'urdf',
        urdf_file_name)
    # print("testing")
    # print(urdf)
    
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'scout',
                                   '-x','0.0', 
                                   '-y','2.0',
                                   '-z','0.0'],
                        output='screen'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                         'use_sim_time': use_sim_time,
                         'robot_description': robot_desc,
                       }],
            arguments=[urdf]),
                                
    ])








