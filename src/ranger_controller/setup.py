from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'ranger_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='meteor',
    maintainer_email='meteor@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_follower_node = ranger_controller.aruco_follower_node:main',
            'pointcloud_follower_node = ranger_controller.pointcloud_follower_node:main',
            'hybrid_controller_node = ranger_controller.hybrid_controller_node:main',
            'check_battery_node = ranger_controller.check_battery_node:main'
        ],
    },
)
