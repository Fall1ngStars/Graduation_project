# aruco_detector/setup.py
from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'aruco_detector'

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
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Aruco marker detection for AGV charging system',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector_node = aruco_detector.aruco_detector_node_final:main',
            'multi_modal_guidance_node = aruco_detector.multi_modal_guidance_node:main',
            'system_monitor_node = aruco_detector.system_monitor_node:main',
            'generate_aruco_markers = aruco_detector.generate_aruco_markers:main',
        ],
    },
)
