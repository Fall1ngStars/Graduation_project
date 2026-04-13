# setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'pointcloud_refinement'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'opencv-python', 'open3d', 'scipy'],
    zip_safe=True,
    maintainer='meteor',
    maintainer_email='meteor@example.com',
    description='Point cloud refinement for AGV charging',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_capture_tool = pointcloud_refinement.capture_tool:main',
            'pointcloud_refinement_node = pointcloud_refinement.refinement_node:main',
            'pointcloud_controller_node = pointcloud_refinement.pointcloud_controller_node:main'
        ],
    },
)
