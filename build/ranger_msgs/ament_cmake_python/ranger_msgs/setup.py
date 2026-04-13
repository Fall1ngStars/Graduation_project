from setuptools import find_packages
from setuptools import setup

setup(
    name='ranger_msgs',
    version='0.0.1',
    packages=find_packages(
        include=('ranger_msgs', 'ranger_msgs.*')),
)
