from setuptools import find_packages, setup
import os
from glob import glob 

package_name = 'vis_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')) 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='regmed',
    maintainer_email='ah.regragui@edu.umi.ac.ma',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main.py = vis_nav.main:main',
            'testing.py = vis_nav.testing:main',
            'depth_image_subscriber.py = vis_nav.depth_image_subscriber:main',
            'demonstration.py = vis_nav.demonstration:main',
            'keyboard_control.py = vis_nav.keyboard_control:main',
            
        ],
    },
)
