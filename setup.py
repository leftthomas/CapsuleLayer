from setuptools import setup, find_packages

VERSION = '0.1.8'

long_description = "PyTorch Capsule Layer, include conv2d and linear layers."

setup_info = dict(
    # Metadata
    name='capsule_layer',
    version=VERSION,
    author='Hao Ren',
    author_email='leftthomas@qq.com',
    url='https://github.com/leftthomas/CapsuleLayer',
    description='PyTorch Capsule Layer',
    long_description=long_description,
    license='MIT',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True
)

setup(**setup_info, install_requires=['torch'])
