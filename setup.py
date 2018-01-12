from setuptools import setup, find_packages

VERSION = '0.0.1'

long_description = "Manually fused PyTorch Capsule Layer"

setup_info = dict(
    # Metadata
    name='capsule_layer',
    version=VERSION,
    author='Left Thomas',
    author_email='leftthomas@qq.com',
    url='https://github.com/leftthomas/CapsuleLayer',
    description='Manually fused PyTorch Capsule Layer',
    long_description=long_description,
    license='BSD',

    # Package info
    packages=find_packages(exclude=('test',)),

    zip_safe=True,

    install_requires=[
        'torch',
        'cupy',
    ]
)

setup(**setup_info)
