#!/usr/bin/env python

import os

from setuptools import setup, find_packages

this_file = os.path.dirname(__file__)

setup(
    name="capsule_layer",
    version="0.1",
    description="An capsule layer using PyTorch FFI",
    url="https://github.com/leftthomas/CapsuleLayer",
    author="leftthomas",
    author_email="leftthomas@qq.com",
    # Require cffi and PyTorch.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)
