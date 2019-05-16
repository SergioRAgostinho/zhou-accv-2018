from setuptools import setup
from distutils.core import Extension



def install_requires():
    """Generate list with dependency requirements"""

    deps = []
    with open("requirements.txt", "r") as f:
        for line in f:
            deps.append(line[:-1])
    return deps


def long_description():
    with open("README.md", "r") as f:
        return f.read()


e3q3c_module = Extension("zhou_accv_2018.e3q3c", sources=["zhou_accv_2018/e3q3cmodule.c"])

setup(
    name="zhou_accv_2018",
    version="1.0.0",
    license="Apache 2.0",
    description="A Stable Algebraic Camera Pose Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences.",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    install_requires=install_requires(),
    author="Sérgio Agostinho",
    author_email="sergio@sergioagostinho.com",
    url="https://github.com/SergioRAgostinho/zhou-accv-2018",
    packages=["zhou_accv_2018"],
    ext_modules=[e3q3c_module],
    python_requires=">=3.5",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
)
