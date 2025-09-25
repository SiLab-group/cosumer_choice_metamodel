"""
Setup script for the Consumer Choice Metamodel Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="consumer-choice-metamodel",
    version="1.0.0",
    author="Consumer Choice Metamodel Framework",
    description="Abstract framework for consumer choice modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add any dependencies here
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
)
