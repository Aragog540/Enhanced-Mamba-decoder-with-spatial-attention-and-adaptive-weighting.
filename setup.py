
from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-unetmamba",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Enhanced UNetMamba with Spatial Attention and Adaptive Background Weighting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Enhanced-UNetMamba",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0", 
            "isort>=5.0.0",
            "flake8>=6.0.0",
        ],
        "wandb": ["wandb>=0.15.0"],
        "advanced": [
            "transformers>=4.30.0",
            "segment-anything>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "enhanced-unetmamba-train=enhanced_unetmamba.training.train:main",
            "enhanced-unetmamba-test=enhanced_unetmamba.testing.test:main",
        ],
    },
    include_package_data=True,
    package_data={
        "enhanced_unetmamba": [
            "config/**/*.py",
            "config/**/*.yaml",
        ],
    },
)
