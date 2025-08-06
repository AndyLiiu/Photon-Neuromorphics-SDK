from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="photon-neuromorphics",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@photon-neuro.io",
    description="Silicon-photonic spiking neural network library with WebAssembly SIMD acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/Photon-Neuromorphics-SDK",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "onnx>=1.12.0",
        "matplotlib>=3.5.0",
        "h5py>=3.6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "hardware": [
            "pyvisa>=1.11.0",
            "serial>=0.0.97",
            "pyserial>=3.5",
        ],
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "photon-neuro=photon_neuro.cli:main",
        ],
    },
)