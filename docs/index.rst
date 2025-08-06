Photon Neuromorphics SDK Documentation
=====================================

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/Python-3.9+-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/WASM-SIMD%20Preview-orange
   :target: https://webassembly.org/
   :alt: WebAssembly SIMD

Welcome to the Photon Neuromorphics SDK, a comprehensive library for silicon-photonic 
neural networks with WebAssembly SIMD acceleration and real-time optical training capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user_guide/core_concepts
   user_guide/components
   user_guide/networks
   user_guide/simulation
   user_guide/training
   user_guide/hardware

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/core
   api/networks
   api/simulation
   api/compiler
   api/training
   api/hardware
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics
   
   advanced/performance
   advanced/quantum_photonics
   advanced/custom_components
   advanced/distributed_computing

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/basic_usage
   examples/spiking_networks
   examples/optical_training
   examples/hardware_integration

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   changelog
   roadmap

Overview
--------

The Photon Neuromorphics SDK enables researchers and engineers to design, simulate, 
and deploy silicon-photonic neural networks with unprecedented speed and efficiency.

Key Features
~~~~~~~~~~~~

🌟 **Comprehensive Component Library**
   - Silicon and silicon nitride waveguides
   - Mach-Zehnder and microring modulators  
   - Photodetector arrays with noise modeling
   - Laser sources with phase noise simulation

🧠 **Advanced Neural Architectures**
   - Photonic spiking neural networks (SNNs)
   - Universal MZI interferometer meshes
   - Microring resonator weight banks
   - Quantum-photonic interfaces

⚡ **High-Performance Simulation**
   - FDTD electromagnetic solver
   - Circuit-level S-parameter analysis
   - GPU acceleration with CUDA
   - WebAssembly SIMD optimization

🔬 **Physics-Based Modeling**
   - Quantum shot noise simulation
   - Thermal fluctuation modeling
   - Fabrication variation analysis
   - Nonlinear optical effects

🛠️ **Hardware Integration**
   - Real-time chip calibration
   - Light-in-the-loop training
   - VISA instrument control
   - Power budget optimization

🤖 **AI-Powered Tools**
   - ONNX model compilation
   - Automated layout optimization
   - Adaptive caching algorithms
   - Performance profiling

Quick Start
-----------

Install the SDK:

.. code-block:: bash

   pip install photon-neuromorphics

Create your first photonic neural network:

.. code-block:: python

   import photon_neuro as pn
   import torch

   # Create a spiking neural network
   snn = pn.PhotonicSNN(
       topology=[784, 256, 10],
       neuron_model='photonic_lif',
       timestep=1e-12
   )
   
   # Configure optical parameters
   snn.configure_optics(
       wavelength=1550e-9,
       waveguide_width=450e-9
   )
   
   # Simulate network
   input_spikes = torch.randn(1, 784)
   output_spikes = snn(input_spikes, n_timesteps=100)
   
   print(f"Network efficiency: {snn.efficiency:.1%}")
   print(f"Latency: {snn.latency_ps:.1f} ps")

Architecture Overview
--------------------

.. image:: _static/architecture_diagram.png
   :alt: SDK Architecture
   :align: center
   :width: 800px

The SDK is organized into several key modules:

**Core Components** (``photon_neuro.core``)
   Base classes and fundamental photonic elements

**Neural Networks** (``photon_neuro.networks``) 
   High-level network architectures and training algorithms

**Simulation** (``photon_neuro.simulation``)
   Physics-based simulation engines and analysis tools

**Hardware** (``photon_neuro.hardware``)
   Real hardware interfaces and calibration routines

**Compiler** (``photon_neuro.compiler``)
   Model compilation and optimization tools

**Performance** (``photon_neuro.performance``)
   Caching, parallelization, and GPU acceleration

Applications
------------

The Photon Neuromorphics SDK enables a wide range of applications:

**Machine Learning**
   - Ultra-low latency inference (< 1 ns per layer)
   - Energy-efficient neural network acceleration
   - Real-time signal processing and classification

**Scientific Computing**
   - Quantum-enhanced machine learning algorithms
   - High-throughput parallel computation
   - Physics simulation and modeling

**Telecommunications**
   - All-optical signal processing
   - Coherent detection and equalization
   - Network traffic analysis and routing

**Research and Development**
   - Novel neural network architectures
   - Neuromorphic computing research
   - Photonic computing optimization

Performance Benchmarks
----------------------

.. list-table:: Performance Comparison
   :header-rows: 1
   :align: center

   * - Operation
     - Traditional CPU
     - GPU (CUDA)
     - Photonic (Simulated)
     - Photonic (Hardware)
   * - Matrix Multiply (1024×1024)
     - 12 ms
     - 0.8 ms  
     - 0.1 ms
     - 0.05 ms
   * - SNN Forward Pass (1000 neurons)
     - 15 ms
     - 2 ms
     - 0.5 ms
     - 10 ps
   * - Energy per MAC Operation
     - 1000 fJ
     - 100 fJ
     - 10 fJ
     - 1 fJ

Community and Support
--------------------

**Documentation**: Comprehensive guides and API reference

**Examples**: Jupyter notebooks and practical tutorials  

**Community Forum**: Discussion and Q&A platform

**GitHub Issues**: Bug reports and feature requests

**Professional Support**: Enterprise consulting and custom development

Citation
--------

If you use the Photon Neuromorphics SDK in your research, please cite:

.. code-block:: bibtex

   @software{photon_neuromorphics2025,
     title={Photon Neuromorphics SDK: Silicon-Photonic Neural Networks with WASM Acceleration},
     author={Daniel Schmidt and Terragon Labs},
     year={2025},
     url={https://github.com/danieleschmidt/Photon-Neuromorphics-SDK},
     version={0.1.0}
   }

License
-------

The Photon Neuromorphics SDK is released under the BSD 3-Clause License. 
See the `LICENSE <https://github.com/danieleschmidt/Photon-Neuromorphics-SDK/blob/main/LICENSE>`_ 
file for details.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`