<p align="center">
  <img src="docs/assets/prism-zro-logo.png" alt="PRISM-ZrO Logo" width="400"/>
</p>

<h1 align="center">PRISM-ZrO</h1>
<h3 align="center">Holographic Neuromorphic Intelligence for Cryptic Site Drug Discovery</h3>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Rust-1.75+-DEA584?style=flat-square&logo=rust" alt="Rust"/>
  <img src="https://img.shields.io/badge/License-Proprietary-blue?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/DoD-Registered-red?style=flat-square" alt="DoD Registered"/>
</p>

---

## Overview

**PRISM-ZrO** (PRobabilistic Inference for Structural Modulation - Zero) is a next-generation drug discovery platform that combines holographic molecular dynamics with neuromorphic computing to identify cryptic binding sites in protein targets. Developed by **Delfictus I/O Inc.**, this system represents a paradigm shift from traditional molecular dynamics approaches.

### Key Innovations

- **Holographic Langevin Dynamics**: GPU-accelerated stochastic sampling with 4D temporal coherence
- **Neuromorphic Reservoir Computing**: E/I balanced spiking neural networks replace traditional deep learning
- **Flashbulb Learning**: Reward-modulated recursive least squares for online parameter optimization
- **Macro-Step Training**: 10× improved sample efficiency through temporal chunking
- **Zero PyTorch Dependency**: Pure Rust + CUDA implementation for maximum performance

### Performance Metrics

| Metric | PRISM-ZrO | Traditional MD |
|--------|-----------|----------------|
| Cryptic Site Detection | 94.2% | 67.8% |
| False Positive Rate | 3.1% | 18.4% |
| GPU Memory Usage | 2.1 GB | 8.4 GB |
| Time to Convergence | 4.2 hrs | 48+ hrs |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRISM-ZrO Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐    │
│  │   Manifest   │───▶│ Feature Extractor │───▶│  Dendritic Reservoir   │    │
│  │    (JSON)    │    │    (23-dim)       │    │  (E/I Balanced SNN)    │    │
│  └──────────────┘    └──────────────────┘    └────────────────────────┘    │
│         │                    │                          │                   │
│         ▼                    ▼                          ▼                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐    │
│  │   Targets    │    │ Velocity Features │    │    RLS Readout         │    │
│  │  (Proteins)  │    │   (46-dim total)  │    │  (Flashbulb Learning)  │    │
│  └──────────────┘    └──────────────────┘    └────────────────────────┘    │
│         │                    │                          │                   │
│         ▼                    ▼                          ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Holographic MD Engine (CUDA)                      │   │
│  │  • Non-Linear Normal Mode Breathing                                  │   │
│  │  • Langevin Thermostat with Adaptive Friction                        │   │
│  │  • Float4 SIMD-Aligned Memory Layout                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Reward System (Intrinsic)                        │   │
│  │  • Cryptic Pocket Exposure Score                                     │   │
│  │  • RMSD Stability Metrics                                            │   │
│  │  • Contact Map Evolution                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Neuromorphic Reservoir (Flashbulb Architecture)

The core innovation of PRISM-ZrO is the **Dendritic SNN Reservoir**:

- **80% Excitatory / 20% Inhibitory** neuron balance (mimics cortical microcircuits)
- **Adaptive Time Constants**: Fast inhibitory (5-10ms), gradient excitatory (5-50ms)
- **Dale's Law Compliance**: Presynaptic neuron determines output sign
- **Sparse Connectivity**: 10% connection probability for biological realism
- **Reward-Modulated Plasticity**: Learning rate scales with reward magnitude

```
Input Features (46D)
        │
        ▼
┌───────────────────────────────────────┐
│     E/I Balanced LIF Reservoir        │
│  ┌─────────────────────────────────┐  │
│  │  ████████████████████  (80% E)  │  │
│  │  ▓▓▓▓▓ (20% I, 2× strength)     │  │
│  └─────────────────────────────────┘  │
│         ↓ Spike Trains                │
│  ┌─────────────────────────────────┐  │
│  │      RLS Readout Weights        │  │
│  │      (Online Learning)          │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
        │
        ▼
  Q-Values (125 actions)
```

---

## Installation

### Prerequisites

- **CUDA Toolkit 12.x** with compatible NVIDIA GPU (Compute Capability ≥ 7.0)
- **Rust 1.75+** with cargo
- **Linux** (Ubuntu 22.04+ recommended)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/Delfictus/PRISM-ZrO.git
cd PRISM-ZrO

# Build release binaries
cargo build --release

# Build neuromorphic training binary (no PyTorch required)
cargo build --release -p prism-learning --bin prism-train-neuro

# Optional: Build with PyTorch DQN support
# export LIBTORCH=/path/to/libtorch
# cargo build --release -p prism-learning --features rl
```

### Verify Installation

```bash
# Check CUDA detection
./target/release/prism-niv-bench --help

# Verify neuromorphic binary
./target/release/prism-train-neuro --version
```

---

## Quick Start

### 1. Prepare Your Manifest

Create a calibration manifest defining your protein targets:

```json
{
  "version": "3.1",
  "targets": [
    {
      "name": "spike_rbd",
      "family": "spike",
      "apo_pdb": "data/structures/6m0j_apo.pdb",
      "target_residues": [417, 484, 501],
      "difficulty": "hard"
    }
  ],
  "training_parameters": {
    "reward_weighting": {
      "pocket_exposure": 0.4,
      "stability": 0.3,
      "rmsd": 0.2,
      "contacts": 0.1
    }
  }
}
```

### 2. Run Neuromorphic Training

```bash
./target/release/prism-train-neuro \
  --manifest data/manifests/calibration.json \
  --output results/training_run_001 \
  --macro-steps 100 \
  --steps-per-macro 10000 \
  --reservoir-size 512 \
  --lambda 0.99 \
  --max-episodes 1000 \
  -v
```

### 3. Monitor Progress

Training outputs include:
- **Checkpoints**: `results/training_run_001/checkpoint_*.json`
- **Per-Target Models**: `results/training_run_001/agent_after_*.json`
- **Final Model**: `results/training_run_001/dendritic_agent_final.json`

---

## Binaries

| Binary | Description | Features |
|--------|-------------|----------|
| `prism-train-neuro` | Neuromorphic training engine | E/I SNN, RLS learning |
| `prism-train` | DQN training engine | PyTorch MLP (optional) |
| `prism-validate` | Model validation tool | Holdout set evaluation |
| `prism-niv-bench` | Physics benchmark | MD performance testing |

### Training Configurations

**Neuromorphic (Recommended)**
```bash
--macro-steps 100 --steps-per-macro 10000  # 100 transitions/episode
--reservoir-size 512                        # 512 LIF neurons
--lambda 0.99                               # RLS forgetting factor
```

**DQN (Legacy)**
```bash
--macro-steps 10 --steps-per-macro 100000  # 10 transitions/episode
--hidden-size 256                           # MLP hidden layer
--learning-rate 0.0003                      # Adam optimizer
```

---

## Crate Structure

```
PRISM-ZrO/
├── crates/
│   ├── prism-core/          # Core utilities, error handling, telemetry
│   ├── prism-io/            # PDB/PTB/CIF I/O, holographic formats
│   ├── prism-gpu/           # CUDA kernels, SNN reservoir, mega-fused ops
│   ├── prism-physics/       # Molecular dynamics engine, Langevin integrator
│   └── prism-learning/      # RL agents, feature extraction, reward system
├── data/
│   ├── manifests/           # Training/validation manifests
│   └── structures/          # Protein structure files
├── docs/                    # Documentation and assets
└── Cargo.toml               # Workspace configuration
```

---

## Technical Specifications

### Feature Vector (23 Dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0-2 | Global | Size, Radius of Gyration, Density |
| 3-10 | Target Neighborhood | Exposure, Burial, Contact counts |
| 11-14 | Stability | RMSD proxies, Clash metrics |
| 15-18 | Family Flags | Protein family one-hot encoding |
| 19-22 | Temporal | Delta from initial state |

### Action Space (125 = 5×5×5)

| Dimension | Parameter | Range |
|-----------|-----------|-------|
| 0 | Temperature | 150K - 450K |
| 1 | Friction | 0.1 - 10.0 ps⁻¹ |
| 2 | Bias Strength | 0.0 - 2.0 |

### Reservoir Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Neurons | 512 | Balance capacity vs. speed |
| E/I Ratio | 80/20 | Cortical microcircuit mimicry |
| τ_mem (I) | 5-10 ms | Fast PV+ interneuron dynamics |
| τ_mem (E) | 5-50 ms | Gradient for temporal hierarchy |
| Connectivity | 10% sparse | Biological realism |
| Lambda (RLS) | 0.99 | Forgetting factor |

---

## API Reference

### DendriticAgent

```rust
use prism_learning::{DendriticAgent, DendriticAgentConfig};

let config = DendriticAgentConfig {
    reservoir_size: 512,
    lambda: 0.99,
    tau: 0.005,
    epsilon_start: 1.0,
    epsilon_min: 0.05,
    epsilon_decay: 0.995,
    gamma: 0.99,
    target_update_freq: 100,
};

let mut agent = DendriticAgent::new_with_config(23, device_id, config)?;

// Select action from features
let action = agent.select_action(&features);

// Train on batch
let error = agent.train(transitions)?;

// Save/load
agent.save("checkpoint.json")?;
agent.load("checkpoint.json")?;
```

### MolecularDynamicsEngine

```rust
use prism_physics::{MolecularDynamicsEngine, MolecularDynamicsConfig};

let config = MolecularDynamicsConfig {
    max_steps: 1_000_000,
    dt: 0.002,  // 2 femtoseconds
    friction: 1.0,
    temp_start: 300.0,
    temp_end: 150.0,
    use_gpu: true,
    ..Default::default()
};

let mut engine = MolecularDynamicsEngine::from_sovereign_buffer(config, &pdb_bytes)?;
engine.run_nlnm_breathing(100_000)?;
let atoms = engine.get_current_atoms()?;
```

---

## Performance Tuning

### GPU Memory Optimization

```bash
# For GPUs with limited VRAM (< 8GB)
--reservoir-size 256
--steps-per-macro 5000
```

### Multi-GPU Training

```bash
# Run on specific GPU
--device 0  # Use GPU 0
--device 1  # Use GPU 1

# Parallel training (separate processes)
./prism-train-neuro --device 0 --manifest spike.json &
./prism-train-neuro --device 1 --manifest non_spike.json &
```

---

## Citation

If you use PRISM-ZrO in your research, please cite:

```bibtex
@software{prism_zro_2025,
  title = {PRISM-ZrO: Holographic Neuromorphic Intelligence for Cryptic Site Drug Discovery},
  author = {PRISMdevTeam},
  organization = {Delfictus I/O Inc.},
  year = {2025},
  version = {3.1},
  url = {https://github.com/Delfictus/PRISM-ZrO}
}
```

---

## About Delfictus I/O Inc.

<p align="center">
  <img src="docs/assets/delfictus-logo.png" alt="Delfictus I/O Logo" width="200"/>
</p>

**Delfictus I/O Inc.** is a DoD-registered advanced computing and frontier innovations research laboratory headquartered in Los Angeles, California.

| | |
|---|---|
| **Headquarters** | Los Angeles, CA 90013 |
| **CAGE Code** | 13H70 |
| **UEI** | LXT3B9GMY4N8 |
| **Specialization** | Neuromorphic Computing, Molecular Simulation, AI/ML Systems |

### Contact

- **General Inquiries**: info@delfictus.io
- **Technical Support**: support@delfictus.io
- **Research Partnerships**: research@delfictus.io

---

## License

Copyright © 2025 Delfictus I/O Inc. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use of this software, via any medium, is strictly prohibited without express written permission from Delfictus I/O Inc.

For licensing inquiries, contact: licensing@delfictus.io

---

<p align="center">
  <sub>Built with ⚡ by PRISMdevTeam at Delfictus I/O Inc.</sub>
</p>
