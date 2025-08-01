# Unitree Go2 Environments

This directory contains the Unitree Go2 robot locomotion environments for reinforcement learning training and evaluation.

## Available Environments

### Standard Velocity Environments

- **`Unitree-Go2-Velocity`**: Standard velocity control environment for training. It could be used as teacher network
- **`Unitree-Go2-Velocity-Play`**: Velocity control environment for evaluation/playback

### Safe Velocity Environments

- **`Unitree-Go2-Velocity-Safe`**: Safe velocity control environment for training with safety constraints
- **`Unitree-Go2-Velocity-Safe-Play`**: Safe velocity control environment for evaluation/playback
- **`Unitree-Go2-Velocity-Safe-Keyboard`**: Safe velocity control environment with keyboard input support

### Vision-Based Velocity Environments

- **`Unitree-Go2-Velocity-Vision`**: Vision-based velocity control environment for training
- **`Unitree-Go2-Velocity-Vision-Play`**: Vision-based velocity control environment for evaluation/playback
- **`Unitree-Go2-Velocity-Vision-Keyboard`**: Vision-based velocity control environment with keyboard input support

### Real Robot Environments

- **`Unitree-Go2-Velocity-Real`**: Real robot velocity control environment for training without height scanner
- **`Unitree-Go2-Velocity-Real-Play`**: Real robot velocity control environment for evaluation/playback

### Distillation Robot Environments

- **`Unitree-Go2-Velocity-Distillation`**: Distillation velocity control environment for training with teacher-student learning
- **`Unitree-Go2-Velocity-Distillation-Play`**: Distillation velocity control environment for evaluation/playback