# DDPG for Continuous Control with TensorFlow & Gymnasium

This repository contains a clean, optimized implementation of the **Deep Deterministic Policy Gradient (DDPG)** algorithm using **TensorFlow 2.x** and **Gymnasium**.

The code is specifically configured to solve the `Pendulum-v1` environment, a classic continuous control problem where the goal is to swing a pendulum up and balance it in an inverted position using continuous torque values.

## 🌟 Key Features

* **Ornstein-Uhlenbeck Action Noise:** Implements correlated noise (`OUActionNoise`) to mimic physical inertia, aiding efficient exploration in continuous environments.
* **Actor-Critic Architecture:** Uses the Keras Functional API for clean model definitions.
    * **Actor:** Maps states to specific continuous actions.
    * **Critic:** Predicts Q-values (future rewards) for state-action pairs.
* **Performance Optimization:** Utilizes `@tf.function` to compile the training step into a static graph, significantly speeding up the training loop.
* **Stability Mechanisms:** Includes a **Replay Buffer** for off-policy learning and **Target Networks** with Polyak averaging (soft updates) to stabilize convergence.

## 🛠️ Dependencies

To run this code, you will need Python installed along with the following libraries:

```bash
pip install tensorflow gymnasium numpy matplotlib
