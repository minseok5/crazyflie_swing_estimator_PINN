# Crazyflie Swing Simulation

A standalone MuJoCo simulation for learning-based swing-state estimation of a quadrotor carrying a suspended payload.

## Current Scope

- Crazyflie 2.1 Brushless model
- PID-based flight control
- Motor dynamics model
- Hover simulation

## Requirements

- Python 3.10 or later
- MuJoCo
- NumPy
- Matplotlib

## Run Simulation

```bash
python hover_sim.py
```

## Interactive Viewer

```bash
python hover_sim_viewer.py
```

## Run Tests

```bash
python -m pytest tests/
```
