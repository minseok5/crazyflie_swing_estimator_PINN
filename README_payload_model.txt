Crazyflie slung-payload prototype files

Files:
- create_payload_model.py: generates models/crazyflie/cf21B_500_payload.xml from the existing base XML without modifying it.
- hover_sim_payload_viewer.py: loads the new XML and runs a basic PID hover viewer.

Place both Python files in the repository root (next to hover_sim_viewer.py).
From the repository root, run:
  python create_payload_model.py
  python hover_sim_payload_viewer.py

Initial prototype parameters:
- Payload mass: 0.020 kg
- Pendulum length: 0.200 m
- Two orthogonal hinge joints
- Rod visualization is non-colliding; the nested bodies/joints supply the pendulum dynamics.
- Controller feed-forward mass is set to drone + payload mass.

Important validation note:
These files are a prototype and have not been run in the user's local MuJoCo environment. In particular, the requested initial position (0, 0, 0) is retained, but because the model's drone collision sphere extends below the ground at z=0, contact behavior should be checked. If it contacts the ground, raise INITIAL_POS and POS_DES together, e.g. to [0, 0, 0.5].
