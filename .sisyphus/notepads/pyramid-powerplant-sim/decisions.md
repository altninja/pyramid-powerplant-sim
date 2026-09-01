# Decisions - Christopher Dunn Giza Power Plant Simulation

## Architectural Decisions
1. **Separation of Concerns**: Python multi-physics engine for rigorous numerical solvers; Three.js static replay for 60 FPS visual telemetry demonstration.
2. **Multi-Scale Time-Stepping**: Infrasound ($7.83\text{ Hz}$) and acoustics ($438\text{ Hz}$) solved via adaptive sub-cycling; microwave maser solved via envelope population rate equations.
3. **Exact Dimensions**: Primary survey metrics (Cole 1925, Petrie 1883, Gantenbrink 1993) hardcoded into `engine/config.py` and `engine/geometry.py`.
