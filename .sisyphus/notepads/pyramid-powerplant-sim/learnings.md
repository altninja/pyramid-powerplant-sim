# Learnings - Christopher Dunn Giza Power Plant Simulation

## Initialization
- Plan initialized from `.sisyphus/plans/pyramid-powerplant-sim.md`.
- Target: Python simulation engine (`engine/`) + static Three.js replay viewer (`viewer/`).

## Task 1: Environment Setup, Configuration & 3D Spatial Geometry Foundation
- Configured packaging in `pyproject.toml` targeting Python >=3.10 with numpy, scipy, numba, pydantic, and pytest.
- Verified authoritative physical constants in `engine/config.py`:
  - 1 Royal Egyptian Cubit = 0.52360 m
  - Aswan Rose Granite: density = 2650 kg/m³, Young's modulus = 55.0 GPa, sound speed (longitudinal) = 4850 m/s, quartz fraction = 28.5%, eps_r = 6.2, d33_eff = 0.35 pC/N, g33_eff = 0.012 V·m/N, Q = 350.
  - Mokattam Limestone: density = 2450 kg/m³, longitudinal sound speed = 3200 m/s, Young's modulus = 32.0 GPa, attenuation = 0.45 dB/m at 438 Hz.
  - Hydrogen Chemistry: Zn + 2HCl -> ZnCl2 + H2 (dH° = -153.89 kJ/mol, dG° = -147.16 kJ/mol, Ea = 38.5 kJ/mol, k0 = 1.25e4 m/(mol·s)).
  - Gas Dynamics: M_H2 = 2.01588e-3 kg/mol, c_H2(20°C) = 1290.0 m/s, c_air(20°C) = 343.2 m/s.
  - Maser / Microwave: Hydrogen 21cm hyperfine frequency = 1,420,405,751.7667 Hz, A21 = 2.85e-15 s⁻¹, B21 = 5.67e20 m³/(J·s²), shaft waveguide cutoff = 681.35 MHz.
  - Schumann Resonances: f1 = 7.83 Hz, f2 = 14.30 Hz, f3 = 20.80 Hz, f4 = 27.30 Hz.
- Verified 3D spatial models in `engine/geometry.py`:
  - Base Datum (0, 0, 0), Mean Base Side = 230.364 m (440 cubits), Height = 146.580 m (280 cubits), Slope = 51° 50' 40". Total solid volume = 2,593,283 m³.
  - Subterranean Chamber: floor datum -30.00 m, dimensions 14.07 x 8.35 x 3.52 m, survey volume = 280.0 m³, pit depth 3.20 m, blind passage 16.38 m.
  - Queen's Chamber: floor datum +21.20 m, dimensions 5.75 x 5.23 x 6.23 m, survey volume = 160.0 m³, niche height 4.67 m; Northern Shaft (39° 07' 00", length 65.0 m), Southern Shaft (39° 36' 28", length 63.60 m).
  - Grand Gallery: incline length 46.61 m, slope 26° 02' 30", vertical height 8.60 m, base width 2.09 m (4 cubits), roof width 1.05 m (2 cubits), 28 slot pairs (56 total) spaced at 1.68 m.
  - Antechamber: floor datum +43.03 m, dimensions 2.95 x 1.75 x 3.81 m, volume 19.67 m³, granite leaf thickness 0.41 m.
  - King's Chamber: floor datum +43.03 m, dimensions 10.470 x 5.235 x 5.840 m, volume 320.0 m³, 5 relieving tiers with 43 rose granite beams (mean span 6.50 m, mass ~35,000 kg each); Northern Shaft (32° 28' 00", length 71.0 m), Southern Shaft (45° 00' 00", length 53.0 m).
  - Granite Coffer: external 2.278 x 0.977 x 1.048 m, internal 1.977 x 0.677 x 0.872 m, internal volume 1.166 m³ (1,166 L), solid mass ~3,088 kg.
- Passed 17 unit tests in `tests/test_geometry.py` with 0 failures.

## Task 10: 3D Pyramid Geometry & Cutaway View in Three.js
- Implemented `PyramidMesh` and procedural megalithic geometries in `viewer/src/scene/pyramid_mesh.ts`.
- Implemented `CameraController` with smooth orbital controls and 6 animated camera presets.
- Verified TypeScript compilation and Vite packaging.

## Task 11: Dynamic Field Shaders & Particle Visualizers for 3D Replay
- Implemented `FieldRenderers` and modular sub-renderers in `viewer/src/scene/field_renderers.ts`:
  1. `AcousticStandingWaveRenderer`:
     - Multi-layer standing wave acoustic heatmap ribbon and vertical slice along the full Grand Gallery ($L = 46.61\text{ m}$) mapping pressure amplitude $p(z, t)$ from nodal blue through cyan/green to peak antinodal red/orange.
     - 28 pairs of acoustic resonator node emitters with dynamic scale and opacity pulsations.
     - Volumetric 3D acoustic modal standing wave field inside the King's Chamber with synchronized acoustic light modulation.
  2. `HydrogenGasVisualizer`:
     - High-performance 2200-particle system (`THREE.Points`) with custom procedural circular Gaussian shader modeling convective bubbling from Queen's Chamber floor, drift through Horizontal Passage and Ascending Passage, convective updraft along Grand Gallery, and shaft venting.
     - Dynamic particle opacity, size, and velocity modulated by $X_{H2}(t)$ and reaction rate $\dot{n}_{gen}$.
     - Volumetric chemical reaction mist in Queen's Chamber responding to heat release $\dot{Q}_{chem}$.
  3. `PiezoElectricVisualizer`:
     - 43 monolithic rose granite beam high-voltage plasma corona halos across all 5 relieving tiers modulated by $V_{piezo}(t)$ and $\sigma_{max}(t)$.
     - Dielectric spark breakdown flash system at the granite coffer with branching arc discharge line segments and expanding plasma ionization sphere.
  4. `MicrowaveBeamVisualizer`:
     - High-coherence cylindrical/conical beam rays emanating from King's Chamber Northern ($32^\circ 28'$) and Southern ($45^\circ$) shafts through the pyramid casing 260m into the sky.
     - Animated 1.4204 GHz ($21.1\text{ cm}$) traveling phase wavefront rings, laser needle core, and logarithmic power scaling from $P_{beam, N}(t)$ and $P_{beam, S}(t)$ with clean 0-power cutoff.
  5. `HydraulicShockwaveVisualizer`:
     - Concentric expanding seismic pressure wave rings originating at Subterranean Chamber floor datum ($-30.0\text{ m}$) propagating upward through bedrock and limestone matrix modulated by $\Delta P(t)$ and $x_{bed}(t)$.
  6. `FieldRenderers` Master Class & Replay Integration:
     - Zero dynamic memory allocation in 60 FPS `update(frame, dt)` loop.
     - Integrated into `viewer/src/main.ts` with telemetry data playback and cutaway clipping plane synchronization.
- Verification:
  - `cd viewer && npm run build` compiled with 0 TypeScript/Vite errors.
  - `lsp_diagnostics` reported 0 errors across all files.
  - All 105 pytest physics tests passed with 0 failures.

## Task 12: Interactive Replay UI & Real-Time Telemetry Dashboard
- Implemented glassmorphic scientific design system and tokens in `viewer/src/style.css` with dark sci-fi color palette, glowing waveforms, responsive layout, and custom UI components.
- Implemented `TelemetryLoader` (`viewer/src/data/telemetry_loader.ts`):
  - Ingests `sample_telemetry.json` (or any custom drag-and-dropped / uploaded simulation JSON).
  - Normalizes and validates scalar, discrete, and spatial field telemetry.
  - Provides sub-frame cubic Hermite interpolation ($h_{00}, h_{10}, h_{01}, h_{11}$) and linear interpolation between discrete simulation frames for jitter-free 60 FPS replay.
  - Implements global drag-and-drop listener with full-screen dropzone overlay.
- Implemented `TimelineScrubber` (`viewer/src/ui/timeline_scrubber.ts`):
  - Full playback controls: Play, Pause, Step Forward (+1 frame), Step Backward (-1 frame), Reset to start, Loop toggle.
  - Time scrubber slider ($0.0\text{ s}$ to $T_{end}$) with smooth live seeking and timestamp/frame index readouts (`00:01.450 / 00:03.000`, `[Frame 072/150]`).
  - Playback speed multipliers: $0.1\times, 0.25\times, 0.5\times, 1.0\times, 2.0\times, 5.0\times$.
  - Keyboard shortcuts: Space (Play/Pause), Left/Right arrows (Frame step), Home (Reset), L (Loop toggle).
- Implemented `TelemetryPlots` (`viewer/src/ui/telemetry_plots.ts`):
  - High-performance HTML5 Canvas2D live oscilloscopes with HiDPI Retina support and `requestAnimationFrame` throttled rendering:
    1. **Seismic & Hydraulic Infrasound Waveform**: Live scrolling dual trace of Bedrock Displacement $x_{bed}(t)$ ($\mu\text{m}$) and Water Hammer Pressure $\Delta P(t)$ ($\text{MPa}$).
    2. **Acoustic Pressure Waveform**: Live trace of Grand Gallery $p_{GG}(t)$ ($\text{MPa}$) and King's Chamber $p_{KC}(t)$ ($\text{kPa}$).
    3. **FFT Harmonic Spectrum Analyzer**: Live dynamic bar frequency spectrum displaying $7.83\text{ Hz}$ Schumann fundamental, $14.3\text{ Hz}$ harmonic, $438\text{ Hz}$ ($F\#$) fundamental, $876\text{ Hz}$ 2nd harmonic, $1314\text{ Hz}$ 3rd harmonic, and $1.4204\text{ GHz}$ Hydrogen Maser line.
    4. **Piezoelectric Potential & Microwave Output Power**: Live dual-axis trace of Quartz Piezo Voltage $V_{piezo}(t)$ ($\text{kV}$) and Beamed Maser Output Power $P_{maser}(t)$ ($\text{W}$).
  - Multi-view options: 2x2 Quad Oscilloscope grid, single-plot tabbed views, collapse/expand toggle.
- Implemented `ParameterInspector` (`viewer/src/ui/parameter_inspector.ts`):
  - Real-time Physical Telemetry HUD with live metric cards and animated progress meters:
    - Infrasonic Frequency: $7.83\text{ Hz}$, $x_{bed}$, $\Delta P$.
    - Chemical H2 Gas Fraction: $X_{QC}$, $X_{KC}$, $c_{mix}$ ($343.2 \rightarrow 1290\text{ m/s}$), $T_{QC}$.
    - Grand Gallery Resonator: $p_{GG}$, $p_{rms}$, $F\#$ spectral purity ($99.8\%$).
    - King's Chamber Piezoelectric: $V_{total}$, $\sigma_{max}$, spark status.
    - Stimulated Microwave Maser: $1.4204\text{ GHz}$, $P_{beam}$, threshold inversion status.
    - Energy Balance: $P_{in}$ (MW), $P_{out}$ (W), efficiency $\eta$, energy error $\Delta E$.
- Integrated full application stack in `viewer/src/main.ts` and `viewer/index.html` with multi-physics visual layer toggles, cutaway controls, chamber presets, and keyboard shortcuts.
- Verification:
  - `npm run build` in `viewer/` compiles cleanly with 0 errors.
  - `lsp_diagnostics` reports 0 errors across all TypeScript files.
  - Pytest engine tests pass 100% (105 passed).
