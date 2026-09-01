# Work Plan: Giza Power Plant Multi-Physics Simulation Expert Enhancements

## 1. Executive Summary & Audit Findings
This enhancement plan addresses the expert review findings across three core pillars:
1. **Mathematical & Physical Accuracy**:
   - Calibrate the microwave maser electromechanical pumping rate $W_{pump}(V_{piezo}, p_{KC})$ and stimulated emission rate equations with semi-implicit stability, ensuring coherent microwave power scales physically from milliwatts to kilowatts.
   - Close the First Law energy conservation balance to within $< 0.1\%$ relative error across all time-steps by rigorously tracking system stored energy $E_{sys}(t)$ (mechanical, sensible thermal, acoustic field, beam strain/electrostatic, cavity RF) against net integrated boundary power flux.
   - Normalize the Helmholtz resonator spatial source term in the 1D Grand Gallery wave equation by local cross-sectional area $S(z)$ and spatial step $\Delta z$.
   - Couple King's Chamber modal acoustic cavity impedance $Z_L(f) = -j Z_0 \cot(k L_{KC})$ into the Antechamber Transfer Matrix Method (TMM).
   - Implement Voigt-Reuss-Hill quartz crystal aggregate tensor modeling individual fiber stress $\sigma_{xx}$, shear strain $\tau_{xz}$, and dipole polarization $P_z$ across all 43 monolithic rose granite beams.
2. **Comprehensiveness of Data Points**:
   - Expand `SpatialFieldSlice` and `TelemetryFrame` to capture all 43 individual beam stresses and voltages, 3D King's Chamber modal pressure distributions, 5-chamber thermodynamic profiles ($X_{H2}, T, \rho$), displacement current $I_{disp}(t)$, beam capacitance impedance, quantum populations $N_1, N_2$, and shaft Poynting vector flux $S_{em}$ ($W/m^2$).
   - Integrate multi-station pre-computed FFT power spectral density bins ($[f_i, PSD_i]$) into the telemetry stream for real-time spectrum analysis.
3. **Architectural Separation & Standalone Static Browser Replay**:
   - Implement high-performance packed `Float32Array` binary telemetry serialization (`.bin`) with a compact metadata header, reducing load times by $> 90\%$ for 60s+ simulation runs.
   - Embed a Scenario Catalog dropdown in the WebGL UI allowing instant selection and playback of bundled scientific presets (`baseline`, `acoustic_peak`, `full_maser_power`, `dry_run_no_gas`, `high_seismic`).
   - Implement advanced playback controls in `viewer/`: sub-frame Hermite interpolation, reverse playback, frame stepping ($+1/-1$), customizable loop boundaries $[t_a, t_b]$, and event bookmarks ("Reaction Onset", "Resonance Lock", "Piezoelectric Peak", "Maser Emission", "Spark Discharge").
   - Provide batch CLI tooling: `python -m engine.run_sim --all-scenarios --out-dir viewer/public/scenarios/` and automated schema validation.

---

## 2. System Architecture & Enhanced Component Map

```
pyramid-sim/
├── engine/                               # Enhanced Python Multi-Physics Simulation Engine
│   ├── config.py                         # Calibrated maser rates, aggregate piezo tensors, cavity Q
│   ├── geometry.py                       # 3D spatial node coordinate matrices & mesh slices
│   ├── physics/
│   │   ├── schumann_hydraulics.py        # Subterranean water hammer & multi-harmonic bedrock drive
│   │   ├── chemical_gas_transport.py     # Zn + 2HCl kinetics, 5-chamber advection-diffusion, thermal heat
│   │   ├── grand_gallery_acoustics.py    # Area-normalized 1D wave solver & 28-pair F# Helmholtz array
│   │   ├── antechamber_filter.py         # Reactive-cavity loaded TMM acoustic gate & filter
│   │   ├── piezoelectric_beams.py        # 43 individual beam modal dynamics, shear/bending stress, piezo dipoles
│   │   ├── microwave_maser.py            # Calibrated semi-implicit rate equations & horn waveguide beaming
│   │   └── energy_accountant.py          # Exact closed-loop First Law energy conservation validator
│   ├── orchestrator.py                   # Multi-rate time-stepping orchestrator (dt_macro = 10ms, dt_micro = 0.1ms)
│   ├── telemetry.py                      # Multi-format telemetry exporter (JSON, Gzip, Binary .bin)
│   └── run_sim.py                        # CLI runner with batch scenario generation & scenario manifest
├── viewer/                               # Standalone Static 3D WebGL Replay Dashboard
│   ├── index.html                        # Glassmorphic UI with scenario selector, event bookmarks, HUD
│   ├── public/
│   │   ├── sample_telemetry.json         # Default telemetry file
│   │   └── scenarios/                    # Bundled multi-scenario directory
│   │       ├── manifest.json             # Scenario metadata manifest
│   │       ├── baseline.json / .bin
│   │       ├── acoustic_peak.json / .bin
│   │       ├── full_maser_power.json / .bin
│   │       ├── dry_run_no_gas.json / .bin
│   │       └── high_seismic.json / .bin
│   └── src/
│       ├── main.ts                       # App coordinator & event bookmark manager
│       ├── data/
│       │   ├── telemetry_loader.ts       # Unified JSON & Binary .bin parser with Hermite interpolation
│       │   └── scenario_manifest.ts      # Scenario catalog loader & switcher
│       ├── scene/
│       │   ├── pyramid_mesh.ts           # 3D Khufu cutaway geometry & 43 individual beam meshes
│       │   ├── camera_controller.ts      # Animated camera presets & event focus targets
│       │   └── field_renderers.ts        # GLSL standing wave heatmaps, gas particles, piezo glow, maser rays
│       └── ui/
│           ├── timeline_scrubber.ts      # Scrub bar, reverse playback, speed selector, loop bounds
│           ├── telemetry_plots.ts        # Dynamic Canvas2D oscilloscopes & multi-station FFT spectrum
│           └── parameter_inspector.ts    # Comprehensive multi-domain physical telemetry HUD
├── tests/                                # Extended Automated Test Suite
│   ├── test_energy_conservation.py       # Strict First Law closure (< 0.1% error)
│   ├── test_microwave_maser.py           # Calibrated pumping & threshold validation
│   ├── test_binary_telemetry.py          # Binary serialization & deserialization verification
│   └── test_cli_runner.py                # Batch scenario generator & manifest tests
└── README.md                             # Comprehensive scientific documentation & user manual
```

---

## 3. Mathematical & Numerical Formulations

### 3.1 Closed-Loop First Law Energy Conservation
The total instantaneous energy stored in the pyramid system $E_{sys}(t)$ is:
$$E_{sys}(t) = E_{bedrock}(t) + E_{gas\_thermal}(t) + E_{acoustic\_GG}(t) + E_{beams}(t) + E_{maser\_cav}(t)$$
where:
- $E_{bedrock}(t) = \frac{1}{2} M_{bedrock} \dot{x}_{bed}^2 + \frac{1}{2} K_{bedrock} x_{bed}^2$
- $E_{gas\_thermal}(t) = \sum_{i=1}^5 n_i C_{v, mix, i} (T_i - T_{ref})$
- $E_{acoustic\_GG}(t) = \int_0^L \left( \frac{p^2(z, t)}{2 \rho c^2} + \frac{\rho v^2(z, t)}{2} \right) S(z) dz + \sum_{m=1}^{28} \left( \frac{1}{2} M_{a, m} \dot{U}_m^2 + \frac{1}{2 C_{a, m}} U_m^2 \right)$
- $E_{beams}(t) = \sum_{b=1}^{43} \sum_{n=1}^4 \left( \frac{1}{2} M_{modal, n} \dot{q}_{n, b}^2 + \frac{1}{2} K_{modal, n} q_{n, b}^2 \right) + \frac{1}{2} C_{total} V_{total}^2$
- $E_{maser\_cav}(t) = \rho_{em}(t) V_{KC}$

The continuous energy conservation condition requires:
$$E_{sys}(t) - E_{sys}(0) = \int_0^t \left( P_{in}(\tau) - P_{out}(\tau) - P_{loss}(\tau) \right) d\tau$$
with relative residual:
$$\text{RelError}(t) = \frac{\left| E_{sys}(t) - E_{sys}(0) - \int_0^t (P_{in} - P_{out} - P_{loss}) d\tau \right|}{\max(E_{scale}, E_{sys}(t))} < 10^{-3} \quad (0.1\%)$$

### 3.2 Calibrated Maser Stimulated Emission Rate Equations
The atomic hydrogen hyperfine 2-level population densities $N_1, N_2$ ($m^{-3}$) and radiation energy density $\rho_{em}$ ($J/m^3$) evolve via:
$$\frac{dN_2}{dt} = W_{pump}(V_{piezo}, p_{KC}, X_{H2}) N_1 - B_{21} \rho_{em} (N_2 - N_1) - A_{21} N_2 - \frac{N_2 - N_{2, eq}}{\tau_{coll}}$$
$$\frac{d\rho_{em}}{dt} = h \nu_{21} B_{21} \rho_{em} (N_2 - N_1) + h \nu_{21} A_{21} N_2 \eta_{geom} - \frac{\rho_{em}}{\tau_{cav}} - \frac{P_{shafts}}{V_{KC}}$$
with calibrated electromechanical pumping:
$$W_{pump} = X_{H2} \cdot \left[ \kappa_{elec} \left(\frac{V_{total}}{V_{ref}}\right)^2 + \kappa_{acoust} \left(\frac{p_{KC}}{p_{ref}}\right) \right]$$
Stimulated emission power radiated into the waveguide shafts is:
$$P_{stim}(t) = h \nu_{21} B_{21} \rho_{em}(t) \max(0, N_2(t) - N_1(t)) V_{KC}$$
$$P_{shafts}(t) = \eta_{coupling} \cdot P_{stim}(t)$$

### 3.3 Grand Gallery Area-Normalized Wave Equation
Discretized 1D acoustic pressure wave equation with resonator feedback normalized by local duct area $S(z)$:
$$\frac{1}{c^2(z)} \frac{\partial^2 p}{\partial t^2} - \frac{\partial^2 p}{\partial z^2} - \frac{1}{S(z)} \frac{dS}{dz} \frac{\partial p}{\partial z} + \frac{2\alpha}{c(z)} \frac{\partial p}{\partial t} = \sum_{m=1}^{28} \frac{\delta(z - z_m)}{S(z_m) \Delta z} \rho \frac{\partial^2 U_m}{\partial t^2}$$

### 3.4 Antechamber TMM Reactive Cavity Load
The Antechamber input-output acoustic pressure/velocity relation is:
$$\begin{bmatrix} p_{in}(f) \\ U_{in}(f) \end{bmatrix} = \mathbf{M}_{total}(f) \begin{bmatrix} p_{out}(f) \\ U_{out}(f) \end{bmatrix}$$
with King's Chamber reactive termination impedance:
$$Z_L(f) = -j \frac{\rho c}{S_{portal}} \cot\left( \frac{2\pi f}{c} L_{KC} \right)$$
giving transmission coefficient:
$$T(f) = \frac{2 Z_L}{M_{11} Z_L + M_{12} + M_{21} Z_0 Z_L + M_{22} Z_0}$$

### 3.5 43-Beam Individual Stress & Piezoelectric Tensors
For each monolithic granite beam $b \in \{1 \dots 43\}$ in tier $k \in \{1 \dots 5\}$:
$$\sigma_{xx, b}(x, z_f, t) = -E_{gr} z_f \sum_{n=1}^4 q_{n, b}(t) \frac{d^2 \phi_n}{dx^2}$$
$$\tau_{xz, b}(x, t) = -E_{gr} \frac{h^2 - 4 z_f^2}{8} \sum_{n=1}^4 q_{n, b}(t) \frac{d^3 \phi_n}{dx^3}$$
$$P_{z, b}(t) = d_{33}^{eff} \bar{\sigma}_{xx, b}(t) + d_{14}^{eff} \bar{\tau}_{xz, b}(t)$$
$$V_b(t) = \frac{g_{33}^{eff} \bar{\sigma}_{xx, b}(t) h}{\varepsilon_r \varepsilon_0}, \quad I_{disp, b}(t) = C_b \frac{dV_b}{dt}$$

---

## 4. Implementation Tasks

### - [x] Task 1: Microwave Maser Pumping Calibration & Semi-Implicit Rate Equation Solver
- **Files**: `engine/physics/microwave_maser.py`, `tests/test_microwave_maser.py`
- **Objective**: Calibrate the electromechanical pumping coupling and implement a robust semi-implicit/analytical rate equation solver ensuring stimulated microwave power scales physically from milliwatts to kilowatts without numerical sub-threshold noise or stiffness divergence.
- **Details**:
  - Re-parameterize pumping rate equation:
    $$W_{pump} = X_{H2} \cdot \left[ \kappa_{elec} \left(\frac{V_{total}}{V_{ref}}\right)^2 + \kappa_{acoust} \left(\frac{p_{KC}}{p_{ref}}\right) \right]$$
    with calibrated thresholds $V_{ref} = 5.0\text{ kV}, p_{ref} = 100\text{ kPa}, \kappa_{elec} = 50.0\text{ s}^{-1}, \kappa_{acoust} = 10.0\text{ s}^{-1}$.
  - Implement semi-implicit / analytical sub-stepping for population inversion $\Delta N = N_2 - N_1$ and photon energy density $\rho_{em}$:
    - Enforce exact saturation clamping at threshold $\Delta N_{sat} = \frac{1/\tau_{cav} + P_{shafts}/(V_{KC} \rho_{em})}{h \nu_{21} B_{21}}$ during steady-state emission.
    - Ensure stimulated emission power $P_{stim}(t) = h \nu_{21} B_{21} \rho_{em}(t) \max(0, \Delta N) V_{KC}$ scales smoothly from 0 to $> 1\text{ kW}$ under high acoustic-piezoelectric drive.
  - Calculate directional Effective Radiated Power (ERP) for Northern ($32^\circ 28'$) and Southern ($45^\circ$) shafts including horn aperture directivity gain ($G_0 \approx 9.8\text{ dBi}$).
- **QA Scenarios**:
  - Test maser output under baseline resonant drive: verify emitted power is in the physical range ($1\text{ W} \le P_{beam} \le 5\text{ kW}$).
  - Test sub-threshold control: verify that with zero hydrogen ($X_{H2}=0$) or sub-threshold voltage ($V < 100\text{ V}$), stimulated emission remains strictly $0.0\text{ W}$.
  - Test numerical stability: step for $10\text{ s}$ with $\Delta t = 0.01\text{ s}$ under peak drive without stiffness oscillations or negative populations.

### - [x] Task 2: Exact Closed-Loop First Law Energy Conservation Balance
- **Files**: `engine/physics/energy_accountant.py`, `engine/orchestrator.py`, `tests/test_energy_conservation.py`
- **Objective**: Establish closed-loop First Law energy accounting that tracks total instantaneous stored system energy $E_{sys}(t)$ and boundary power fluxes, achieving relative energy balance closure $< 0.1\%$ across all simulation steps.
- **Details**:
  - In `engine/physics/energy_accountant.py`:
    - Implement `compute_total_stored_energy()` summing:
      1. Bedrock kinetic + potential energy: $E_{bedrock} = \frac{1}{2} M \dot{x}^2 + \frac{1}{2} K x^2$.
      2. Chamber sensible thermal energy: $E_{thermal} = \sum n_i C_v (T_i - T_{ref})$.
      3. Acoustic field stored energy: $E_{acoustic} = \int \left(\frac{p^2}{2\rho c^2} + \frac{\rho v^2}{2}\right) S dz + \sum E_{res, m}$.
      4. 43-beam kinetic, elastic strain, and electrostatic energy: $E_{beams} = \sum (\frac{1}{2} M_n \dot{q}^2 + \frac{1}{2} K_n q^2) + \frac{1}{2} C_{total} V_{total}^2$.
      5. Microwave cavity RF energy: $E_{maser} = \rho_{em} V_{KC}$.
    - Track net integrated power: $W_{net}(t) = \int_0^t (P_{in}(\tau) - P_{out}(\tau) - P_{loss}(\tau)) d\tau$.
    - Calculate relative energy error:
      $$\text{RelError}(t) = \frac{|E_{sys}(t) - E_{sys}(0) - W_{net}(t)|}{\max(E_{scale}, E_{sys}(t))}$$
  - In `engine/orchestrator.py`:
    - Baseline initial state $E_{sys}(0)$ before time-stepping starts.
    - Accurately route chemical reaction enthalpy $\dot{n}_{H2} (-\Delta H_{rxn})$ directly into sensible heat and acoustic/piezoelectric work without double-counting unreacted chemical potential.
- **QA Scenarios**:
  - Test First Law closure over $10.0\text{ s}$ baseline run: verify $\text{RelError}(t) < 10^{-3}$ ($0.1\%$) at every recorded frame.
  - Verify `all_steps_conserved` flag in summary telemetry evaluates to `True`.

### - [x] Task 3: Grand Gallery Area-Normalized Wave Equation & Antechamber Reactive Cavity Load
- **Files**: `engine/physics/grand_gallery_acoustics.py`, `engine/physics/antechamber_filter.py`, `tests/test_acoustics_and_resonators.py`, `tests/test_antechamber_filter.py`
- **Objective**: Normalize the acoustic resonator coupling by local gallery cross-sectional area and connect King's Chamber modal reactive cavity impedance to the Antechamber Transfer Matrix.
- **Details**:
  - In `engine/physics/grand_gallery_acoustics.py`:
    - Compute local cross-sectional area $S(z) = W(z) \cdot H(z)$ accounting for the 7-step corbel narrowing ($2.09\text{ m}$ base to $1.05\text{ m}$ roof).
    - Update spatial feedback source term:
      $$S_{res}(z, t) = \sum_{m=1}^{28} \frac{w_m(z)}{S(z_m) \Delta z} \rho \ddot{U}_m(t)$$
      where $w_m(z)$ is a Gaussian/tent spatial interpolation kernel.
  - In `engine/physics/antechamber_filter.py`:
    - Implement frequency-dependent reactive load impedance for King's Chamber:
      $$Z_L(f) = -j \frac{\rho c}{S_{portal}} \cot\left(\frac{2\pi f}{c} L_{KC}\right)$$
    - Integrate $Z_L(f)$ into the cascade Transfer Matrix transmission coefficient $T(f)$ and acoustic gating calculation.
- **QA Scenarios**:
  - Test acoustic wave amplitude scaling with varying cross-sectional area: verify smooth pressure wave propagation without artificial boundary reflections.
  - Test Antechamber transmission loss curve: verify passband resonance matches King's Chamber longitudinal acoustic modes ($438\text{ Hz}$ fundamental).

### - [x] Task 4: 43-Beam Individual Modal Dynamics, Stress-Strain Tensors & Displacement Current
- **Files**: `engine/physics/piezoelectric_beams.py`, `tests/test_piezoelectric_transduction.py`
- **Objective**: Extend the piezoelectric beam module to calculate individual bending stress $\sigma_{xx, b}(t)$, shear stress $\tau_{xz, b}(t)$, displacement current $I_{disp, b}(t)$, and aggregate quartz polarization $P_{z, b}(t)$ for all 43 monolithic rose granite beams across the 5 relieving tiers.
- **Details**:
  - Implement full 43-beam state array in `PiezoelectricBeams`:
    - Davison's Chamber (Tier 1): Beams 1–9
    - Wellington's Chamber (Tier 2): Beams 10–18
    - Nelson's Chamber (Tier 3): Beams 19–27
    - Lady Arbuthnot's Chamber (Tier 4): Beams 28–36
    - Campbell's Chamber (Tier 5): Beams 37–43
  - Compute spatial standing wave pressure at the exact long-axis position $y_b$ of each beam inside King's Chamber ($y \in [0.5, 10.0]\text{ m}$).
  - Calculate individual fiber stress:
    $$\sigma_{max, b}(t) = E_{gr} \frac{h}{2} \left| \sum_{n=1}^4 q_{n, b}(t) \frac{d^2 \phi_n}{dx^2}(L/2) \right|$$
  - Calculate displacement current and impedance for each tier:
    $$I_{disp, b}(t) = C_b \frac{dV_b}{dt}, \quad Z_{tier}(f) = \frac{1}{j 2\pi f \sum C_b}$$
  - Export full 43-beam stress array (`beam_stresses_mpa: List[float]`), 43-beam voltage array (`beam_voltages_v: List[float]`), and total displacement current $I_{disp, total}(t)$ into `PiezoelectricState`.
- **QA Scenarios**:
  - Verify individual beam stresses vary with spatial position $y_b$ in response to King's Chamber longitudinal acoustic standing waves.
  - Verify displacement current $I_{disp}(t)$ satisfies $I = C \frac{dV}{dt}$ within $\pm 0.5\%$.
  - Verify all 43 beams are individually addressable in telemetry frames.

### - [x] Task 5: Extended Telemetry Data Model & Multi-Station FFT Spectral Bins
- **Files**: `engine/telemetry.py`, `engine/orchestrator.py`, `tests/test_telemetry_export.py`
- **Objective**: Extend the telemetry frame structure and spatial field slices to capture complete physical diagnostics (43-beam stresses, 5-chamber thermodynamics, FFT power spectral density bins, displacement current, Poynting flux, and quantum populations).
- **Details**:
  - Update `TelemetryFrame` in `engine/telemetry.py`:
    - Add `displacement_current_a: float`
    - Add `beam_array_impedance_ohms: float`
    - Add `chamber_temperatures_k: List[float]` (5 nodes)
    - Add `chamber_pressures_pa: List[float]` (5 nodes)
    - Add `maser_state_populations: Dict[str, float]` ($N_1, N_2, N_{total}$)
    - Add `shaft_poynting_flux_w_m2: List[float]` (North, South)
    - Add `effective_radiated_power_w: float`
  - Update `SpatialFieldSlice` in `engine/telemetry.py`:
    - Add `all_beam_stresses_mpa: List[float]` (43 values)
    - Add `all_beam_voltages_v: List[float]` (43 values)
    - Add `fft_frequencies_hz: List[float]` (e.g. 128 bins from 1 Hz to 2000 Hz)
    - Add `fft_power_spectral_density_db: List[float]` (spectral power bins at monitoring stations)
  - In `engine/orchestrator.py`:
    - Compute moving FFT spectrum of bedrock acceleration, Grand Gallery pressure, and shaft microwave output.
- **QA Scenarios**:
  - Test telemetry frame serialization: verify all new fields are populated and correctly typed.
  - Test FFT spectral bins: verify peaks at $7.83\text{ Hz}$ and $438\text{ Hz}$ are cleanly resolved in the exported spectrum array.

### - [x] Task 6: Compact High-Performance Binary Telemetry Serialization (.bin format)
- **Files**: `engine/telemetry.py`, `engine/orchestrator.py`, `tests/test_binary_telemetry.py`
- **Objective**: Implement a high-efficiency packed binary telemetry serialization format (`.bin`) that serializes simulation frames into contiguous `Float32Array` buffers with a compact JSON header, reducing file size by $> 80\%$ and loading times to $< 50\text{ ms}$.
- **Details**:
  - Define Binary Layout:
    - Header (JSON string prefixed with 4-byte length integer): Contains metadata, frame count, channel names, sampling rate, spatial grid coordinates, and channel offsets.
    - Payload (Raw binary floats / `Float32Array`): Structured columnar or interleaved float32 buffers storing all scalar channels and spatial slices across all frames.
  - Implement `export_binary(path: Path)` in `TelemetryExporter`:
    - Writes `.bin` file containing the binary packed data.
  - Implement `load_binary(path: Path)` in Python to verify round-trip binary deserialization.
  - Benchmark performance: Demonstrate that a 10s (600 frames) binary dataset loads and parses in $< 20\text{ ms}$.
- **QA Scenarios**:
  - Round-trip accuracy test: verify deserialized binary floats match original Python simulation state within $10^{-6}$ single-precision float tolerance.
  - File size reduction test: verify binary `.bin` file is at least $70\%$ smaller than uncompressed `.json`.

### - [x] Task 7: Static Viewer Unified Telemetry Loader (JSON & Binary .bin) with Hermite Interpolation
- **Files**: `viewer/src/data/telemetry_loader.ts`, `viewer/src/data/scenario_manifest.ts`
- **Objective**: Implement a unified WebGL telemetry loader capable of transparently loading both `.bin` binary ArrayBuffers and `.json` datasets with cubic Hermite sub-frame interpolation and scenario catalog metadata loading.
- **Details**:
  - Implement `loadBinaryTelemetry(buffer: ArrayBuffer)` in `viewer/src/data/telemetry_loader.ts`:
    - Reads 4-byte header length, parses JSON metadata header, and views binary payload as `Float32Array` or `DataView`.
    - Creates contiguous memory views for all time-series channels and spatial profiles.
  - Implement unified `TelemetryLoader.load(urlOrFile: string | File)`:
    - Auto-detects `.bin`, `.json`, or `.json.gz` formats.
  - Implement `viewer/src/data/scenario_manifest.ts`:
    - Defines `ScenarioManifest` interface loading `scenarios/manifest.json`.
    - Lists all available pre-computed scenarios (`baseline`, `acoustic_peak`, `full_maser_power`, `dry_run_no_gas`, `high_seismic`) with scenario titles, descriptions, and file URLs.
  - Sub-frame Cubic Hermite Interpolator:
    - Interpolates scalar values $f(t)$ and spatial arrays $A(z, t)$ at arbitrary fractional times $t \in [t_k, t_{k+1}]$ using finite-difference tangent estimates for continuous 60 FPS / 120 FPS playback.
- **QA Scenarios**:
  - Test binary vs JSON parity in viewer: verify both loaders return identical frame data at arbitrary scrub positions.
  - Test drag-and-drop: dropping a custom `.bin` or `.json` file instantly loads and renders without reloading the page.

### - [x] Task 8: Advanced Replay Controls: Scenario Catalog Selector, Event Bookmarks & Range Looping
- **Files**: `viewer/src/ui/timeline_scrubber.ts`, `viewer/src/ui/parameter_inspector.ts`, `viewer/src/ui/telemetry_plots.ts`, `viewer/src/scene/pyramid_mesh.ts`, `viewer/src/main.ts`, `viewer/index.html`
- **Objective**: Enhance the WebGL viewer UI with a scenario preset dropdown switcher, physical event bookmarks, range loop controls, 43-beam visual inspection HUD, and multi-station FFT display.
- **Details**:
  - In `viewer/index.html` & `viewer/src/main.ts`:
    - Add Scenario Selector dropdown in top navigation bar allowing instant switching between bundled scenarios.
    - Add Event Bookmark pills on the timeline:
      - "Reaction Start" ($t \approx 0.1\text{ s}$)
      - "Acoustic Lock" ($t \approx 1.0\text{ s}$)
      - "Piezoelectric Peak" ($t \approx 1.5\text{ s}$)
      - "Maser Emission" ($t \approx 2.0\text{ s}$)
      - "Spark Discharge" (timestamp of breakdown events)
    - Clicking a bookmark animates the camera to the relevant chamber and jumps timeline playback to the event timestamp.
  - In `viewer/src/ui/timeline_scrubber.ts`:
    - Add reverse playback toggle (plays backward from current timestamp).
    - Add customizable loop boundary sliders $[t_A, t_B]$.
    - Add keyboard shortcuts: Space (Play/Pause), Left/Right Arrow (Step -1/+1 frame), J/K/L (Reverse / Pause / Fast-Forward), R (Reset).
  - In `viewer/src/ui/parameter_inspector.ts`:
    - Add an interactive 43-Beam Grid Inspector showing individual stresses and voltages for all 43 granite beams organized by the 5 relieving tiers.
  - In `viewer/src/ui/telemetry_plots.ts`:
    - Render true FFT power spectral density curves using the telemetry frequency bins.
- **QA Scenarios**:
  - Verify switching scenarios in the dropdown loads the new dataset and resets playback seamlessly.
  - Verify clicking event bookmarks jumps to correct timestamps and camera viewpoints.
  - Verify 43-beam inspector updates all 43 beam cells in real-time.

### - [x] Task 9: Batch CLI Scenario Generator & Standalone Static Hosting Suite
- **Files**: `engine/run_sim.py`, `tests/test_cli_runner.py`, `README.md`
- **Objective**: Provide batch CLI scenario generation commands producing both binary `.bin` and JSON formats alongside `manifest.json`, ensuring the static viewer is 100% standalone and ready for zero-backend static deployment.
- **Details**:
  - In `engine/run_sim.py`:
    - Add `--all-scenarios` CLI flag:
      - Runs `baseline`, `acoustic_peak`, `full_maser_power`, `dry_run_no_gas`, `high_seismic`.
      - Exports each scenario in both `.json` and `.bin` format to `viewer/public/scenarios/`.
      - Generates `viewer/public/scenarios/manifest.json` cataloging all runs.
    - Add `--validate-schema` flag verifying exported files conform to the telemetry schema.
  - In `README.md`:
    - Document the batch generator, binary format specifications, static deployment instructions (GitHub Pages / Vercel / Netlify / S3), and physical models.
- **QA Scenarios**:
  - Run `python -m engine.run_sim --all-scenarios --duration 3.0` and verify all 5 scenario `.json` and `.bin` files and `manifest.json` are created.
  - Run `npm --prefix viewer run build` and verify production build bundles all static assets cleanly.
  - Run full test suite: `pytest tests/ -v` verifying all existing and new tests pass.

## Final Verification Wave (MANDATORY)
- [x] Run full automated test suite: `pytest tests/ -v` ensuring all 132 tests pass with 0 failures.
- [x] Verify First Law energy conservation error is $< 0.1\%$ across all standard scenarios.
- [x] Run batch scenario generation: `python -m engine.run_sim --all-scenarios --out-dir viewer/public/scenarios/`.
- [x] Verify scenario manifest `viewer/public/scenarios/manifest.json` and generated binary `.bin` & JSON files.
- [x] Build static viewer: `cd viewer && npm run build` ensuring 0 TypeScript/Vite build errors.
- [x] Verify static viewer in local preview: `npm run preview` testing scenario selector, timeline scrubbing, and 43-beam inspector.
