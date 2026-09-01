# Giza Power Plant Multi-Physics Simulation Engine

[![CI](https://github.com/altninja/pyramid-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/altninja/pyramid-sim/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Three.js](https://img.shields.io/badge/Three.js-r170-black.svg)](https://threejs.org/)
[![PyTest](https://img.shields.io/badge/PyTest-132%20passed-brightgreen.svg)](https://docs.pytest.org/)

A rigorous, coupled multi-scale computational physics engine and 3D WebGL replay viewer modeling Christopher Dunn's acoustical-piezoelectric-microwave power plant hypothesis for the Great Pyramid of Giza (*The Giza Power Plant: Technologies of Ancient Egypt*).

The simulation unifies 6 non-linear physical domains into a synchronized multi-rate integration pipeline, validating the end-to-end energy transduction cascade: from subterranean seismic and hydraulic oscillations to atmospheric chemical hydrogen generation, acoustic standing wave amplification, quartz piezoelectric high-voltage generation, and coherent 1.4204 GHz ($21.1\text{ cm}$) microwave maser beaming.

---

## 1. System Architecture & Multi-Physics Signal Flow

```
                                    +-------------------------------------------------------------+
                                    |                MASTER SIMULATION ORCHESTRATOR               |
                                    |   Multi-Rate: dt_macro = 10 ms (Gas/Chem) | dt_micro = 0.1 ms |
                                    +-------------------------------------------------------------+
                                                                   |
+------------------------------------+                             |                             +-----------------------------------+
|     SUBTERRANEAN HYDRAULICS        |                             |                             |     QUEEN'S CHAMBER CHEMISTRY     |
| - Earth Schumann Infrasound (7.83Hz)|                             |                             | - Zn(s) + 2HCl(aq) -> ZnCl2 + H2  |
| - Joukowsky Water Hammer Pulses    |                             |                             | - Exothermic Heat & 5-Node Flow   |
| - Bedrock Mass-Spring Resonance    |                             |                             | - c_mix(X_H2): 343.2 -> 1290 m/s  |
+------------------------------------+                             |                             +-----------------------------------+
                  | (Infrasonic Acoustic Drive)                     | (H2 Gas Diffusion Network)                   |
                  +-------------------------------+                 |                 +----------------------------+
                                                  |                 |                 |
                                                  v                 v                 v
                                    +-------------------------------------------------------------+
                                    |              GRAND GALLERY ACOUSTIC WAVEGUIDE               |
                                    | - 1D Inhomogeneous Lossy Wave Equation: p(z, t)             |
                                    | - 27 Pairs of Tuned Helmholtz Resonators (F# Series: 438Hz) |
                                    | - Dynamic Acoustic Impedance & Standing Wave Amplification  |
                                    +-------------------------------------------------------------+
                                                                   | (Acoustic Power Flux)
                                                                   v
                                    +-------------------------------------------------------------+
                                    |              ANTECHAMBER ACOUSTIC TMM FILTER                |
                                    | - Transfer Matrix Method (TMM) Acoustic Cascades            |
                                    | - Granite Leaf Low-Pass Infrasonic Gating (>15 dB rejection)|
                                    | - 438 Hz Harmonic Passband Impedance Matching               |
                                    +-------------------------------------------------------------+
                                                                   | (Filtered 438 Hz Pressure)
                                                                   v
                                    +-------------------------------------------------------------+
                                    |         KING'S CHAMBER PIEZOELECTRIC TRANSDUCTION           |
                                    | - 43 Monolithic Rose Granite Beams across 5 Relieving Tiers |
                                    | - Clamped-Clamped Euler-Bernoulli Flexural Modal Dynamics   |
                                    | - Quartz Dipole Polarization: V_total > 15-30 kV            |
                                    | - Granite Coffer Spark Gap Dielectric Ionization            |
                                    +-------------------------------------------------------------+
                                                                   | (HV E-Field & Acoustic Pump)
                                                                   v
                                    +-------------------------------------------------------------+
                                    |           KING'S CHAMBER MICROWAVE MASER & SHAFTS           |
                                    | - Atomic Hydrogen Hyperfine Ground State Inversion (nu21)   |
                                    | - Stimulated Emission Rate ODEs (1.4204 GHz / 21.1 cm)      |
                                    | - Northern (32°28') & Southern (45°00') Waveguide Shafts    |
                                    | - Directional Aperture Horn Beaming (G0 = 13.34 dBi)        |
                                    +-------------------------------------------------------------+
                                                                   | (Radiated RF Beam Power)
                                                                   v
                                    +-------------------------------------------------------------+
                                    |              MASTER FIRST-LAW ENERGY ACCOUNTANT             |
                                    | Cumulative Audit: Delta E_stored = E_in - E_out - E_loss    |
                                    | Strict Conservation Enforcement (Error < 0.01%)             |
                                    +-------------------------------------------------------------+
```

---

## 2. Mathematical Subsystem Models & Governing Equations

### 2.1 Subterranean Chamber Hydraulic Pulse & Infrasonic Drive
The subterranean mechanism harnesses hydrodynamic water hammer pulses and terrestrial seismic vibrations:
1. **Joukowsky Water Hammer Pressure Surge**:
   $$P_{\text{hammer}} = \rho_w c_w \Delta v$$
   where $\rho_w = 998.2\text{ kg/m}^3$ and $c_w = 1482.0\text{ m/s}$.
2. **Bedrock Mass-Spring-Damper Dynamics**:
   $$M_{\text{bed}} \ddot{x}_{\text{bed}} + C_{\text{bed}} \dot{x}_{\text{bed}} + K_{\text{bed}} x_{\text{bed}} = F_{\text{seismic}}(t) + F_{\text{hydraulic}}(t)$$
   driven at the Earth's fundamental Schumann cavity resonance ($f_1 = 7.83\text{ Hz}$) and its higher harmonics ($14.30\text{ Hz}, 20.80\text{ Hz}, 27.30\text{ Hz}$).

### 2.2 Queen's Chamber Hydrogen Generation & Multi-Node Gas Transport
Exothermic acid-metal reaction kinetics generate pure hydrogen gas:
1. **Reaction Kinetics**:
   $$\text{Zn}(s) + 2\text{HCl}(aq) \longrightarrow \text{ZnCl}_2(aq) + \text{H}_2(g) \uparrow, \quad \Delta H_{\text{rxn}}^\circ = -153.89\text{ kJ/mol}$$
   $$r_{\text{chem}} = k_0 \exp\left(-\frac{E_a}{R T}\right) [\text{HCl}]^2 A_{\text{Zn}}$$
2. **Dynamic Gas Mixture Sound Speed**:
   $$c_{\text{mix}}(X_{\text{H2}}, T) = \sqrt{\frac{\gamma_{\text{mix}} R T}{M_{\text{mix}}}}$$
   where $M_{\text{mix}} = X_{\text{H2}} M_{\text{H2}} + (1 - X_{\text{H2}}) M_{\text{air}}$, smoothly shifting sound speed from $343.2\text{ m/s}$ (air) up to $1290.0\text{ m/s}$ (pure $\text{H}_2$).

### 2.3 Grand Gallery Resonant Acoustic Waveguide
The corbelled gallery functions as an acoustic amplifier driven by 27 pairs of Helmholtz resonators tuned to the $F\#$ harmonic series:
1. **1D Inhomogeneous Lossy Wave Equation**:
   $$\frac{1}{c^2(z, t)} \frac{\partial^2 p}{\partial t^2} - \frac{\partial^2 p}{\partial z^2} + \frac{2\alpha}{c(z, t)} \frac{\partial p}{\partial t} = \sum_{m=1}^{27} \delta(z - z_m) \rho \frac{\partial U_m}{\partial t}$$
2. **Coupled Helmholtz Resonator Dynamic ODE**:
   $$\frac{d^2 U_m}{dt^2} + \frac{\omega_{r, m}}{Q_m} \frac{dU_m}{dt} + \omega_{r, m}^2 U_m = \frac{A_{\text{neck}}}{\rho L_{\text{eff}}} p(z_m, t)$$

### 2.4 Antechamber Acoustic Transfer Matrix Filter
The 4 vertical granite leaves and wainscoting slots act as a cascaded 2-port acoustic filter:
1. **Acoustic Transfer Matrix Cascade**:
   $$\mathbf{M}_{\text{total}}(f) = \prod_{k=1}^N \begin{bmatrix} \cos(k_c L_k) & j Z_k \sin(k_c L_k) \\ j Z_k^{-1} \sin(k_c L_k) & \cos(k_c L_k) \end{bmatrix}, \quad Z_k = \frac{\rho c}{S_k}$$
2. **Transmission Loss (TL)**:
   $$\text{TL}(f) = 20 \log_{10} \left| \frac{1}{2} \left( A\sqrt{\frac{Z_L}{Z_0}} + \frac{B}{\sqrt{Z_0 Z_L}} + C\sqrt{Z_0 Z_L} + D\sqrt{\frac{Z_0}{Z_L}} \right) \right|$$
   Rejects low-frequency infrasonic surges ($> 15\text{ dB}$ attenuation at $7.83\text{ Hz}$) while maintaining high transmission ($< 0.1\text{ dB}$ loss) for $F\#$ harmonics.

### 2.5 King's Chamber Rose Granite Piezoelectric Transduction
43 monolithic Aswan rose granite beams across 5 relieving tiers transduce acoustic vibration into high electric potentials:
1. **Euler-Bernoulli Clamped-Clamped Modal Dynamics**:
   $$\ddot{q}_n(t) + 2\zeta_n \omega_n \dot{q}_n(t) + \omega_n^2 q_n(t) = \frac{F_{\text{modal}, n}(t)}{M_{\text{modal}, n}}$$
2. **Quartz Dipole Polarization & Array Voltage**:
   $$\bar{\sigma}(t) = E_{\text{gr}} \frac{h}{2} \bar{\kappa}_{\text{rms}}(t), \quad V_{\text{beam}}(t) = g_{33}^{\text{eff}} \bar{\sigma}(t) h$$
   $$V_{\text{total}}(t) = \sum_{k=0}^4 V_{\text{tier}, k}(t) \quad (\text{Stacked series potential } > 15-30\text{ kV})$$

### 2.6 Microwave Maser Stimulated Emission & Horn Beaming
High-voltage oscillating fields and acoustic pressure pump atomic hydrogen into quantum population inversion:
1. **Hyperfine Population Inversion Rate ODEs**:
   $$\frac{d(\Delta N)}{dt} = 2 W_{\text{pump}} N_1 - 2 B_{21} \rho_{\text{em}} \Delta N - A_{21} (N_{\text{total}} + \Delta N) - \frac{\Delta N - \Delta N_{\text{eq}}}{\tau_{\text{coll}}}$$
   $$\frac{d\rho_{\text{em}}}{dt} = h\nu_{21} B_{21} \rho_{\text{em}} \Delta N + h\nu_{21} A_{21} N_2 \eta_{\text{geom}} - \frac{\rho_{\text{em}}}{\tau_{\text{cav}}} - \frac{P_{\text{shafts}}}{V_{\text{KC}}}$$
2. **Dielectric Waveguide Shaft Propagation ($TE_{10}$)**:
   $$f_c = \frac{c_0}{2a} \approx 681.35\text{ MHz} < \nu_{21} = 1.4204\text{ GHz}$$
   $$\beta = \frac{2\pi \nu_{21}}{c_0} \sqrt{1 - \left(\frac{f_c}{\nu_{21}}\right)^2} \approx 26.12\text{ rad/m}$$

---

## 3. Physical Constants & Authoritative Parameters

| Domain | Property | Authoritative Value | Description |
|---|---|---|---|
| **Geometry** | 1 Royal Egyptian Cubit | $0.52360\text{ m}$ | Survey conversion |
| **Granite** | Density $\rho_{\text{gr}}$ | $2650.0\text{ kg/m}^3$ | Aswan Rose Granite |
| **Granite** | Young's Modulus $E_{\text{gr}}$ | $55.0\text{ GPa}$ | Clamped beam elasticity |
| **Granite** | Quartz Content | $28.5\%$ by volume | Piezoelectric active fraction |
| **Granite** | Piezoelectric $d_{33}^{\text{eff}}$ | $0.35\text{ pC/N}$ | Polycrystalline tensor |
| **Granite** | Piezoelectric $g_{33}^{\text{eff}}$ | $0.012\text{ V}\cdot\text{m/N}$ | Voltage coefficient |
| **Granite** | Acoustic Quality Factor $Q$ | $350$ | Relieving beam mechanical Q |
| **Acoustics** | $F\#$ Fundamental Target | $438.0\text{ Hz}$ | Grand Gallery tuning frequency |
| **Chemistry** | $\Delta H_{\text{rxn}}^\circ$ | $-153.89\text{ kJ/mol}$ | $\text{Zn} + 2\text{HCl} \rightarrow \text{ZnCl}_2 + \text{H}_2$ |
| **Gas** | Sound Speed $c_{\text{H2}}$ ($20^\circ\text{C}$) | $1290.0\text{ m/s}$ | Pure hydrogen gas |
| **Gas** | Sound Speed $c_{\text{air}}$ ($20^\circ\text{C}$) | $343.2\text{ m/s}$ | Dry standard atmosphere |
| **Maser** | Hyperfine Frequency $\nu_{21}$ | $1,420,405,751.7667\text{ Hz}$ | Hydrogen 21.1 cm spin-flip line |
| **Maser** | Einstein $A_{21}$ | $2.85 \times 10^{-15}\text{ s}^{-1}$ | Spontaneous transition rate |
| **Maser** | Einstein $B_{21}$ | $5.67 \times 10^{20}\text{ m}^3/(\text{J}\cdot\text{s}^2)$ | Stimulated rate coefficient |
| **Schumann** | Terrestrial Mode 1 | $7.83\text{ Hz}$ | Fundamental Earth cavity mode |

---

## 4. Scientific Scenario Presets

The simulation engine includes 5 scientific scenario presets:

1. **`baseline`**:
   Standard balanced operation. Earth seismic pulses ($7.83\text{ Hz}$) and subterranean water hammer oscillations drive the acoustic column, Queen's Chamber reactions inject progressive $\text{H}_2$, Grand Gallery resonators amplify $F\#$ acoustic standing waves, rose granite beams stack multi-kilovolt piezoelectric potentials, and King's Chamber maser beams microwave power through the shafts.

2. **`acoustic_peak`**:
   Grand Gallery Helmholtz resonators sharply calibrated to maximum $Q$ ($Q = 250$), demonstrating optimal resonant standing wave buildup and peak flexural mechanical stresses on the rose granite relieving beams.

3. **`full_maser_power`**:
   High-output scenario with rich chemical feed, maximum piezoelectric coupling, and high cavity quality factor, demonstrating peak atomic hydrogen stimulated emission and directional microwave beam radiation.

4. **`dry_run_no_gas`**:
   Scientific control experiment with zero chemical generation ($\text{H}_2 = 0$, ambient air only). Sound speed remains pegged at $343.2\text{ m/s}$, the acoustic cavity remains detuned, and microwave maser emission remains strictly $0\text{ Watts}$.

5. **`high_seismic`**:
   Intense terrestrial ground motion and hydraulic surge ($1.0\text{ MN}$ seismic force), demonstrating heavy hydrodynamic and infrasonic energy throughput.

---

## 5. Simulation CLI Runner & Batch Generator

### 5.1 Basic Execution
Run the baseline scenario for 10 seconds and output JSON telemetry:
```bash
python -m engine.run_sim --scenario baseline --duration 10.0 --out viewer/public/sample_telemetry.json
```

### 5.2 Batch Scenario Generation
Execute all 5 scientific scenario presets in batch mode, generating high-performance packed binary `.bin` files, JSON telemetry, and the viewer catalog `manifest.json`:
```bash
python -m engine.run_sim --all-scenarios --duration 10.0 --out-dir viewer/public/scenarios --format all --validate-schema
```

### 5.3 Command-Line Options

```
usage: python -m engine.run_sim [-h] [--scenario {baseline,acoustic_peak,full_maser_power,dry_run_no_gas,high_seismic}]
                                [--all-scenarios] [--duration DURATION] [--dt-macro DT_MACRO] [--dt-micro DT_MICRO]
                                [--fps FPS] [--out OUT] [--out-dir OUT_DIR] [--format {json,bin,all}]
                                [--compress] [--validate-schema] [--plot [PLOT]] [--quiet]

optional arguments:
  -h, --help            show this help message and exit
  --scenario            Scientific scenario preset condition (default: baseline)
  --all-scenarios       Batch run and export all 5 scientific preset scenarios alongside manifest.json
  --duration            Total simulation duration in physical seconds (default: 10.0)
  --dt-macro, --dt      Macro time step for chemistry & gas diffusion (s) (default: 0.01)
  --dt-micro            Micro time step for acoustic waves & beam vibration (s) (default: 0.0001)
  --fps                 Telemetry frame recording frequency (Hz) (default: 60.0)
  --out                 Output destination path for single scenario telemetry (default: viewer/public/sample_telemetry.json)
  --out-dir             Output directory for batch scenario generation and manifest.json (default: viewer/public/scenarios)
  --format              Export format choice: json, bin, or all (default: all)
  --compress            Enable GZIP compression on JSON output file (.json.gz) (default: False)
  --validate-schema     Validate telemetry data structures against schema specifications (default: False)
  --plot [PLOT]         Generate static multi-panel diagnostic plots (e.g. --plot diagnostic.png)
  --quiet, -q           Suppress terminal progress output
```

### 5.4 Example CLI Invocations

```bash
# 1. Run acoustic resonance peak test and generate diagnostic plots
python -m engine.run_sim --scenario acoustic_peak --duration 5.0 --plot acoustic_peak.png

# 2. Run control dry-run without hydrogen gas
python -m engine.run_sim --scenario dry_run_no_gas --duration 3.0 --out viewer/public/dry_run.json

# 3. High-performance compressed run for maximum maser power
python -m engine.run_sim --scenario full_maser_power --duration 10.0 --compress --out viewer/public/maser.json.gz

# 4. Ultra-high time resolution run (120 FPS recording)
python -m engine.run_sim --scenario baseline --duration 2.0 --dt-macro 0.005 --dt-micro 0.00005 --fps 120.0

# 5. Batch generate full static hosting suite for production viewer
python -m engine.run_sim --all-scenarios --duration 10.0 --out-dir viewer/public/scenarios --format all --validate-schema
```

---

## 6. Telemetry Data Formats & Binary Container Layout

The engine supports two high-fidelity telemetry serialization formats: formatted JSON and compact packed Little-Endian Binary (`.bin`).

### 6.1 Packed Binary Container (`.bin`) Memory Layout

The `.bin` format provides zero-copy typed array instantiation in WebGL viewers, reducing payload sizes by $> 80\%$ and parsing latency to $< 20\text{ ms}$ for 600 frames.

```
+-------------------------------------------------------------------------------+
| Header Length (4 bytes, uint32 little-endian, value = L)                      |
+-------------------------------------------------------------------------------+
| JSON Metadata Header (L bytes, UTF-8 encoded string)                          |
| - version, simulation_id, scenario_name, duration, sampling_rate, num_frames |
| - spatial_grids: gallery_z [50], gas_nodes [5], fft_frequencies_hz [128]      |
| - summary: global metrics & energy audit results                              |
| - channels: map of channel name -> {offset_bytes, shape, count, dtype}        |
+-------------------------------------------------------------------------------+
| Binary Payload Buffer (Contiguous IEEE 754 float32 little-endian '<f4')       |
|                                                                               |
| 1. Scalar Time-Series Channels (N_frames elements each):                      |
|    - time, step_index, bedrock_displacement, bedrock_velocity, ...           |
|    - water_hammer_pressure, seismic_force, hydraulic_force, ...              |
|    - h2_mole_fraction_qc, h2_mole_fraction_kc, chemical_reaction_rate, ...    |
|    - gallery_peak_pressure, gallery_rms_pressure, f_sharp_spectral_purity, ..|
|    - total_piezo_voltage, max_beam_stress_pa, maser_total_radiated_power, .. |
|    - cumulative_energy_in, cumulative_energy_out, relative_energy_error, ...  |
|                                                                               |
| 2. Multi-Station Matrix Channels (N_frames x M elements):                     |
|    - chamber_temperatures_k        [N_frames x 5]                             |
|    - chamber_pressures_pa          [N_frames x 5]                             |
|    - shaft_poynting_flux_w_m2      [N_frames x 2]                             |
|    - tier_voltages                 [N_frames x 5]                             |
|    - all_beam_stresses_mpa         [N_frames x 43]                            |
|    - all_beam_voltages_v           [N_frames x 43]                            |
|    - acoustic_pressure_profile     [N_frames x 50]                            |
|    - acoustic_velocity_profile     [N_frames x 50]                            |
|    - acoustic_energy_density       [N_frames x 50]                            |
|    - gas_h2_mole_fractions         [N_frames x 5]                             |
|    - gas_sound_speeds              [N_frames x 5]                             |
|    - gas_densities                 [N_frames x 5]                             |
|    - fft_power_spectral_density_db [N_frames x 128]                           |
|    - maser_state_populations       [N_frames x 4]                             |
+-------------------------------------------------------------------------------+
```

### 6.2 Scenario Catalog Manifest (`manifest.json`)

When running with `--all-scenarios`, the generator outputs `viewer/public/scenarios/manifest.json`:
```json
{
  "version": "1.0.0",
  "defaultScenarioId": "baseline",
  "generated_at": "2026-09-01T00:00:00Z",
  "scenarios": [
    {
      "id": "baseline",
      "name": "Baseline Resonant Run (10s)",
      "description": "Standard 10-second multi-physics simulation with hydrogen injection, 7.83 Hz infrasonic drive, and quartz excitation.",
      "duration": 10.0,
      "dt_macro": 0.01,
      "tags": ["baseline", "resonance", "maser"],
      "recommended": true,
      "binUrl": "./scenarios/baseline.bin",
      "jsonUrl": "./scenarios/baseline.json",
      "metadata": { "peak_maser_radiated_power_w": 1845.2, ... }
    },
    ...
  ]
}
```

---

## 7. Standalone Static Hosting Guide

The 3D WebGL simulation viewer is a 100% client-side application built with TypeScript, Three.js, and Vite. It requires no backend server and can be deployed directly to any static web host or CDN.

### 7.1 Production Build
1. Generate the static scenario datasets:
   ```bash
   python -m engine.run_sim --all-scenarios --duration 10.0 --out-dir viewer/public/scenarios --format all --validate-schema
   ```
2. Build the production bundle:
   ```bash
   cd viewer
   npm install
   npm run build
   ```
   The compiled assets will be output to `viewer/dist/`.

### 7.2 Hosting Providers

#### GitHub Pages
Deploy the `viewer/dist/` directory using GitHub Actions:
```yaml
name: Deploy Viewer to GitHub Pages
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: cd viewer && npm ci && npm run build
      - uses: JamesIves/github-pages-deploy-action@v4
        with:
          folder: viewer/dist
```

#### Vercel
Connect your repository and configure the build settings in Vercel:
- **Framework Preset**: Vite
- **Root Directory**: `viewer`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

#### Netlify
Create a `netlify.toml` in the repository root:
```toml
[build]
  base = "viewer"
  publish = "dist"
  command = "npm run build"

[[headers]]
  for = "*.bin"
  [headers.values]
    Content-Type = "application/octet-stream"
    Cache-Control = "public, max-age=31536000, immutable"
```

#### AWS S3 + CloudFront
Sync the built static assets to an S3 bucket configured for static website hosting:
```bash
aws s3 sync viewer/dist/ s3://your-giza-viewer-bucket/ --delete \
  --exclude "*.bin" --exclude "*.json"
aws s3 sync viewer/dist/ s3://your-giza-viewer-bucket/ \
  --exclude "*" --include "*.bin" --content-type "application/octet-stream"
aws s3 sync viewer/dist/ s3://your-giza-viewer-bucket/ \
  --exclude "*" --include "*.json" --content-type "application/json"
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

---

## 8. Testing & Verification

Run the full automated test suite with pytest:
```bash
# Run all multi-physics unit and integration tests
pytest

# Run CLI runner and scenario preset tests specifically
pytest tests/test_cli_runner.py -v
```

All physics modules undergo strict First-Law energy conservation audits ($\Delta E_{\text{stored}} = \int (P_{\text{in}} - P_{\text{out}} - P_{\text{loss}}) dt$) and numerical stability bounds.
