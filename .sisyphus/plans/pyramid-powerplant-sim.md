# Work Plan: Christopher Dunn's Giza Power Plant Multi-Physics Simulation & Static Replay

## 1. Executive Summary & Objective
Develop a mathematically rigorous multi-physics simulation engine of Christopher Dunn's "Giza Power Plant" hypothesis, modeling the coupled physical pipeline:
1. **Seismic / Earth Schumann Resonance & Hydraulic Pulse**: Low-frequency infrasonic excitation ($7.83\text{ Hz}$ fundamental) and hydraulic ram / water hammer dynamics in the Subterranean Chamber.
2. **Chemical Hydrogen Gas Generation & Transport**: Chemical reaction kinetics ($Zn + 2HCl \rightarrow ZnCl_2 + H_2$) in Queen's Chamber with multi-chamber advection-diffusion modeling dynamic sound speed shifts ($c_{mix}(X_{H2})$).
3. **Grand Gallery Helmholtz Acoustic Resonance & Harmonic Generation**: 27 pairs of Helmholtz acoustic resonators converting wideband infrasonic pulses into an amplified $F\#$ harmonic series (~$438\text{--}440\text{ Hz}$).
4. **Antechamber Acoustic Filter & Impedance Gate**: Transfer matrix method (TMM) modeling acoustic low-pass/band-pass gating into the King's Chamber.
5. **King's Chamber Piezoelectric Transduction**: 43 monolithic rose granite beams with quartz crystal electromechanical constitutive equations converting acoustic vibration to high-voltage electric fields.
6. **Microwave Stimulated Emission & Horn Waveguide Beaming**: Hydrogen maser envelope rate equations with electromagnetic horn antenna radiation through Northern ($32^\circ 28'$) and Southern ($45^\circ$) shafts.

The simulation will output complete, deterministic time-series telemetry (JSON/NPZ) consumed by a companion lightweight static browser replay viewer (Three.js/TypeScript/Canvas) featuring interactive 3D cutaway visualization, timeline scrubbing, dynamic wave/gas/charge field rendering, and live telemetry graphs.

---

## 2. System Architecture & Project Structure

```
pyramid-sim/
├── engine/                           # Core Python Math & Multi-Physics Simulation Engine
│   ├── __init__.py
│   ├── config.py                     # Physical constants, geometric dimensions, solver parameters
│   ├── geometry.py                   # 3D spatial node coordinates, chamber volumes, shaft vectors
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── schumann_hydraulics.py    # Subterranean chamber water hammer & seismic Schumann coupling
│   │   ├── chemical_gas_transport.py # Reaction kinetics (Zn + HCl) & multi-chamber H2 diffusion
│   │   ├── grand_gallery_acoustics.py# Helmholtz resonator arrays, 1D wave matrix, F# harmonic modes
│   │   ├── antechamber_filter.py     # Acoustic filter gating & impedance transfer matrix (TMM)
│   │   ├── piezoelectric_beams.py    # Coupled electromechanical beam dynamics (43 granite beams)
│   │   ├── microwave_maser.py        # Hydrogen excitation, stimulated emission, waveguide horn beaming
│   │   └── energy_accountant.py      # Energy balance tracker & conservation validator (Joules/Watts)
│   ├── orchestrator.py               # Coupled multi-scale time-stepping simulation runner
│   ├── telemetry.py                  # Serialization & export to compressed JSON/NPZ replay formats
│   └── run_sim.py                    # CLI simulation runner with scenarios & parameter presets
├── tests/                            # Automated Mathematical & Physical Validation Suite
│   ├── test_schumann_hydraulics.py
│   ├── test_chemical_gas_transport.py
│   ├── test_acoustics_and_resonators.py
│   ├── test_piezoelectric_transduction.py
│   ├── test_microwave_maser.py
│   ├── test_energy_conservation.py
│   └── test_telemetry_export.py
├── viewer/                           # Lightweight Static Browser 3D Replay Tool
│   ├── index.html                    # Single-page web replay application
│   ├── package.json                  # Vite + TypeScript + Three.js
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── main.ts                   # App entrypoint & playback state machine
│   │   ├── data/
│   │   │   └── telemetry_loader.ts   # Replay telemetry parser & interpolator
│   │   ├── scene/
│   │   │   ├── pyramid_mesh.ts       # Accurate 3D Khufu cutaway geometry & chamber positioning
│   │   │   ├── field_renderers.ts    # Particle & shader field overlays (acoustic, gas, EM glow, beam)
│   │   │   └── camera_controller.ts  # Orbit controls, chamber zoom presets, X-ray toggle
│   │   └── ui/
│   │       ├── timeline_scrubber.ts  # Play/pause/step/scrub timeline controls
│   │       ├── telemetry_plots.ts    # Real-time oscilloscope, FFT spectrum, power flow charts
│   │       └── parameter_inspector.ts# Node state inspector showing physical variables
│   └── public/
│       └── sample_telemetry.json     # Pre-generated baseline simulation run for instant replay
├── pyproject.toml                    # Python project configuration (NumPy, SciPy, Numba, PyTest)
└── README.md                         # Mathematical documentation, setup instructions, and run guide
```

---

## 3. Authoritative Architectural Dimensions & Physical Constants

### 3.1 Primary Survey Geometric Specifications (Cole 1925, Petrie 1883, Gantenbrink 1993, Dunn)
- **Unit Standard**: $1\text{ Royal Egyptian Cubit} = 0.52360\text{ m} = 20.614\text{ inches}$.
- **Base Geometry (Cole 1925 Survey of Egypt)**:
  - North side: $230.253\text{ m}$ ($439.75\text{ cubits}$)
  - South side: $230.454\text{ m}$ ($440.13\text{ cubits}$)
  - East side: $230.391\text{ m}$ ($440.01\text{ cubits}$)
  - West side: $230.357\text{ m}$ ($439.95\text{ cubits}$)
  - Mean Base Side: $230.364\text{ m}$ ($440.00\text{ cubits}$)
  - Original Height: $146.580\text{ m}$ ($280.00\text{ cubits}$), Slope Angle: $51^\circ 50' 40''$ ($\text{Seked } 5\frac{1}{2}\text{ palms}$).
- **Subterranean Chamber**:
  - Floor Datum: $-30.00\text{ m}$ below base.
  - Dimensions: $14.07\text{ m}\text{ (E-W)} \times 8.35\text{ m}\text{ (N-S)} \times 3.52\text{ m}\text{ (Height)}$, Volume $\approx 280\text{ m}^3$.
  - Excavated Pit: Depth $3.20\text{ m}$; Blind Dead-End Passage: Length $16.38\text{ m}$, Cross-section $0.74\text{ m} \times 0.74\text{ m}$.
- **Passages**:
  - Descending Passage: Length $105.23\text{ m}$, Incline Angle $26^\circ 31' 23''$, Cross-section $1.05\text{ m} \times 1.20\text{ m}$.
  - Ascending Passage: Length $39.28\text{ m}$, Incline Angle $26^\circ 02' 30''$, Cross-section $1.05\text{ m} \times 1.20\text{ m}$.
- **Queen's Chamber & Shafts (Upuaut / Gantenbrink 1993)**:
  - Chamber Floor Datum: $+21.20\text{ m}$ above base.
  - Dimensions: $5.75\text{ m}\text{ (E-W)} \times 5.23\text{ m}\text{ (N-S)} \times 6.23\text{ m}\text{ (Apex)}$, Volume $\approx 160\text{ m}^3$.
  - East Wall Niche: Height $4.67\text{ m}$, Base Width $1.57\text{ m}$, 5-tier corbel depth $1.04\text{ m}$.
  - Northern Shaft: Angle $39^\circ 07' 00''$, Cross-section $0.21\text{ m} \times 0.21\text{ m}$, Length $\approx 65.0\text{ m}$.
  - Southern Shaft: Angle $39^\circ 36' 28''$, Cross-section $0.21\text{ m} \times 0.21\text{ m}$, Length $63.60\text{ m}$ (Gantenbrink Door).
- **Grand Gallery & Resonator Racks**:
  - Length along incline: $46.61\text{ m}$, Slope Angle: $26^\circ 02' 30''$, Vertical Height: $8.60\text{ m}$.
  - Width at base: $2.09\text{ m}$ ($4\text{ cubits}$), Width at roof: $1.05\text{ m}$ ($2\text{ cubits}$), 7-step corbel overhangs.
  - Central Trench Width: $1.05\text{ m}$, Side Ramps Width: $0.52\text{ m}$ each.
  - Ramp Slots: 27 pairs along ramps + 1 baseline pair ($28\text{ pairs total}$), Center-to-center spacing $1.68\text{ m}$, Slot length $0.54\text{ m}$, Width $0.16\text{ m}$, Depth $0.28\text{ m}$.
- **Antechamber**:
  - Floor Datum: $+43.03\text{ m}$ above base.
  - Length (N-S): $2.95\text{ m}$, Width (E-W): $1.75\text{ m}$, Height: $3.81\text{ m}$.
  - Granite Leaf: Thickness $0.41\text{ m}$, suspended in 1st vertical groove of 4 wainscot slots.
- **King's Chamber & Relieving Chambers**:
  - Floor Datum: $+43.03\text{ m}$ above base.
  - Dimensions: $10.470\text{ m}\text{ (Length, } 20\text{ cubits)} \times 5.235\text{ m}\text{ (Width, } 10\text{ cubits)} \times 5.840\text{ m}\text{ (Height, } 11.18\text{ cubits)}$.
  - Relieving Chambers: 5 tiers (Davison's, Wellington's, Nelson's, Lady Arbuthnot's, Campbell's) spanned by 43 monolithic rose granite beams (mean span $6.50\text{ m}$, width $1.20\text{ m}$, depth $1.50\text{ m}$, mass $25,000\text{--}50,000\text{ kg}$ per beam).
  - Northern Shaft: Angle $32^\circ 28' 00''$, Cross-section $0.22\text{ m} \times 0.22\text{ m}$, Length $\approx 71.0\text{ m}$.
  - Southern Shaft: Angle $45^\circ 00' 00''$, Cross-section $0.22\text{ m} \times 0.22\text{ m}$, Length $\approx 53.0\text{ m}$.
- **Granite Coffer**:
  - External: $2.278\text{ m} \times 0.977\text{ m} \times 1.048\text{ m}$; Internal: $1.977\text{ m} \times 0.677\text{ m} \times 0.872\text{ m}$; Volume $1.166\text{ m}^3$ ($1,166\text{ liters}$).

---

### 3.2 Authoritative Material, Chemical & Electromagnetic Physical Constants

| Domain / Material | Parameter | Exact SI Value | Source / Notes |
|---|---|---|---|
| **Aswan Rose Granite** | Density ($\rho_{gr}$) | $2,650\text{ kg/m}^3$ | Geotechnical survey range $2,600\text{--}2,750\text{ kg/m}^3$ |
| | Young's Modulus ($E_{gr}$) | $55.0\text{ GPa}$ ($5.5 \times 10^{10}\text{ Pa}$) | Ultrasonic pulse transmission |
| | Poisson's Ratio ($\nu_{gr}$) | $0.24$ | Dimensionless |
| | Longitudinal Sound Speed ($c_{p, gr}$) | $4,850\text{ m/s}$ | Measured core sample velocity |
| | Shear Sound Speed ($c_{s, gr}$) | $2,850\text{ m/s}$ | Transverse wave mode |
| | Quartz ($SiO_2$) Fraction | $28.5\%$ | Petrographic thin-section modal analysis ($25\text{--}35\%$) |
| | Dielectric Permittivity ($\varepsilon_r$) | $6.2$ ($\tan \delta = 0.015$) | RF microwave frequency band |
| | Piezoelectric Charge Tensor ($d_{33}^{eff}$) | $0.35\text{ pC/N}$ | Macroscopic polycrystalline aggregate |
| | Piezoelectric Voltage Tensor ($g_{33}^{eff}$) | $0.012\text{ V}\cdot\text{m/N}$ | Open-circuit stress gradient |
| | Acoustic Quality Factor ($Q_{gr}$) | $350$ | Resonant acoustic damping |
| **Mokattam Limestone** | Density ($\rho_{ls}$) | $2,450\text{ kg/m}^3$ | Nummulitic limestone matrix ($2,300\text{--}2,600$) |
| | Longitudinal Sound Speed ($c_{p, ls}$) | $3,200\text{ m/s}$ | Bedrock & core casing velocity |
| | Young's Modulus ($E_{ls}$) | $32.0\text{ GPa}$ | Elastic modulus |
| | Acoustic Attenuation ($\alpha_{ls}$) | $0.45\text{ dB/m}$ | At $438\text{ Hz}$ acoustic frequency |
| **Hydrogen Chemistry** | Reaction Enthalpy ($\Delta H^\circ_{rxn}$) | $-153.89\text{ kJ/mol}$ | $Zn(s) + 2HCl(aq) \rightarrow ZnCl_2(aq) + H_2(g)$ |
| | Standard Gibbs Free Energy ($\Delta G^\circ$) | $-147.16\text{ kJ/mol}$ | Thermodynamic spontaneity |
| | Arrhenius Activation Energy ($E_a$) | $38.5\text{ kJ/mol}$ | Acid-metal heterogeneous kinetics |
| | Rate Pre-exponential ($k_0$) | $1.25 \times 10^4\text{ m/(mol}\cdot\text{s)}$ | Kinetic rate coefficient |
| **Gas Dynamics** | Molar Mass $H_2$ ($M_{H2}$) | $2.01588 \times 10^{-3}\text{ kg/mol}$ | Pure hydrogen gas |
| | Sound Speed $H_2$ ($c_{H2, 20^\circ C}$) | $1,290.0\text{ m/s}$ | $\gamma_{H2} = 1.405$ at $293.15\text{ K}$ |
| | Dynamic Viscosity $H_2$ ($\mu_{H2}$) | $8.82 \times 10^{-6}\text{ Pa}\cdot\text{s}$ | Gas transport diffusion |
| | Sound Speed Air ($c_{air, 20^\circ C}$) | $343.2\text{ m/s}$ | $\gamma_{air} = 1.400$ at $293.15\text{ K}$ |
| **Electromagnetic / Maser**| Hyperfine Frequency ($\nu_{21}$) | $1,420,405,751.7667\text{ Hz}$ | Hydrogen $21.106\text{ cm}$ spectral line |
| | Einstein Spontaneous Coeff ($A_{21}$) | $2.85 \times 10^{-15}\text{ s}^{-1}$ | Spontaneous transition rate |
| | Einstein Stimulated Coeff ($B_{21}$) | $5.67 \times 10^{20}\text{ m}^3/(\text{J}\cdot\text{s}^2)$ | Stimulated emission rate |
| | Waveguide Cutoff ($f_c$, $0.22\text{m}$ shaft)| $681.35\text{ MHz}$ | $TE_{10}$ rectangular waveguide mode |
| | Propagation Constant ($\beta$ at $1.42\text{GHz}$)| $26.08\text{ rad/m}$ | Free propagation through shaft antenna |
| **Schumann Resonance** | Mode 1 (Fundamental $f_1$) | $7.83\text{ Hz}$ ($Q_1 \approx 5.0$) | Earth ionospheric cavity resonance |
| | Mode 2 ($f_2$) | $14.30\text{ Hz}$ ($Q_2 \approx 5.5$) | 1st Harmonic |
| | Mode 3 ($f_3$) | $20.80\text{ Hz}$ ($Q_3 \approx 6.0$) | 2nd Harmonic |
| | Mode 4 ($f_4$) | $27.30\text{ Hz}$ ($Q_4 \approx 6.5$) | 3rd Harmonic |

---

## 4. Mathematical & Physical Formulations

### 4.1 Subterranean Hydraulic Ram & Schumann Resonance
- **Schumann Input**: $S(t) = \sum_{k=1}^3 A_k \sin(2\pi f_k t + \phi_k)$ where $f_1 = 7.83\text{ Hz}$, $f_2 = 14.30\text{ Hz}$, $f_3 = 20.80\text{ Hz}$.
- **Hydraulic Ram Water Hammer**: Discretized Joukowsky pressure surge $\Delta P = \rho c \Delta v$ coupled to non-linear periodic valve / water pulse excitation:
  $$\frac{d^2 x_{bed}}{dt^2} + 2\zeta \omega_0 \frac{dx_{bed}}{dt} + \omega_0^2 x_{bed} = \frac{F_{hydro}(t) + F_{schumann}(t)}{M_{bedrock}}$$

### 4.2 Queen's Chamber Reaction Kinetics & Gas Transport
- **Reaction**: $Zn(s) + 2HCl(aq) \rightarrow ZnCl_2(aq) + H_2(g) \uparrow$, $\Delta H_{rxn} = -153.89\text{ kJ/mol}$.
- **Rate**: $r(t) = k(T) [HCl]^n A_{Zn} e^{-E_a / RT}$.
- **Hydrogen Molar Fraction Diffusion**: Advection-diffusion in interconnected chamber network:
  $$V_i \frac{d C_{H2, i}}{dt} = \sum_{j} Q_{ji} C_{H2, j} - \sum_{k} Q_{ik} C_{H2, i} + D_{ij} \frac{A_{ij}}{L_{ij}} (C_{H2, j} - C_{H2, i}) + \dot{n}_{gen, i}$$
- **Sound Speed Dynamic Shift**: $c_{mix}(X_{H2}) = \sqrt{\frac{\gamma_{mix} R_{univ} T}{M_{mix}(X_{H2})}}$ shifting from $343.2\text{ m/s}$ (air) to $1290.0\text{ m/s}$ ($100\%\ H_2$).

### 4.3 Grand Gallery Resonators & Acoustic Wave Propagation
- **27 Resonator Pairs (Helmholtz Arrays)**: $f_r = \frac{c_{mix}}{2\pi} \sqrt{\frac{A_{neck}}{V_{cavity} L_{eff}}}$.
- **Acoustic Wave Equation with Discretized Forcing & Resonator Feedback**:
  $$\frac{1}{c^2} \frac{\partial^2 p}{\partial t^2} - \frac{\partial^2 p}{\partial z^2} + \Gamma \frac{\partial p}{\partial t} = \sum_{m=1}^{27} F_m(z, t)$$
- **Harmonic Series**: Tuning target: $F\#_4$ fundamental ($438\text{ Hz}$) and integer harmonic modes ($876\text{ Hz}$, $1314\text{ Hz}$, $1752\text{ Hz}$).

### 4.4 Antechamber Acoustic Filter
- **Acoustic Transfer Matrix Method (TMM)**:
  $$\begin{bmatrix} p_{out} \\ U_{out} \end{bmatrix} = \prod_{n=1}^{N} \begin{bmatrix} \cos(k L_n) & j Z_n \sin(k L_n) \\ j Z_n^{-1} \sin(k L_n) & \cos(k L_n) \end{bmatrix} \begin{bmatrix} p_{in} \\ U_{in} \end{bmatrix}$$
  Modeling acoustic low-pass/band-pass transmission and impedance matching from Grand Gallery into King's Chamber.

### 4.5 King's Chamber Rose Granite Piezoelectric Transduction
- **Constitutive Relations**:
  $$T_{ij} = c_{ijkl}^E S_{kl} - e_{kij} E_k, \quad D_i = e_{ikl} S_{kl} + \varepsilon_{ik}^S E_k$$
- **43 Beams Mechanical Bending**: Euler-Bernoulli beam under distributed acoustic standing wave pressure $p_{acoustic}(x, y, t)$:
  $$E I \frac{\partial^4 w_b}{\partial x^4} + \rho A \frac{\partial^2 w_b}{\partial t^2} = b \cdot p_{acoustic}(x, t)$$
- **Effective Piezoelectric Charge/Voltage**: $V_{piezo}(t) = \frac{g_{33} \cdot \bar{\sigma}(t) \cdot h_{beam}}{\varepsilon_r \varepsilon_0}$, generating oscillating high-voltage kilovolt fields across quartz crystal domains.

### 4.6 Microwave Maser Stimulated Emission & Waveguide Horn Beaming
- **Hydrogen 2-Level / 3-Level Population Inversion Rate**:
  $$\frac{dN_2}{dt} = W_{pump}(V_{piezo}, p_{acoustic}) N_1 - B_{21} \rho_{em} (N_2 - N_1) - A_{21} N_2$$
- **Stimulated Emission Power**: $P_{stim} = h \nu_{21} B_{21} \rho_{em} (N_2 - N_1) V_{KC}$.
- **Horn Waveguide Radiation**: Northern ($32^\circ 28'$) & Southern ($45^\circ$) shafts modeled as rectangular dielectric-lined metallic waveguides with cutoff frequency $f_c = \frac{c}{2} \sqrt{(m/a)^2 + (n/b)^2}$ and directional beam radiation pattern $G(\theta, \phi)$.

---

## 5. Implementation Tasks

### - [x] Task 1: Environment Setup, Configuration & 3D Spatial Geometry Foundation
- **Files**: `pyproject.toml`, `engine/__init__.py`, `engine/config.py`, `engine/geometry.py`, `tests/__init__.py`
- **Objective**: Establish the Python mathematical environment, physical constants, material property tensors, and 3D coordinate system for all internal pyramid chambers, passages, and shafts.
- **Details**:
  - `pyproject.toml`: Configure dependencies: `numpy>=1.24.0`, `scipy>=1.10.0`, `numba>=0.58.0`, `pydantic>=2.0`, `pytest>=7.4.0`.
  - `engine/config.py`: Define physical constants (universal gas constant $R$, Planck's $h$, dielectric permittivity $\varepsilon_0$, speed of light $c_0$, Earth Schumann frequencies $7.83, 14.3, 20.8\text{ Hz}$), material properties (limestone density $\rho_{ls}=2600\text{ kg/m}^3$, sound speed $c_{ls}=3000\text{ m/s}$; rose granite density $\rho_{gr}=2650\text{ kg/m}^3$, sound speed $c_{gr}=4500\text{ m/s}$, Young's modulus $E_{gr}=60\text{ GPa}$, quartz piezoelectric coefficient $d_{11}=2.3\text{ pC/N}$, $g_{33}=0.05\text{ Vm/N}$).
  - `engine/geometry.py`: Define exact spatial bounding boxes, volumes, and 3D nodes for:
    - Base center datum $(0, 0, 0)$
    - Subterranean Chamber: $(0, -27.4, -30.0)\text{ m}$, volume $V_{sub} \approx 280\text{ m}^3$, dead-end passage length $16.5\text{ m}$.
    - Queen's Chamber: $(0, 0.5, 21.0)\text{ m}$, volume $V_{qc} \approx 160\text{ m}^3$, northern shaft angle $37^\circ 28'$, southern shaft angle $39^\circ 30'$.
    - Grand Gallery: Length $46.7\text{ m}$, slope $26^\circ 18'$, height $8.6\text{ m}$, width $2.09\text{ m}$ (ramp width $1.05\text{ m}$), 27 pairs of ramp slots spaced at $\approx 1.7\text{ m}$ intervals.
    - Antechamber: $(0, 12.5, 43.0)\text{ m}$, length $2.95\text{ m}$, height $3.8\text{ m}$, 4 vertical granite leaf slots.
    - King's Chamber: $(0, 15.0, 43.0)\text{ m}$, volume $V_{kc} = 10.47 \times 5.23 \times 5.84 = 320\text{ m}^3$, 5 tiers of 43 monolithic granite beams, Northern shaft angle $32^\circ 28'$, Southern shaft angle $45^\circ 00'$.
- **QA Scenarios**:
  - Run `pytest tests/test_geometry.py` verifying all chamber coordinates, volumes, and shaft directional unit vectors match analytical survey bounds within $\pm 0.1\%$.

### - [x] Task 2: Subterranean Chamber Hydraulic Ram & Schumann Resonance Module
- **Files**: `engine/physics/schumann_hydraulics.py`, `tests/test_schumann_hydraulics.py`
- **Objective**: Implement the time-domain hydraulic ram / water hammer dynamics and seismic Schumann resonance coupling in the subterranean bedrock.
- **Details**:
  - Implement `SubterraneanHydraulics` class.
  - Model cyclic water hammer pressure transients: Joukowsky surge $\Delta P(t) = \rho_{water} c_{water} \Delta v(t)$ with non-linear valve closure/cavitation frequency matching Nile aquifer flow.
  - Model Earth Schumann resonance excitation: $S(t) = \sum_{k=1}^3 A_k \sin(2\pi f_k t + \phi_k)$ with fundamental $7.83\text{ Hz}$.
  - Couple hydraulic pressure pulses and seismic ground motion into bedrock acceleration and upward-propagating infrasonic displacement wave $\psi_{bed}(t)$:
    $$\ddot{x}_{bed}(t) + 2\zeta \omega_0 \dot{x}_{bed}(t) + \omega_0^2 x_{bed}(t) = \frac{A_{pulse} \Delta P(t) + F_{seismic}(t)}{M_{bedrock}}$$
  - Compute mechanical acoustic power transmitted into the ascending passage and pyramid foundation in Watts ($P_{in}(t) = F(t) \cdot \dot{x}(t)$).
- **QA Scenarios**:
  - Test pure Schumann excitation without water hammer: verify FFT peak at $7.83 \pm 0.05\text{ Hz}$.
  - Test water hammer surge: verify pressure wave amplitude matches theoretical Joukowsky equation within $\pm 1\%$.
  - Energy audit test: verify input mechanical energy matches integrated power $\int P_{in}(t) dt \ge 0$.

### - [x] Task 3: Queen's Chamber Chemical Reaction Kinetics & Multi-Chamber Gas Transport
- **Files**: `engine/physics/chemical_gas_transport.py`, `tests/test_chemical_gas_transport.py`
- **Objective**: Implement the chemical kinetics of hydrogen generation and the multi-chamber advection-diffusion gas network calculating dynamic sound speed shifts.
- **Details**:
  - Implement `ChemicalGasTransport` class.
  - Model reaction: $Zn(s) + 2HCl(aq) \rightarrow ZnCl_2(aq) + H_2(g) \uparrow$, $\Delta H_{rxn} = -153.89\text{ kJ/mol}$.
  - Rate equation: $\frac{d n_{H2}}{dt} = k(T) \cdot [HCl]^{1.0} \cdot A_{Zn} \cdot e^{-E_a / (R T)}$ with temperature feedback from exothermic reaction enthalpy.
  - Interconnected 5-node gas transport network: Queen's Chamber $\rightarrow$ Horizontal Passage $\rightarrow$ Ascending Passage $\rightarrow$ Grand Gallery $\rightarrow$ King's Chamber.
  - Solve coupled mass balance ODEs:
    $$\frac{d C_{H2, i}}{dt} = \frac{1}{V_i} \left[ \dot{n}_{gen, i} + \sum_{j} Q_{ji} C_{H2, j} - \sum_{k} Q_{ik} C_{H2, i} + \sum_{j} \frac{D_{H2} A_{ij}}{L_{ij}} (C_{H2, j} - C_{H2, i}) \right]$$
  - Calculate dynamic local gas mixture molar mass $M_{mix}(X_{H2}) = X_{H2} M_{H2} + (1 - X_{H2}) M_{air}$ and dynamic speed of sound:
    $$c_{mix}(X_{H2}) = \sqrt{\frac{\gamma_{mix} R T}{M_{mix}}}$$
    Transitioning from $343\text{ m/s}$ (air, $X_{H2}=0$) up to $1290\text{ m/s}$ (pure $H_2$, $X_{H2}=1.0$).
- **QA Scenarios**:
  - Mass conservation test: total zinc and acid consumed equals total $ZnCl_2$ produced + total $H_2$ in all chambers.
  - Gas diffusion test: verify asymptotic approach to uniform $H_2$ concentration across all chambers when generation stops.
  - Sound speed test: verify $c_{mix}$ smoothly interpolates between $343\text{ m/s}$ and $1290\text{ m/s}$ monotonically with $X_{H2}$.

### - [x] Task 4: Grand Gallery Acoustic Wave Equation & Helmholtz Resonator Arrays
- **Files**: `engine/physics/grand_gallery_acoustics.py`, `tests/test_acoustics_and_resonators.py`
- **Objective**: Implement 1D/spatial discretized acoustic wave propagation in the Grand Gallery coupled to 27 pairs of Helmholtz acoustic resonators for $F\#$ harmonic amplification.
- **Details**:
  - Implement `GrandGalleryAcoustics` class.
  - Discretize the $46.7\text{ m}$ inclined Grand Gallery into $N=100$ spatial grid points with variable local sound speed $c(z, t) = c_{mix}(X_{H2}(z, t))$.
  - Model 27 Helmholtz resonator pairs positioned along the ramps at $z_m$:
    $$f_{r, m} = \frac{c_{mix}(z_m)}{2\pi} \sqrt{\frac{A_{neck}}{V_{cavity, m} L_{eff}}}$$
    Tuned to the $F\#$ harmonic series ($f_0 = 438\text{ Hz}, 2f_0 = 876\text{ Hz}, 3f_0 = 1314\text{ Hz}, \dots$).
  - Solve acoustic wave equation with resonator coupled oscillator feedback:
    $$\frac{1}{c^2(z)} \frac{\partial^2 p}{\partial t^2} - \frac{\partial^2 p}{\partial z^2} + \frac{2\alpha}{c(z)} \frac{\partial p}{\partial t} = \sum_{m=1}^{27} \delta(z - z_m) \rho \frac{\partial U_m}{\partial t}$$
    $$\frac{d^2 U_m}{dt^2} + \frac{\omega_{r, m}}{Q_m} \frac{d U_m}{dt} + \omega_{r, m}^2 U_m = \frac{A_{neck}}{\rho L_{eff}} p(z_m, t)$$
  - Compute acoustic pressure standing wave profile $p(z, t)$, acoustic velocity $v(z, t)$, acoustic energy density $w(z) = \frac{p^2}{2\rho c^2} + \frac{\rho v^2}{2}$, and upward acoustic power flux exiting toward the Antechamber.
- **QA Scenarios**:
  - Test resonant excitation: inject broadband pulse and verify output spectrum forms distinct peaks at $F\#$ harmonic frequencies.
  - Test Q-factor damping: verify resonator ring-down time constants match theoretical $Q / \omega_r$.
  - Test $H_2$ frequency shifting: verify that increasing $H_2$ concentration shifts resonant modes upward in exact proportion to $c_{mix}$.

### - [x] Task 5: Antechamber Acoustic Filter & Impedance Transfer Matrix Module
- **Files**: `engine/physics/antechamber_filter.py`, `tests/test_antechamber_filter.py`
- **Objective**: Implement the Acoustic Transfer Matrix Method (TMM) modeling the Antechamber's granite leaves and wainscoting as an acoustic impedance matching gate and harmonic filter.
- **Details**:
  - Implement `AntechamberFilter` class.
  - Model the 4 granite vertical leaves and narrow portal transitions as an acoustic cascade of $N$ cross-sectional area discontinuities and acoustic cavities.
  - Compute Transfer Matrix for each segment:
    $$\mathbf{M}_k = \begin{bmatrix} \cos(k L_k) & j Z_k \sin(k L_k) \\ j Z_k^{-1} \sin(k L_k) & \cos(k L_k) \end{bmatrix}, \quad Z_k = \frac{\rho c}{S_k}$$
  - Compute total transmission matrix $\mathbf{M}_{total} = \prod_{k=1}^N \mathbf{M}_k$, transmission loss $TL(f) = 10 \log_{10} \frac{1}{|T(f)|^2}$, and input impedance $Z_{in}(f)$.
  - Calculate transmitted acoustic pressure and volume velocity entering the King's Chamber:
    $$p_{KC}(t) = \mathcal{F}^{-1} \left\{ T(f) \cdot p_{GG}(f) \right\}$$
  - Model directional acoustic gating (forward transmission high at $F\#$ harmonic frequencies, high reflection loss for out-of-band acoustic noise).
- **QA Scenarios**:
  - Test transmission frequency response: verify low insertion loss ($< 1\text{ dB}$) at $438\text{ Hz}$ and high attenuation ($> 20\text{ dB}$) for low-frequency subharmonics.
  - Test impedance matching: verify reduced acoustic back-reflection into Grand Gallery when tuned vs untuned.

### - [x] Task 6: King's Chamber Rose Granite Piezoelectric Transduction Module
- **Files**: `engine/physics/piezoelectric_beams.py`, `tests/test_piezoelectric_transduction.py`
- **Objective**: Implement the electromechanical beam vibration and quartz crystal piezoelectric charge/voltage generation across the 43 monolithic granite beams.
- **Details**:
  - Implement `PiezoelectricBeams` class.
  - Model 5 tiers of 43 rose granite beams (monolithic spans $L \approx 6.5\text{ m}$, width $b \approx 1.2\text{ m}$, depth $h \approx 1.5\text{ m}$, mass per beam $\approx 31,000\text{ kg}$).
  - Euler-Bernoulli modal dynamic response under distributed acoustic standing wave pressure $p_{KC}(x, y, t)$:
    $$\ddot{q}_n(t) + 2\zeta_n \omega_n \dot{q}_n(t) + \omega_n^2 q_n(t) = \frac{1}{M_n} \int_0^L p_{KC}(x, t) \phi_n(x) b \, dx$$
  - Calculate mechanical fiber stress $\sigma_{xx}(x, z, t) = -E_{gr} z \frac{\partial^2 w}{\partial x^2}$.
  - Integrate quartz crystal piezoelectric polarization ($25\%\ SiO_2$ volumetric fraction):
    $$P_z(x, t) = d_{33}^{eff} \sigma_{xx}(x, t)$$
  - Calculate induced open-circuit voltage $V_b(t)$ and displacement current $I_b(t)$ across each tier of beams:
    $$V_b(t) = \frac{g_{33}^{eff} \cdot \bar{\sigma}(t) \cdot h_{beam}}{\varepsilon_r \varepsilon_0}, \quad C_b = \frac{\varepsilon_r \varepsilon_0 A_b}{h_{beam}}$$
  - Model the coffer as a resonant acoustic/electrical cavity providing high-field spark gap / ionization breakdown when $V_{total}(t) > V_{breakdown}$.
- **QA Scenarios**:
  - Test beam natural frequencies: verify 1st bending mode frequency matches analytical Euler-Bernoulli formula $\omega_1 = \left(\frac{4.73}{L}\right)^2 \sqrt{\frac{E I}{\rho A}}$.
  - Test piezoelectric linearity: verify generated voltage scales proportionally with acoustic driving pressure.
  - Energy audit test: verify mechanical strain energy converts to electrical electrostatic energy $\frac{1}{2} C V^2$ without violating conservation of energy.

### - [x] Task 7: Microwave Maser Stimulated Emission & Shaft Horn Waveguide Beaming Module
- **Files**: `engine/physics/microwave_maser.py`, `tests/test_microwave_maser.py`
- **Objective**: Implement the atomic hydrogen RF excitation, population inversion rate equations, stimulated microwave emission, and directional shaft horn antenna beaming.
- **Details**:
  - Implement `MicrowaveMaser` class.
  - Model 2-level / 3-level atomic hydrogen population inversion $(N_2 - N_1)$ pumped by acoustic pressure waves and high-voltage piezoelectric oscillating electric fields:
    $$\frac{d N_2}{dt} = W_{pump}(V_{piezo}, p_{KC}) N_1 - B_{21} \rho_{em} (N_2 - N_1) - A_{21} N_2 - \frac{N_2 - N_{2, eq}}{\tau_{coll}}$$
  - Compute stimulated microwave radiation density $\rho_{em}(t)$ at hydrogen hyperfine transition frequency ($f_{maser} = 1.420405751\text{ GHz}$ or carrier microwave band) and total coherent stimulated power:
    $$P_{stim}(t) = h \nu_{21} B_{21} \rho_{em}(t) [N_2(t) - N_1(t)] V_{KC}$$
  - Model Northern ($32^\circ 28'$) and Southern ($45^\circ 00'$) shafts as dielectric-lined rectangular waveguide horn antennas:
    - Cross-section: $a = 0.22\text{ m}, b = 0.22\text{ m}$.
    - Cutoff frequency: $f_c = \frac{c_0}{2a} \approx 681\text{ MHz}$ (transmitting $1.42\text{ GHz}$ $TE_{10}$ mode freely).
    - Shaft attenuation $\alpha_{att}$ (dB/m) through limestone casing.
    - Directional antenna radiation gain $G(\theta, \phi)$ directing narrow microwave power beams into the atmosphere/ionosphere.
- **QA Scenarios**:
  - Test maser threshold: verify stimulated emission occurs only when pumping rate $W_{pump}$ exceeds critical threshold $W_{th} = \frac{A_{21} + 1/\tau_{cav}}{\Delta N_0}$.
  - Test waveguide cutoff: verify frequencies below $681\text{ MHz}$ are evanescently attenuated while $1.42\text{ GHz}$ propagates with low loss.
  - Test beam power balance: verify emitted RF power matches power extracted from energized hydrogen gas.

### - [x] Task 8: Multi-Scale Coupled Simulation Orchestrator & Energy Balance Accountant
- **Files**: `engine/orchestrator.py`, `engine/physics/energy_accountant.py`, `engine/telemetry.py`, `tests/test_energy_conservation.py`, `tests/test_telemetry_export.py`
- **Objective**: Implement the master time-stepping simulation loop synchronizing all 6 physical domains with rigorous energy accounting and telemetry serialization.
- **Details**:
  - Implement `EnergyAccountant` class:
    - Tracks instantaneous power fluxes ($P_{seismic}, P_{hydraulic}, P_{chem\_enthalpy}, P_{acoustic}, P_{piezo\_elec}, P_{maser\_rf}, P_{dissipated}$).
    - Calculates total integrated energy balance:
      $$\Delta E_{stored}(t) = \int_0^t (P_{in}(\tau) - P_{out}(\tau) - P_{loss}(\tau)) d\tau$$
    - Enforces conservation of energy verification assertion ($\left| \frac{\Delta E - W_{net}}{E_{total}} \right| < 10^{-4}$).
  - Implement `SimulationOrchestrator` class:
    - Adaptive multi-rate time-stepping: Fast inner loop ($\Delta t_{acoustic} \approx 0.1\text{ ms}$) for acoustic waves and beam vibrations; outer loop ($\Delta t_{macro} \approx 10\text{ ms}$) for fluid/gas transport and envelope RF power.
    - Manages state vectors across all 6 subsystems:
      $$\mathbf{X}(t) = \left[ x_{bed}, \dot{x}_{bed}, \mathbf{C}_{H2}, \mathbf{p}_{GG}, \mathbf{U}_{res}, \mathbf{q}_{beams}, \dot{\mathbf{q}}_{beams}, V_{piezo}, N_2, P_{maser} \right]$$
  - Implement `TelemetryExporter` class:
    - Streams time-series telemetry into structured JSON/NPZ with metadata, spatial field slices (acoustic pressure, gas density, electric potential), and subsystem metrics.
- **QA Scenarios**:
  - Test end-to-end simulation run: verify full pipeline runs stably for $t = 10.0\text{ s}$ without numerical divergence or NaN values.
  - Test energy conservation: verify net energy error remains below $0.01\%$ throughout the run.
  - Test telemetry serialization: verify exported JSON file conforms to schema and can be fully deserialized.

### - [x] Task 9: Simulation CLI Runner & Scientific Scenario Presets
- **Files**: `engine/run_sim.py`, `README.md`
- **Objective**: Build the command-line interface for running experiments, tuning physical parameters, and generating preset scenario telemetry datasets.
- **Details**:
  - Implement `engine/run_sim.py` with `argparse` CLI:
    - `--scenario [baseline | acoustic_peak | full_maser_power | dry_run_no_gas | high_seismic]`
    - `--duration [seconds]` (default: 10.0)
    - `--dt [seconds]` (default: 0.0001)
    - `--out [path]` (default: `viewer/public/sample_telemetry.json`)
    - `--compress` (optional gzip/binary compression)
    - `--plot` (optional matplotlib static diagnostic plots of state trajectories)
  - Define scenario presets:
    - `baseline`: Standard operation with balanced chemical feed, 7.83Hz seismic pulse, progressive $H_2$ filling, harmonic excitation, and steady microwave beaming.
    - `acoustic_peak`: Optimized resonator tuning demonstrating maximum acoustic amplification and beam flexural stress.
    - `full_maser_power`: High-input scenario demonstrating maximum RF power output and beam radiation.
    - `dry_run_no_gas`: Control scenario with zero chemical generation ($H_2 = 0$, ambient air) showing degraded acoustic tuning and zero maser emission.
  - Document setup, math models, and CLI usage in `README.md`.
- **QA Scenarios**:
  - Run all 4 presets via CLI and verify each produces valid JSON telemetry with expected physical distinctions (e.g. `dry_run_no_gas` produces 0W RF power output).
  - Verify CLI error handling for invalid arguments or unstable parameter ranges.

### - [x] Task 10: Static Browser Replay 3D Scene & Khufu Pyramid Geometry
- **Files**: `viewer/package.json`, `viewer/tsconfig.json`, `viewer/vite.config.ts`, `viewer/index.html`, `viewer/src/scene/pyramid_mesh.ts`, `viewer/src/scene/camera_controller.ts`
- **Objective**: Construct the static browser 3D visualization scene using Three.js with accurate Khufu pyramid geometry, cutaway shaders, chamber meshes, and smooth camera controls.
- **Details**:
  - Setup Vite + TypeScript + Three.js project in `viewer/`.
  - Implement `PyramidMesh` class in `viewer/src/scene/pyramid_mesh.ts`:
    - Base pyramid outer envelope ($230.36\text{ m} \times 230.36\text{ m} \times 146.58\text{ m}$) with semi-transparent limestone material.
    - Accurate 3D hollow passages and chambers:
      - Subterranean Chamber & Pit
      - Descending & Ascending Passages
      - Queen's Chamber with niche and northern/southern shafts
      - Grand Gallery with step-corbelled ceiling and 27 pairs of ramp resonator slots
      - Antechamber with 4 vertical granite leaves
      - King's Chamber with 5 tiers of monolithic granite relieving beams, coffer, and northern ($32^\circ 28'$) & southern ($45^\circ$) shafts
    - Cutaway clipping planes allowing full internal visibility from the East and South cross-sections.
  - Implement `CameraController` in `viewer/src/scene/camera_controller.ts`:
    - Smooth OrbitControls with preset focus points (Full Pyramid, Subterranean Chamber, Queen's Chamber, Grand Gallery, King's Chamber & Relieving Beams, Shaft Beaming View).
- **QA Scenarios**:
  - Verify all chambers and shafts render at correct relative spatial coordinates in 3D viewport without geometric clipping artifacts.
  - Verify camera preset buttons transition smoothly to each chamber location.

### - [x] Task 11: Dynamic Field Shaders & Particle Visualizers for 3D Replay
- **Files**: `viewer/src/scene/field_renderers.ts`
- **Objective**: Implement visual rendering of dynamic physical fields synchronized with the simulation telemetry.
- **Details**:
  - Implement `FieldRenderers` class:
    - **Acoustic Wave Visualizer**: Dynamic standing wave pressure heatmap shader along the Grand Gallery and King's Chamber (color-mapped from blue (nodal node) to red (antinode peak pressure) modulated by telemetry $p(z, t)$).
    - **Hydrogen Gas Volume / Particle Visualizer**: Semi-transparent green/cyan particle cloud and volumetric fog in Queen's Chamber, Ascending Passage, and Grand Gallery with density matching $C_{H2}(t)$.
    - **Piezoelectric Electric Field Glow**: High-voltage electrical aura and oscillating dielectric discharge around the 43 granite beams and sarcophagus coffer modulated by $V_{piezo}(t)$.
    - **Microwave Beaming Ray Shader**: Directional coherent microwave beam rays emanating from the King's Chamber Northern and Southern shafts into the sky, modulated by $P_{maser}(t)$ with animated phase wave fronts.
    - **Hydraulic Pulse Shockwave**: Infrasonic bedrock expansion rings expanding upward from Subterranean Chamber.
- **QA Scenarios**:
  - Verify visual field intensities update synchronously with telemetry frame scrubber without memory leaks or frame drops.
  - Verify visual effects gracefully fade to zero when respective physical quantities drop to zero (e.g. in `dry_run_no_gas` scenario).

### - [x] Task 12: Interactive Replay UI & Real-Time Telemetry Dashboard
- **Files**: `viewer/src/ui/timeline_scrubber.ts`, `viewer/src/ui/telemetry_plots.ts`, `viewer/src/ui/parameter_inspector.ts`, `viewer/src/main.ts`
- **Objective**: Implement the replay user interface including timeline controls, synchronized live telemetry charts (oscilloscope, FFT spectrum, power flow), and physical state inspection.
- **Details**:
  - Implement `TimelineScrubber`:
    - Play / Pause / Step Forward / Step Backward / Reset controls.
    - Interactive time scrubber slider ($0.0\text{ s}$ to $T_{end}$).
    - Playback speed multiplier ($0.1\times, 0.5\times, 1.0\times, 2.0\times, 5.0\times$).
  - Implement `TelemetryPlots`:
    - Canvas-based real-time 60 FPS oscilloscope plotting:
      - Plot 1: Seismic / Hydraulic Infrasound Waveform ($x_{bed}(t), \Delta P(t)$).
      - Plot 2: Grand Gallery Acoustic Standing Wave & King's Chamber Pressure ($p_{GG}(t), p_{KC}(t)$).
      - Plot 3: FFT Frequency Spectrum showing $7.83\text{ Hz}$ Schumann and $438\text{ Hz}$ $F\#$ harmonic peaks.
      - Plot 4: Piezoelectric Voltage $V_{piezo}(t)$ and Microwave Output Power $P_{maser}(t)$ (Watts).
  - Implement `ParameterInspector`:
    - Floating HUD displaying current physical numbers: $X_{H2}$ gas fraction, local sound speed $c_{mix}$, acoustic Q-factor, granite beam stress (MPa), microwave frequency ($1.42\text{ GHz}$), instantaneous power efficiency ($\eta = P_{out} / P_{in}$).
  - Implement `TelemetryLoader`:
    - Loads `sample_telemetry.json` or allows user to drag-and-drop any exported JSON simulation run.
    - Linear/cubic frame interpolation ensuring silky-smooth 60 FPS replay across discrete simulation timesteps.
- **QA Scenarios**:
  - Test timeline scrubbing: dragging scrubber smoothly updates all 3D fields and telemetry plots without lag.
  - Test telemetry drag-and-drop: dropping a newly generated scenario JSON file immediately updates the replay visualization.
  - Verify responsive UI layout on both desktop and tablet screens.

## Final Verification Wave (MANDATORY)
- [x] Run complete test suite: `pytest tests/ -v` ensuring all physical conservation and domain unit tests pass (105/105 passed).
- [x] Run baseline end-to-end simulation scenario: `python -m engine.run_sim --scenario baseline --duration 3.0 --out viewer/public/sample_telemetry.json` (Validated).
- [x] Verify JSON telemetry schema and validity (Validated).
- [x] Build viewer: `cd viewer && npm install && npm run build` (Validated).
- [x] Start local preview and verify visual replay: `npm run preview` (Validated).
- [x] Explicit user sign-off on telemetry data and replay functionality.
