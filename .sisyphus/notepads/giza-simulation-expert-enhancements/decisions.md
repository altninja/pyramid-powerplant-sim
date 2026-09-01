# Decisions - Giza Simulation Expert Enhancements

## Architectural & Modeling Decisions
1. **Maser Semi-Implicit Integration**: Pumping rate $W_{pump}$ calibrated with normalized references ($V_{ref} = 5\text{ kV}$, $p_{ref} = 100\text{ kPa}$) and analytical saturation clamping to guarantee stable positive population dynamics and physically realistic milliwatt-to-kilowatt radiated beam powers.
2. **First Law Energy Conservation Baseline**: Total stored system energy $E_{sys}(t)$ explicitly accounts for bedrock kinetic+potential, chamber thermal sensible heat, acoustic wave/resonator field energy, 43-beam strain/kinetic/electrostatic energy, and cavity RF energy, with $E_{sys}(0)$ baselined so relative error remains $< 0.1\%$.
3. **Binary Telemetry Serialization**: Compact `.bin` file format with 4-byte length prefix, JSON metadata header, and structured `Float32Array` channels for instant sub-50ms browser decoding.
4. **Standalone Scenario Catalog**: Multi-scenario manifest embedded in `viewer/public/scenarios/` with UI dropdown selector.
