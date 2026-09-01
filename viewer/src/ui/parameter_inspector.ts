import { TelemetryFrame } from '../data/telemetry_loader';

interface BeamCellInfo {
  tierIndex: number;
  beamInTier: number;
  globalIndex: number;
  tierName: string;
  element: HTMLElement;
}

const TIER_METADATA = [
  { name: "Davison", fullName: "Davison's Chamber", count: 9 },
  { name: "Wellington", fullName: "Wellington's Chamber", count: 9 },
  { name: "Nelson", fullName: "Nelson's Chamber", count: 9 },
  { name: "Arbuthnot", fullName: "Lady Arbuthnot's Chamber", count: 9 },
  { name: "Campbell", fullName: "Campbell's Chamber", count: 7 },
];

export class ParameterInspector {
  private container: HTMLElement;
  private isCollapsed: boolean = false;

  private valBedrockDisp: HTMLElement | null = null;
  private valHammerPress: HTMLElement | null = null;
  private barInfrasoundFill: HTMLElement | null = null;

  private valH2Qc: HTMLElement | null = null;
  private valH2Kc: HTMLElement | null = null;
  private valSoundSpeed: HTMLElement | null = null;
  private valChemTemp: HTMLElement | null = null;
  private barGasFill: HTMLElement | null = null;

  private valGalleryPeak: HTMLElement | null = null;
  private valGalleryRms: HTMLElement | null = null;
  private valPurity: HTMLElement | null = null;
  private barAcousticFill: HTMLElement | null = null;

  private valPiezoVolt: HTMLElement | null = null;
  private valBeamStress: HTMLElement | null = null;
  private badgeSpark: HTMLElement | null = null;
  private barPiezoFill: HTMLElement | null = null;

  private beamCells: BeamCellInfo[] = [];
  private beamTooltip: HTMLElement | null = null;
  private hoveredBeamIndex: number | null = null;
  private latestFrame: TelemetryFrame | null = null;

  private valMaserPower: HTMLElement | null = null;
  private badgeMaserStatus: HTMLElement | null = null;
  private barMaserFill: HTMLElement | null = null;

  private valPowerIn: HTMLElement | null = null;
  private valPowerOut: HTMLElement | null = null;
  private valEfficiency: HTMLElement | null = null;
  private valEnergyBalance: HTMLElement | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.initDOM();
  }

  private initDOM(): void {
    let globalBeamCounter = 1;
    let tierRowsHtml = '';

    for (let t = 0; t < TIER_METADATA.length; t++) {
      const tier = TIER_METADATA[t];
      let cellsHtml = '';
      for (let b = 0; b < tier.count; b++) {
        const beamNum = globalBeamCounter;
        cellsHtml += `<div class="beam-cell" data-beam-idx="${beamNum - 1}" data-tier="${t}" data-tier-idx="${b}" title="Beam #${beamNum} (${tier.name} #${b + 1})">${beamNum}</div>`;
        globalBeamCounter++;
      }
      tierRowsHtml += `
        <div class="beam-tier-row">
          <span class="beam-tier-label">${tier.name} (${tier.count})</span>
          <div class="beam-tier-cells">${cellsHtml}</div>
        </div>
      `;
    }

    this.container.innerHTML = `
      <div class="glass-panel panel-section inspector-panel" id="inspector-panel-inner">
        <div class="section-header">
          <span>Physical Telemetry HUD</span>
          <div style="display: flex; gap: 6px; align-items: center;">
            <span class="section-badge">REAL-TIME</span>
            <button class="btn btn-icon" id="btn-inspector-collapse" title="Toggle Inspector [I]" style="width: 22px; height: 22px; font-size: 0.75rem;">✕</button>
          </div>
        </div>

        <div class="inspector-scroll">
          <div class="hud-group">
            <div class="hud-group-title">
              <span>1. Infrasonic & Hydraulic Drive</span>
              <span style="color: var(--cyan-accent); font-family: var(--font-mono);">7.83 Hz</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Bedrock Disp (x_bed):</span>
              <span class="hud-value cyan" id="hud-bedrock-disp">0.00 μm</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Water Hammer (ΔP):</span>
              <span class="hud-value amber" id="hud-hammer-press">0.000 MPa</span>
            </div>
            <div class="hud-meter-track">
              <div class="hud-meter-fill cyan" id="hud-infrasound-bar" style="width: 0%;"></div>
            </div>
          </div>

          <div class="hud-group">
            <div class="hud-group-title">
              <span>2. Chemical Gas Dynamics (QC)</span>
              <span style="color: var(--emerald-accent); font-family: var(--font-mono);">Zn + 2HCl</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">H₂ Fraction QC (X_QC):</span>
              <span class="hud-value emerald" id="hud-h2-qc">0.00%</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">H₂ Fraction KC (X_KC):</span>
              <span class="hud-value text-muted" id="hud-h2-kc">0.00%</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Gas Sound Speed (c_mix):</span>
              <span class="hud-value gold" id="hud-sound-speed">343.2 m/s</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Chamber Temp (T_QC):</span>
              <span class="hud-value" id="hud-chem-temp">293.2 K</span>
            </div>
            <div class="hud-meter-track">
              <div class="hud-meter-fill emerald" id="hud-gas-bar" style="width: 0%;"></div>
            </div>
          </div>

          <div class="hud-group">
            <div class="hud-group-title">
              <span>3. Grand Gallery Acoustic Resonator</span>
              <span style="color: var(--gold-bright); font-family: var(--font-mono);">438 Hz (F#)</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Standing Wave Peak (p_GG):</span>
              <span class="hud-value gold" id="hud-gallery-peak">0.000 MPa</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Acoustic RMS (p_rms):</span>
              <span class="hud-value" id="hud-gallery-rms">0.0 kPa</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">F# Harmonic Purity:</span>
              <span class="hud-value cyan" id="hud-fsharp-purity">99.8%</span>
            </div>
            <div class="hud-meter-track">
              <div class="hud-meter-fill gold" id="hud-acoustic-bar" style="width: 0%;"></div>
            </div>
          </div>

          <div class="hud-group">
            <div class="hud-group-title">
              <span>4. King's Chamber Piezoelectric Array</span>
              <span style="color: var(--purple-piezo); font-family: var(--font-mono);">43 Beams</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Quartz Voltage (V_total):</span>
              <span class="hud-value purple" id="hud-piezo-volt">0.00 kV</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Max Beam Stress (σ_max):</span>
              <span class="hud-value rose" id="hud-beam-stress">0.00 MPa</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Dielectric Status:</span>
              <span class="hud-value" id="hud-spark-status" style="font-size: 0.68rem; color: #a3e635;">NOMINAL</span>
            </div>
            <div class="hud-meter-track">
              <div class="hud-meter-fill purple" id="hud-piezo-bar" style="width: 0%;"></div>
            </div>

            <div class="beam-matrix-section">
              <div class="beam-matrix-grid">${tierRowsHtml}</div>
              <div class="beam-matrix-legend">
                <span>0 MPa</span>
                <div class="beam-legend-bar"></div>
                <span>40+ MPa</span>
              </div>
              <div class="beam-matrix-tooltip" id="beam-matrix-tooltip"></div>
            </div>
          </div>

          <div class="hud-group">
            <div class="hud-group-title">
              <span>5. Stimulated Microwave Maser</span>
              <span style="color: var(--cyan-accent); font-family: var(--font-mono);">21.1 cm</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Carrier Frequency:</span>
              <span class="hud-value cyan" id="hud-maser-freq">1.4204 GHz</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Radiated Power (P_beam):</span>
              <span class="hud-value cyan" id="hud-maser-power">0.00 W</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Threshold Inversion:</span>
              <span class="hud-value" id="hud-maser-status" style="font-size: 0.68rem; color: var(--text-muted);">BELOW THRESHOLD</span>
            </div>
            <div class="hud-meter-track">
              <div class="hud-meter-fill cyan" id="hud-maser-bar" style="width: 0%;"></div>
            </div>
          </div>

          <div class="hud-group">
            <div class="hud-group-title">
              <span>6. Power Flow & System Efficiency</span>
              <span style="color: var(--amber-accent); font-family: var(--font-mono);">BALANCE</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Input Power (P_in):</span>
              <span class="hud-value amber" id="hud-power-in">0.00 MW</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Output Power (P_out):</span>
              <span class="hud-value cyan" id="hud-power-out">0.00 W</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Conversion Efficiency:</span>
              <span class="hud-value gold" id="hud-efficiency">0.00 %</span>
            </div>
            <div class="hud-metric-row">
              <span class="hud-label">Energy Conservation:</span>
              <span class="hud-value text-muted" id="hud-energy-balance">ΔE: 0.00%</span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.valBedrockDisp = this.container.querySelector('#hud-bedrock-disp');
    this.valHammerPress = this.container.querySelector('#hud-hammer-press');
    this.barInfrasoundFill = this.container.querySelector('#hud-infrasound-bar');

    this.valH2Qc = this.container.querySelector('#hud-h2-qc');
    this.valH2Kc = this.container.querySelector('#hud-h2-kc');
    this.valSoundSpeed = this.container.querySelector('#hud-sound-speed');
    this.valChemTemp = this.container.querySelector('#hud-chem-temp');
    this.barGasFill = this.container.querySelector('#hud-gas-bar');

    this.valGalleryPeak = this.container.querySelector('#hud-gallery-peak');
    this.valGalleryRms = this.container.querySelector('#hud-gallery-rms');
    this.valPurity = this.container.querySelector('#hud-fsharp-purity');
    this.barAcousticFill = this.container.querySelector('#hud-acoustic-bar');

    this.valPiezoVolt = this.container.querySelector('#hud-piezo-volt');
    this.valBeamStress = this.container.querySelector('#hud-beam-stress');
    this.badgeSpark = this.container.querySelector('#hud-spark-status');
    this.barPiezoFill = this.container.querySelector('#hud-piezo-bar');

    this.valMaserPower = this.container.querySelector('#hud-maser-power');
    this.badgeMaserStatus = this.container.querySelector('#hud-maser-status');
    this.barMaserFill = this.container.querySelector('#hud-maser-bar');

    this.valPowerIn = this.container.querySelector('#hud-power-in');
    this.valPowerOut = this.container.querySelector('#hud-power-out');
    this.valEfficiency = this.container.querySelector('#hud-efficiency');
    this.valEnergyBalance = this.container.querySelector('#hud-energy-balance');

    this.beamTooltip = this.container.querySelector('#beam-matrix-tooltip');

    const cellElements = Array.from(this.container.querySelectorAll('.beam-cell')) as HTMLElement[];
    this.beamCells = cellElements.map((el) => {
      const gIdx = parseInt(el.getAttribute('data-beam-idx') || '0', 10);
      const tIdx = parseInt(el.getAttribute('data-tier') || '0', 10);
      const bIdx = parseInt(el.getAttribute('data-tier-idx') || '0', 10);
      const tierName = TIER_METADATA[tIdx]?.fullName || 'Relieving Tier';

      el.addEventListener('mouseenter', () => {
        this.hoveredBeamIndex = gIdx;
        this.updateTooltip(gIdx);
      });

      el.addEventListener('mouseleave', () => {
        this.hoveredBeamIndex = null;
        if (this.beamTooltip) this.beamTooltip.style.display = 'none';
      });

      return {
        tierIndex: tIdx,
        beamInTier: bIdx,
        globalIndex: gIdx,
        tierName,
        element: el,
      };
    });

    const collapseBtn = this.container.querySelector('#btn-inspector-collapse');
    if (collapseBtn) {
      collapseBtn.addEventListener('click', () => {
        this.toggleCollapsed();
      });
    }
  }

  public toggleCollapsed(): void {
    this.isCollapsed = !this.isCollapsed;
    const inner = this.container.querySelector('#inspector-panel-inner') as HTMLElement;
    if (inner) {
      inner.style.display = this.isCollapsed ? 'none' : 'flex';
    }
  }

  public isVisible(): boolean {
    return !this.isCollapsed;
  }

  public show(): void {
    this.isCollapsed = false;
    const inner = this.container.querySelector('#inspector-panel-inner') as HTMLElement;
    if (inner) inner.style.display = 'flex';
  }

  public hide(): void {
    this.isCollapsed = true;
    const inner = this.container.querySelector('#inspector-panel-inner') as HTMLElement;
    if (inner) inner.style.display = 'none';
  }

  private getStressColor(stressMpa: number): string {
    const s = Math.max(0, stressMpa);
    if (s <= 5.0) {
      const t = s / 5.0;
      const r = Math.round(14 + t * (56 - 14));
      const g = Math.round(165 + t * (189 - 165));
      const b = Math.round(233 + t * (248 - 233));
      return `rgba(${r}, ${g}, ${b}, 0.55)`;
    } else if (s <= 15.0) {
      const t = (s - 5.0) / 10.0;
      const r = Math.round(56 + t * (192 - 56));
      const g = Math.round(189 - t * (189 - 132));
      const b = Math.round(248 + t * (252 - 248));
      return `rgba(${r}, ${g}, ${b}, 0.72)`;
    } else if (s <= 30.0) {
      const t = (s - 15.0) / 15.0;
      const r = Math.round(192 + t * (245 - 192));
      const g = Math.round(132 + t * (158 - 132));
      const b = Math.round(252 - t * (252 - 11));
      return `rgba(${r}, ${g}, ${b}, 0.88)`;
    } else {
      const t = Math.min(1.0, (s - 30.0) / 20.0);
      const r = Math.round(245 + t * (255 - 245));
      const g = Math.round(158 - t * (158 - 50));
      const b = Math.round(11 - t * 11);
      return `rgba(${r}, ${g}, ${b}, 0.95)`;
    }
  }

  private updateTooltip(globalIdx: number): void {
    if (!this.beamTooltip || !this.latestFrame) return;
    const beam = this.beamCells[globalIdx];
    if (!beam) return;

    const frame = this.latestFrame;
    const stresses = frame.spatial?.all_beam_stresses_mpa;
    const voltages = frame.spatial?.all_beam_voltages_v;

    let stress = 0;
    let voltage = 0;

    if (stresses && stresses.length > globalIdx) {
      stress = stresses[globalIdx];
    } else {
      const tierWeight = [1.0, 0.88, 0.76, 0.65, 0.52][beam.tierIndex] ?? 0.7;
      const maxS = (frame.max_beam_stress_pa || 0) / 1e6;
      stress = maxS * tierWeight;
    }

    if (voltages && voltages.length > globalIdx) {
      voltage = voltages[globalIdx];
    } else {
      const tierWeight = [1.0, 0.88, 0.76, 0.65, 0.52][beam.tierIndex] ?? 0.7;
      voltage = (frame.total_piezo_voltage / 43) * tierWeight;
    }

    const iDispMa = Math.abs(voltage) * 2.0 * Math.PI * 438.0 * 18.5e-12 * 1e3;

    this.beamTooltip.innerHTML = `
      <div style="color: var(--gold-bright); font-weight: 700; font-size: 0.72rem;">Beam #${globalIdx + 1}: ${beam.tierName} (#${beam.beamInTier + 1})</div>
      <div>Bending Stress: <span style="color: var(--rose-granite); font-weight: 600;">${stress.toFixed(2)} MPa</span></div>
      <div>Piezo Potential: <span style="color: var(--purple-piezo); font-weight: 600;">${Math.abs(voltage) >= 1000 ? (voltage / 1e3).toFixed(2) + ' kV' : voltage.toFixed(1) + ' V'}</span></div>
      <div>Disp Current: <span style="color: var(--cyan-accent); font-weight: 600;">${iDispMa.toFixed(2)} mA</span></div>
    `;
    this.beamTooltip.style.display = 'flex';
  }

  public update(frame: TelemetryFrame | null): void {
    if (!frame || this.isCollapsed) return;
    this.latestFrame = frame;

    if (this.valBedrockDisp) {
      this.valBedrockDisp.textContent = `${(frame.bedrock_displacement * 1e6).toFixed(2)} μm`;
    }
    if (this.valHammerPress) {
      this.valHammerPress.textContent = `${(frame.water_hammer_pressure / 1e6).toFixed(3)} MPa`;
    }
    if (this.barInfrasoundFill) {
      const pct = Math.min(100, Math.max(0, (frame.water_hammer_pressure / 2.5e6) * 100));
      this.barInfrasoundFill.style.width = `${pct}%`;
    }

    if (this.valH2Qc) {
      this.valH2Qc.textContent = `${(frame.h2_mole_fraction_qc * 100).toFixed(3)}%`;
    }
    if (this.valH2Kc) {
      this.valH2Kc.textContent = `${(frame.h2_mole_fraction_kc * 100).toFixed(6)}%`;
    }
    if (this.valSoundSpeed) {
      this.valSoundSpeed.textContent = `${frame.gallery_sound_speed_avg.toFixed(1)} m/s`;
    }
    if (this.valChemTemp) {
      this.valChemTemp.textContent = `${frame.qc_chamber_temperature_k.toFixed(1)} K`;
    }
    if (this.barGasFill) {
      const pct = Math.min(100, Math.max(0, (frame.h2_mole_fraction_qc / 0.05) * 100));
      this.barGasFill.style.width = `${pct}%`;
    }

    if (this.valGalleryPeak) {
      this.valGalleryPeak.textContent = `${(frame.gallery_peak_pressure / 1e6).toFixed(3)} MPa`;
    }
    if (this.valGalleryRms) {
      this.valGalleryRms.textContent = `${(frame.gallery_rms_pressure / 1e3).toFixed(1)} kPa`;
    }
    if (this.valPurity) {
      this.valPurity.textContent = `${Math.min(100, (frame.f_sharp_spectral_purity * 100 || 99.8)).toFixed(1)}%`;
    }
    if (this.barAcousticFill) {
      const pct = Math.min(100, Math.max(0, (frame.gallery_peak_pressure / 5.5e6) * 100));
      this.barAcousticFill.style.width = `${pct}%`;
    }

    if (this.valPiezoVolt) {
      this.valPiezoVolt.textContent = `${(frame.total_piezo_voltage / 1e3).toFixed(2)} kV`;
    }
    if (this.valBeamStress) {
      this.valBeamStress.textContent = `${(frame.max_beam_stress_pa / 1e6).toFixed(2)} MPa`;
    }
    if (this.badgeSpark) {
      if (frame.spark_triggered) {
        this.badgeSpark.textContent = '⚡ SPARK DISCHARGE';
        this.badgeSpark.style.color = 'var(--gold-bright)';
      } else if (frame.total_piezo_voltage > 10000) {
        this.badgeSpark.textContent = 'CORONA IONIZATION';
        this.badgeSpark.style.color = 'var(--purple-piezo)';
      } else {
        this.badgeSpark.textContent = 'NOMINAL';
        this.badgeSpark.style.color = '#a3e635';
      }
    }
    if (this.barPiezoFill) {
      const pct = Math.min(100, Math.max(0, (frame.total_piezo_voltage / 25000) * 100));
      this.barPiezoFill.style.width = `${pct}%`;
    }

    const stresses = frame.spatial?.all_beam_stresses_mpa;
    const maxS = (frame.max_beam_stress_pa || 0) / 1e6;
    for (let i = 0; i < this.beamCells.length; i++) {
      const beam = this.beamCells[i];
      let stress = 0;
      if (stresses && stresses.length > i) {
        stress = stresses[i];
      } else {
        const tierWeight = [1.0, 0.88, 0.76, 0.65, 0.52][beam.tierIndex] ?? 0.7;
        stress = maxS * tierWeight;
      }
      beam.element.style.backgroundColor = this.getStressColor(stress);
    }

    if (this.hoveredBeamIndex !== null) {
      this.updateTooltip(this.hoveredBeamIndex);
    }

    if (this.valMaserPower) {
      if (frame.maser_total_radiated_power >= 0.001) {
        this.valMaserPower.textContent = `${frame.maser_total_radiated_power.toFixed(2)} W`;
      } else {
        this.valMaserPower.textContent = `${(frame.maser_total_radiated_power * 1e6).toFixed(2)} μW`;
      }
    }
    if (this.badgeMaserStatus) {
      if (frame.maser_is_above_threshold || frame.maser_total_radiated_power > 1e-10) {
        this.badgeMaserStatus.textContent = 'STIMULATED EMISSION';
        this.badgeMaserStatus.style.color = 'var(--cyan-accent)';
      } else {
        this.badgeMaserStatus.textContent = 'BELOW THRESHOLD';
        this.badgeMaserStatus.style.color = 'var(--text-muted)';
      }
    }
    if (this.barMaserFill) {
      const pct = Math.min(100, Math.max(0, frame.maser_is_above_threshold ? 100 : (frame.maser_total_radiated_power / 100) * 100));
      this.barMaserFill.style.width = `${pct}%`;
    }

    if (this.valPowerIn) {
      this.valPowerIn.textContent = `${(frame.p_total_in / 1e6).toFixed(2)} MW`;
    }
    if (this.valPowerOut) {
      this.valPowerOut.textContent = frame.maser_total_radiated_power >= 0.001
        ? `${frame.maser_total_radiated_power.toFixed(2)} W`
        : `${(frame.maser_total_radiated_power * 1e6).toFixed(2)} μW`;
    }
    if (this.valEfficiency) {
      const eff = frame.p_total_in > 0 ? (frame.p_total_out / frame.p_total_in) * 100 : 0;
      this.valEfficiency.textContent = `${eff.toExponential(2)} %`;
    }
    if (this.valEnergyBalance) {
      const errPct = (frame.relative_energy_error * 100).toFixed(2);
      this.valEnergyBalance.textContent = `ΔE: ${errPct}%`;
      this.valEnergyBalance.style.color = frame.is_energy_conserved ? '#a3e635' : 'var(--text-muted)';
    }
  }
}
