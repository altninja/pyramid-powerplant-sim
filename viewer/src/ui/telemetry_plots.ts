import { TelemetryFrame } from '../data/telemetry_loader';

export type PlotType = 'seismic' | 'acoustic' | 'fft' | 'piezo';

interface PlotConfig {
  id: PlotType;
  title: string;
  unitLeft: string;
  unitRight?: string;
  color1: string;
  color2?: string;
  label1: string;
  label2?: string;
}

const PLOT_CONFIGS: Record<PlotType, PlotConfig> = {
  seismic: {
    id: 'seismic',
    title: 'SEISMIC & HYDRAULIC INFRASOUND',
    unitLeft: 'x_bed (μm)',
    unitRight: 'ΔP (MPa)',
    color1: '#00f2fe',
    color2: '#ff9f43',
    label1: 'Bedrock Disp',
    label2: 'Water Hammer',
  },
  acoustic: {
    id: 'acoustic',
    title: 'ACOUSTIC STANDING WAVE PRESSURE',
    unitLeft: 'p_GG (MPa)',
    unitRight: 'p_KC (kPa)',
    color1: '#38bdf8',
    color2: '#fb7185',
    label1: 'Grand Gallery',
    label2: "King's Chamber",
  },
  fft: {
    id: 'fft',
    title: 'FFT HARMONIC SPECTRUM ANALYZER',
    unitLeft: 'Magnitude (dB)',
    color1: '#ffd700',
    color2: '#00f2fe',
    label1: 'Acoustic / Infrasonic Harmonics',
    label2: '1.4204 GHz Maser Line',
  },
  piezo: {
    id: 'piezo',
    title: 'PIEZOELECTRIC VOLTAGE & MASER BEAM POWER',
    unitLeft: 'V_piezo (kV)',
    unitRight: 'P_maser (W)',
    color1: '#c084fc',
    color2: '#10b981',
    label1: 'Quartz Voltage',
    label2: 'Maser Beam Power',
  },
};

export class TelemetryPlots {
  private container: HTMLElement;
  private canvases: Map<PlotType, HTMLCanvasElement> = new Map();
  private contexts: Map<PlotType, CanvasRenderingContext2D> = new Map();
  private readoutElements: Map<PlotType, HTMLElement> = new Map();

  private activeTab: 'all' | PlotType = 'all';
  private isCollapsed: boolean = false;
  private gridContainer: HTMLElement | null = null;
  private dpr: number = 1;

  private cachedFrames: TelemetryFrame[] = [];
  private currentTime: number = 0;
  private currentFrame: TelemetryFrame | null = null;
  private animFrameId: number | null = null;
  private isDirty: boolean = true;

  constructor(container: HTMLElement) {
    this.container = container;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.initDOM();
    this.setupResizeObserver();
  }

  private initDOM(): void {
    this.container.innerHTML = `
      <div class="plots-header">
        <div class="plots-tabs">
          <button class="plots-tab active" data-tab="all">Quad Oscilloscopes</button>
          <button class="plots-tab" data-tab="seismic">Seismic/Hydraulic</button>
          <button class="plots-tab" data-tab="acoustic">Acoustic Standing Wave</button>
          <button class="plots-tab" data-tab="fft">FFT Spectrum</button>
          <button class="plots-tab" data-tab="piezo">Piezo/Maser</button>
        </div>
        <div style="display: flex; gap: 6px;">
          <button class="btn btn-icon" id="btn-plots-collapse" title="Toggle Oscilloscopes [O]">✕</button>
        </div>
      </div>
      <div class="plots-grid" id="plots-grid-container">
        ${(['seismic', 'acoustic', 'fft', 'piezo'] as PlotType[])
          .map(
            (type) => `
          <div class="plot-card" id="plot-card-${type}">
            <div class="plot-card-header">
              <span class="plot-title">${PLOT_CONFIGS[type].title}</span>
              <span class="plot-readout" id="plot-readout-${type}">--</span>
            </div>
            <canvas class="plot-canvas" id="plot-canvas-${type}"></canvas>
          </div>
        `
          )
          .join('')}
      </div>
    `;

    this.gridContainer = this.container.querySelector('#plots-grid-container');

    const types: PlotType[] = ['seismic', 'acoustic', 'fft', 'piezo'];
    types.forEach((type) => {
      const canvas = this.container.querySelector(`#plot-canvas-${type}`) as HTMLCanvasElement;
      if (canvas) {
        this.canvases.set(type, canvas);
        const ctx = canvas.getContext('2d', { alpha: false });
        if (ctx) this.contexts.set(type, ctx);
      }
      const readout = this.container.querySelector(`#plot-readout-${type}`) as HTMLElement;
      if (readout) {
        this.readoutElements.set(type, readout);
      }
    });

    const tabs = this.container.querySelectorAll('.plots-tab');
    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const tabVal = tab.getAttribute('data-tab') as 'all' | PlotType;
        if (tabVal) {
          this.setActiveTab(tabVal);
        }
      });
    });

    const collapseBtn = this.container.querySelector('#btn-plots-collapse');
    if (collapseBtn) {
      collapseBtn.addEventListener('click', () => {
        this.toggleCollapsed();
      });
    }

    this.resizeCanvases();
  }

  public setActiveTab(tabId: 'all' | PlotType): void {
    this.activeTab = tabId;

    const tabs = this.container.querySelectorAll('.plots-tab');
    tabs.forEach((t) => {
      if (t.getAttribute('data-tab') === tabId) {
        t.classList.add('active');
      } else {
        t.classList.remove('active');
      }
    });

    const types: PlotType[] = ['seismic', 'acoustic', 'fft', 'piezo'];
    if (this.gridContainer) {
      if (tabId === 'all') {
        this.gridContainer.classList.remove('single-view');
        types.forEach((t) => {
          const card = this.container.querySelector(`#plot-card-${t}`) as HTMLElement;
          if (card) card.style.display = 'flex';
        });
      } else {
        this.gridContainer.classList.add('single-view');
        types.forEach((t) => {
          const card = this.container.querySelector(`#plot-card-${t}`) as HTMLElement;
          if (card) card.style.display = t === tabId ? 'flex' : 'none';
        });
      }
    }

    this.resizeCanvases();
    this.requestRender();
  }

  public toggleCollapsed(): void {
    this.isCollapsed = !this.isCollapsed;
    if (this.isCollapsed) {
      this.container.classList.add('collapsed');
    } else {
      this.container.classList.remove('collapsed');
      this.resizeCanvases();
      this.requestRender();
    }
  }

  public isVisible(): boolean {
    return !this.isCollapsed;
  }

  public show(): void {
    this.isCollapsed = false;
    this.container.classList.remove('collapsed');
    this.resizeCanvases();
    this.requestRender();
  }

  public hide(): void {
    this.isCollapsed = true;
    this.container.classList.add('collapsed');
  }

  private setupResizeObserver(): void {
    const observer = new ResizeObserver(() => {
      this.resizeCanvases();
      this.requestRender();
    });
    observer.observe(this.container);
  }

  private resizeCanvases(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvases.forEach((canvas) => {
      const rect = canvas.getBoundingClientRect();
      const w = Math.floor(rect.width);
      const h = Math.floor(rect.height);
      if (w > 0 && h > 0) {
        if (canvas.width !== Math.floor(w * this.dpr) || canvas.height !== Math.floor(h * this.dpr)) {
          canvas.width = Math.floor(w * this.dpr);
          canvas.height = Math.floor(h * this.dpr);
        }
      }
    });
    this.isDirty = true;
  }

  public update(currentTime: number, currentFrame: TelemetryFrame | null, allFrames: TelemetryFrame[]): void {
    this.currentTime = currentTime;
    this.currentFrame = currentFrame;
    this.cachedFrames = allFrames;
    this.isDirty = true;
    this.requestRender();
  }

  public requestRender(): void {
    if (!this.isDirty || this.isCollapsed) return;
    if (this.animFrameId !== null) return;

    this.animFrameId = requestAnimationFrame(() => {
      this.animFrameId = null;
      this.render();
    });
  }

  private render(): void {
    this.isDirty = false;
    if (this.isCollapsed) return;

    if (this.activeTab === 'all' || this.activeTab === 'seismic') {
      this.renderSeismicPlot();
    }
    if (this.activeTab === 'all' || this.activeTab === 'acoustic') {
      this.renderAcousticPlot();
    }
    if (this.activeTab === 'all' || this.activeTab === 'fft') {
      this.renderFftPlot();
    }
    if (this.activeTab === 'all' || this.activeTab === 'piezo') {
      this.renderPiezoPlot();
    }
  }

  private renderSeismicPlot(): void {
    const ctx = this.contexts.get('seismic');
    const canvas = this.canvases.get('seismic');
    if (!ctx || !canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 24 * this.dpr, bottom: 20 * this.dpr, left: 40 * this.dpr, right: 40 * this.dpr };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    this.drawBackgroundAndGrid(ctx, w, h, pad, 't (s)', 'μm', 'MPa');

    if (this.cachedFrames.length === 0 || !this.currentFrame) return;

    const frames = this.cachedFrames;
    const duration = frames[frames.length - 1].time || 3.0;

    let maxDisp = 1e-6;
    let maxPress = 1e5;
    for (let i = 0; i < frames.length; i++) {
      const d = Math.abs(frames[i].bedrock_displacement);
      const p = Math.abs(frames[i].water_hammer_pressure);
      if (d > maxDisp) maxDisp = d;
      if (p > maxPress) maxPress = p;
    }
    maxDisp = Math.max(maxDisp, 1e-5);
    maxPress = Math.max(maxPress, 1e5);

    const curDisp = (this.currentFrame.bedrock_displacement * 1e6).toFixed(2);
    const curPress = (this.currentFrame.water_hammer_pressure / 1e6).toFixed(3);
    const readout = this.readoutElements.get('seismic');
    if (readout) {
      readout.innerHTML = `<span style="color:#00f2fe">x: ${curDisp} μm</span> | <span style="color:#ff9f43">ΔP: ${curPress} MPa</span>`;
    }

    this.drawTimeSeriesLine(
      ctx,
      frames,
      duration,
      (f) => f.bedrock_displacement,
      -maxDisp * 1.1,
      maxDisp * 1.1,
      pad,
      plotW,
      plotH,
      '#00f2fe',
      'rgba(0, 242, 254, 0.12)'
    );

    this.drawTimeSeriesLine(
      ctx,
      frames,
      duration,
      (f) => f.water_hammer_pressure,
      0,
      maxPress * 1.15,
      pad,
      plotW,
      plotH,
      '#ff9f43',
      'rgba(255, 159, 67, 0.12)'
    );

    this.drawPlayheadMarker(ctx, this.currentTime, duration, pad, plotW, plotH);
  }

  private renderAcousticPlot(): void {
    const ctx = this.contexts.get('acoustic');
    const canvas = this.canvases.get('acoustic');
    if (!ctx || !canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 24 * this.dpr, bottom: 20 * this.dpr, left: 42 * this.dpr, right: 42 * this.dpr };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    this.drawBackgroundAndGrid(ctx, w, h, pad, 't (s)', 'MPa', 'kPa');

    if (this.cachedFrames.length === 0 || !this.currentFrame) return;

    const frames = this.cachedFrames;
    const duration = frames[frames.length - 1].time || 3.0;

    let maxGG = 1000;
    let maxKC = 100;
    for (let i = 0; i < frames.length; i++) {
      if (frames[i].gallery_peak_pressure > maxGG) maxGG = frames[i].gallery_peak_pressure;
      if (frames[i].top_pressure_kc_entry > maxKC) maxKC = frames[i].top_pressure_kc_entry;
      if (frames[i].acoustic_pressure_sub > maxKC) maxKC = Math.max(maxKC, frames[i].acoustic_pressure_sub);
    }

    const curGG = (this.currentFrame.gallery_peak_pressure / 1e6).toFixed(3);
    const curKC = ((this.currentFrame.top_pressure_kc_entry || this.currentFrame.acoustic_pressure_sub) / 1e3).toFixed(1);
    const readout = this.readoutElements.get('acoustic');
    if (readout) {
      readout.innerHTML = `<span style="color:#38bdf8">p_GG: ${curGG} MPa</span> | <span style="color:#fb7185">p_KC: ${curKC} kPa</span>`;
    }

    this.drawTimeSeriesLine(
      ctx,
      frames,
      duration,
      (f) => f.gallery_peak_pressure,
      0,
      maxGG * 1.15,
      pad,
      plotW,
      plotH,
      '#38bdf8',
      'rgba(56, 189, 248, 0.14)'
    );

    this.drawTimeSeriesLine(
      ctx,
      frames,
      duration,
      (f) => f.top_pressure_kc_entry || f.acoustic_pressure_sub,
      0,
      maxKC * 1.15,
      pad,
      plotW,
      plotH,
      '#fb7185',
      'rgba(251, 113, 133, 0.14)'
    );

    this.drawPlayheadMarker(ctx, this.currentTime, duration, pad, plotW, plotH);
  }

  private renderFftPlot(): void {
    const ctx = this.contexts.get('fft');
    const canvas = this.canvases.get('fft');
    if (!ctx || !canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 24 * this.dpr, bottom: 22 * this.dpr, left: 40 * this.dpr, right: 24 * this.dpr };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1 * this.dpr;
    const gridRows = 4;
    for (let r = 0; r <= gridRows; r++) {
      const y = pad.top + (r / gridRows) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();

      const dbVal = (-(r / gridRows) * 80).toFixed(0);
      ctx.fillStyle = '#64748b';
      ctx.font = `${7.5 * this.dpr}px JetBrains Mono, monospace`;
      ctx.textAlign = 'right';
      ctx.fillText(`${dbVal} dB`, pad.left - 4 * this.dpr, y + 3 * this.dpr);
    }

    const maxFreqHz = 1600;
    const freqGridPoints = [0, 250, 500, 750, 1000, 1250, 1500];
    for (const fg of freqGridPoints) {
      const x = pad.left + (fg / maxFreqHz) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = `${7 * this.dpr}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(`${fg}Hz`, x, pad.top + plotH + 12 * this.dpr);
    }

    if (!this.currentFrame) return;

    const f = this.currentFrame;
    const schumannAmp = Math.min(1.0, Math.max(0.1, Math.abs(f.schumann_excitation) * 0.8 + 0.15));
    const fSharpAmp = Math.min(1.0, Math.max(0.08, (f.gallery_peak_pressure / 5.5e6) * (f.f_sharp_spectral_purity || 0.85)));
    const fSharp2nd = fSharpAmp * 0.38;
    const fSharp3rd = fSharpAmp * 0.18;
    const maserAmp = f.maser_is_above_threshold || f.maser_total_radiated_power > 1e-12 ? 1.0 : 0.08;

    const rawFreqs = f.spatial?.fft_frequencies_hz;
    const rawPsd = f.spatial?.fft_power_spectral_density_db;
    const hasTelemetryFft = rawFreqs && rawPsd && rawFreqs.length > 2 && rawPsd.length > 2;

    let points: Array<{ freq: number; db: number }> = [];

    if (hasTelemetryFft && rawFreqs && rawPsd) {
      const len = Math.min(rawFreqs.length, rawPsd.length);
      for (let i = 0; i < len; i++) {
        if (rawFreqs[i] <= maxFreqHz) {
          points.push({ freq: rawFreqs[i], db: Math.max(-80, Math.min(0, rawPsd[i])) });
        }
      }
    } else {
      const numSamples = 160;
      for (let i = 0; i <= numSamples; i++) {
        const freq = (i / numSamples) * maxFreqHz;
        const noise = -68.0 + Math.sin(freq * 0.3) * 2.0 + Math.cos(freq * 0.7) * 1.5;

        const peakSchumann = Math.exp(-Math.pow((freq - 7.83) / 1.8, 2)) * (60.0 * schumannAmp);
        const peakSchumann2 = Math.exp(-Math.pow((freq - 14.3) / 2.2, 2)) * (36.0 * schumannAmp);
        const peakFSharp1 = Math.exp(-Math.pow((freq - 438.0) / 4.5, 2)) * (64.0 * fSharpAmp);
        const peakFSharp2 = Math.exp(-Math.pow((freq - 876.0) / 6.0, 2)) * (48.0 * fSharp2nd);
        const peakFSharp3 = Math.exp(-Math.pow((freq - 1314.0) / 7.5, 2)) * (35.0 * fSharp3rd);

        const totalDb = Math.min(0, noise + peakSchumann + peakSchumann2 + peakFSharp1 + peakFSharp2 + peakFSharp3);
        points.push({ freq, db: totalDb });
      }
    }

    if (points.length > 1) {
      ctx.beginPath();
      const firstX = pad.left + (points[0].freq / maxFreqHz) * plotW;
      const firstY = pad.top + ((0 - points[0].db) / 80) * plotH;
      ctx.moveTo(firstX, firstY);

      for (let i = 1; i < points.length; i++) {
        const px = pad.left + (points[i].freq / maxFreqHz) * plotW;
        const py = pad.top + ((0 - points[i].db) / 80) * plotH;
        ctx.lineTo(px, py);
      }

      const lastX = pad.left + (points[points.length - 1].freq / maxFreqHz) * plotW;
      ctx.lineTo(lastX, pad.top + plotH);
      ctx.lineTo(firstX, pad.top + plotH);
      ctx.closePath();

      const areaGrad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
      areaGrad.addColorStop(0, 'rgba(255, 215, 0, 0.28)');
      areaGrad.addColorStop(0.5, 'rgba(0, 242, 254, 0.12)');
      areaGrad.addColorStop(1, 'rgba(6, 9, 16, 0.0)');
      ctx.fillStyle = areaGrad;
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(firstX, firstY);
      for (let i = 1; i < points.length; i++) {
        const px = pad.left + (points[i].freq / maxFreqHz) * plotW;
        const py = pad.top + ((0 - points[i].db) / 80) * plotH;
        ctx.lineTo(px, py);
      }
      ctx.strokeStyle = '#ffd700';
      ctx.lineWidth = 1.6 * this.dpr;
      ctx.stroke();
    }

    const annotatedPeaks = [
      { freq: 7.83, label: '7.83 Hz', sub: 'Schumann', col: '#00f2fe', mag: schumannAmp },
      { freq: 14.3, label: '14.3 Hz', sub: '2nd Mode', col: '#00c2de', mag: schumannAmp * 0.5 },
      { freq: 438.0, label: '438 Hz', sub: 'F# (1st)', col: '#ffd700', mag: fSharpAmp },
      { freq: 876.0, label: '876 Hz', sub: 'F# (2nd)', col: '#e5c07b', mag: fSharp2nd },
      { freq: 1314.0, label: '1314 Hz', sub: 'F# (3rd)', col: '#fbbf24', mag: fSharp3rd },
    ];

    annotatedPeaks.forEach((pk) => {
      const px = pad.left + (pk.freq / maxFreqHz) * plotW;
      const peakDb = Math.max(-70, -6.0 - (1.0 - pk.mag) * 45.0);
      const py = pad.top + ((0 - peakDb) / 80) * plotH;

      ctx.beginPath();
      ctx.setLineDash([2 * this.dpr, 2 * this.dpr]);
      ctx.moveTo(px, pad.top + plotH);
      ctx.lineTo(px, py);
      ctx.strokeStyle = pk.col;
      ctx.lineWidth = 1 * this.dpr;
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(px, py, 3.5 * this.dpr, 0, Math.PI * 2);
      ctx.fillStyle = pk.col;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1 * this.dpr;
      ctx.stroke();

      ctx.fillStyle = pk.col;
      ctx.font = `bold ${7.5 * this.dpr}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(pk.label, px, Math.max(pad.top + 8 * this.dpr, py - 6 * this.dpr));
    });

    const maserX = pad.left + plotW - 68 * this.dpr;
    const maserY = pad.top + 14 * this.dpr;
    ctx.fillStyle = 'rgba(11, 15, 25, 0.85)';
    ctx.strokeStyle = maserAmp > 0.5 ? '#c084fc' : 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1 * this.dpr;
    ctx.beginPath();
    ctx.roundRect(maserX - 4 * this.dpr, maserY - 10 * this.dpr, 70 * this.dpr, 16 * this.dpr, 3 * this.dpr);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = maserAmp > 0.5 ? '#c084fc' : '#94a3b8';
    ctx.font = `bold ${7 * this.dpr}px JetBrains Mono, monospace`;
    ctx.textAlign = 'left';
    ctx.fillText('1.42GHz MASER', maserX, maserY);

    const readout = this.readoutElements.get('fft');
    if (readout) {
      const fSharpPct = (fSharpAmp * 100).toFixed(0);
      const maserStatus = f.maser_is_above_threshold ? 'ACTIVE' : 'SUB-TH';
      readout.innerHTML = `<span style="color:#ffd700">F# (438Hz): ${fSharpPct}%</span> | <span style="color:#00f2fe">Schumann: 7.83Hz</span> | <span style="color:#c084fc">Maser: ${maserStatus}</span>`;
    }
  }

  private renderPiezoPlot(): void {
    const ctx = this.contexts.get('piezo');
    const canvas = this.canvases.get('piezo');
    if (!ctx || !canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 24 * this.dpr, bottom: 20 * this.dpr, left: 40 * this.dpr, right: 40 * this.dpr };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    this.drawBackgroundAndGrid(ctx, w, h, pad, 't (s)', 'kV', 'W');

    if (this.cachedFrames.length === 0 || !this.currentFrame) return;

    const frames = this.cachedFrames;
    const duration = frames[frames.length - 1].time || 3.0;

    let maxV = 1000;
    let maxP = 1e-3;
    for (let i = 0; i < frames.length; i++) {
      if (frames[i].total_piezo_voltage > maxV) maxV = frames[i].total_piezo_voltage;
      if (frames[i].maser_total_radiated_power > maxP) maxP = frames[i].maser_total_radiated_power;
    }

    const curV = (this.currentFrame.total_piezo_voltage / 1e3).toFixed(2);
    const curP = this.currentFrame.maser_total_radiated_power >= 0.001
      ? `${this.currentFrame.maser_total_radiated_power.toFixed(2)} W`
      : `${(this.currentFrame.maser_total_radiated_power * 1e6).toFixed(1)} μW`;

    const readout = this.readoutElements.get('piezo');
    if (readout) {
      readout.innerHTML = `<span style="color:#c084fc">V_piezo: ${curV} kV</span> | <span style="color:#10b981">P_maser: ${curP}</span>`;
    }

    this.drawTimeSeriesLine(
      ctx,
      frames,
      duration,
      (f) => f.total_piezo_voltage,
      0,
      maxV * 1.15,
      pad,
      plotW,
      plotH,
      '#c084fc',
      'rgba(192, 132, 252, 0.15)'
    );

    this.drawTimeSeriesLine(
      ctx,
      frames,
      duration,
      (f) => f.maser_total_radiated_power,
      0,
      maxP * 1.15,
      pad,
      plotW,
      plotH,
      '#10b981',
      'rgba(16, 185, 129, 0.15)'
    );

    this.drawPlayheadMarker(ctx, this.currentTime, duration, pad, plotW, plotH);
  }

  private drawBackgroundAndGrid(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    pad: { top: number; bottom: number; left: number; right: number },
    xUnit: string,
    yUnitL: string,
    yUnitR?: string
  ): void {
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, w, h);

    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1 * this.dpr;

    const xGridCount = 5;
    for (let i = 0; i <= xGridCount; i++) {
      const x = pad.left + (i / xGridCount) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + plotH);
      ctx.stroke();
    }

    const yGridCount = 4;
    for (let j = 0; j <= yGridCount; j++) {
      const y = pad.top + (j / yGridCount) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    ctx.fillStyle = '#64748b';
    ctx.font = `${7 * this.dpr}px JetBrains Mono, monospace`;
    ctx.textAlign = 'left';
    ctx.fillText(yUnitL, 4 * this.dpr, pad.top - 4 * this.dpr);

    if (yUnitR) {
      ctx.textAlign = 'right';
      ctx.fillText(yUnitR, w - 4 * this.dpr, pad.top - 4 * this.dpr);
    }

    ctx.textAlign = 'center';
    ctx.fillText(xUnit, pad.left + plotW / 2, h - 4 * this.dpr);
  }

  private drawTimeSeriesLine(
    ctx: CanvasRenderingContext2D,
    frames: TelemetryFrame[],
    duration: number,
    valFn: (f: TelemetryFrame) => number,
    minVal: number,
    maxVal: number,
    pad: { top: number; bottom: number; left: number; right: number },
    plotW: number,
    plotH: number,
    strokeColor: string,
    fillColor?: string
  ): void {
    if (frames.length < 2) return;
    const range = Math.max(1e-12, maxVal - minVal);

    ctx.beginPath();
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i];
      const normX = Math.max(0, Math.min(1, f.time / duration));
      const x = pad.left + normX * plotW;
      const v = valFn(f);
      const normY = Math.max(0, Math.min(1, (v - minVal) / range));
      const y = pad.top + plotH - normY * plotH;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.6 * this.dpr;
    ctx.stroke();

    if (fillColor) {
      const lastX = pad.left + Math.min(1, frames[frames.length - 1].time / duration) * plotW;
      const firstX = pad.left + Math.min(1, frames[0].time / duration) * plotW;
      const zeroNorm = Math.max(0, Math.min(1, (0 - minVal) / range));
      const zeroY = pad.top + plotH - zeroNorm * plotH;

      ctx.lineTo(lastX, zeroY);
      ctx.lineTo(firstX, zeroY);
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
    }
  }

  private drawPlayheadMarker(
    ctx: CanvasRenderingContext2D,
    currentTime: number,
    duration: number,
    pad: { top: number; bottom: number; left: number; right: number },
    plotW: number,
    plotH: number
  ): void {
    if (duration <= 0) return;
    const normX = Math.max(0, Math.min(1, currentTime / duration));
    const px = pad.left + normX * plotW;

    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 1.5 * this.dpr;
    ctx.setLineDash([3 * this.dpr, 3 * this.dpr]);
    ctx.beginPath();
    ctx.moveTo(px, pad.top);
    ctx.lineTo(px, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#ffd700';
    ctx.beginPath();
    ctx.arc(px, pad.top + plotH, 3 * this.dpr, 0, Math.PI * 2);
    ctx.fill();
  }
}
