export interface TelemetrySpatial {
  gallery_z?: number[];
  acoustic_pressure_profile?: number[];
  acoustic_velocity_profile?: number[];
  acoustic_energy_density?: number[];
  gas_nodes?: string[];
  gas_h2_mole_fractions?: number[];
  gas_sound_speeds?: number[];
  gas_densities?: number[];
  tier_voltages?: number[];
  all_beam_stresses_mpa?: number[];
  all_beam_voltages_v?: number[];
  fft_frequencies_hz?: number[];
  fft_power_spectral_density_db?: number[];
  north_shaft_power?: number;
  south_shaft_power?: number;
}

export interface TelemetryFrame {
  time: number;
  step_index: number;
  bedrock_displacement: number;
  bedrock_velocity: number;
  bedrock_acceleration: number;
  water_hammer_pressure: number;
  seismic_force: number;
  hydraulic_force: number;
  schumann_excitation: number;
  acoustic_pressure_sub: number;
  h2_mole_fraction_qc: number;
  h2_mole_fraction_kc: number;
  chemical_reaction_rate: number;
  qc_chamber_temperature_k: number;
  cumulative_h2_moles: number;
  qc_heat_release_w: number;
  chamber_temperatures_k?: number[];
  chamber_pressures_pa?: number[];
  gallery_peak_pressure: number;
  gallery_rms_pressure: number;
  gallery_sound_speed_avg: number;
  gallery_total_acoustic_energy: number;
  f_sharp_spectral_purity: number;
  top_pressure_kc_entry: number;
  antechamber_p_in: number;
  antechamber_p_out: number;
  antechamber_transmission_loss_db: number;
  antechamber_p_trans: number;
  total_piezo_voltage: number;
  total_piezo_charge: number;
  displacement_current_a?: number;
  beam_array_impedance_ohms?: number;
  total_mechanical_energy: number;
  total_electrostatic_energy: number;
  max_beam_stress_pa: number;
  spark_triggered: boolean;
  spark_count: number;
  ion_density: number;
  maser_total_radiated_power: number;
  effective_radiated_power_w?: number;
  maser_population_inversion: number;
  maser_photon_energy_density: number;
  maser_pumping_rate: number;
  maser_is_above_threshold: boolean;
  maser_north_beam_power: number;
  maser_south_beam_power: number;
  shaft_poynting_flux_w_m2?: number[];
  maser_state_populations?: Record<string, number>;
  maser_cumulative_radiated_energy: number;
  p_total_in: number;
  p_total_out: number;
  p_total_loss: number;
  cumulative_energy_in: number;
  cumulative_energy_out: number;
  cumulative_energy_loss: number;
  total_stored_energy: number;
  delta_stored_energy: number;
  net_work: number;
  energy_balance_error: number;
  relative_energy_error: number;
  is_energy_conserved: boolean;
  spatial?: TelemetrySpatial;
}

export interface TelemetryDataset {
  simulation_id: string;
  version: string;
  scenario_name: string;
  duration: number;
  dt_macro: number;
  dt_micro: number;
  total_frames: number;
  metadata: Record<string, any>;
  summary: Record<string, any>;
  frames: TelemetryFrame[];
  binaryChannels?: Record<string, Float32Array>;
}

export type InterpolationMode = 'hermite' | 'linear';

function hermiteScalar(
  ym1: number,
  y0: number,
  y1: number,
  yp2: number,
  s: number,
  dt: number,
  dt0: number,
  dt1: number,
  clampMin?: number,
  clampMax?: number
): number {
  const s2 = s * s;
  const s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;

  const m0 = dt0 > 1e-9 ? ((y1 - ym1) / dt0) * dt : y1 - y0;
  const m1 = dt1 > 1e-9 ? ((yp2 - y0) / dt1) * dt : y1 - y0;

  let val = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
  if (clampMin !== undefined) val = Math.max(clampMin, val);
  if (clampMax !== undefined) val = Math.min(clampMax, val);
  return val;
}

function interpolateArrayHermite(
  am1?: number[],
  a0?: number[],
  a1?: number[],
  ap2?: number[],
  s: number = 0,
  dt: number = 0.01,
  dt0: number = 0.02,
  dt1: number = 0.02,
  clampMin?: number,
  clampMax?: number
): number[] | undefined {
  if (!a0 && !a1) return undefined;
  if (!a0) return a1;
  if (!a1) return a0;

  const len = Math.min(a0.length, a1.length);
  const out = new Array<number>(len);
  const hasM1 = am1 && am1.length >= len;
  const hasP2 = ap2 && ap2.length >= len;

  for (let i = 0; i < len; i++) {
    const y0 = a0[i];
    const y1 = a1[i];
    const ym1 = hasM1 ? (am1 as number[])[i] : 2 * y0 - y1;
    const yp2 = hasP2 ? (ap2 as number[])[i] : 2 * y1 - y0;

    out[i] = hermiteScalar(ym1, y0, y1, yp2, s, dt, dt0, dt1, clampMin, clampMax);
  }
  return out;
}

function interpolateArrayLinear(
  a0?: number[],
  a1?: number[],
  s: number = 0,
  clampMin?: number,
  clampMax?: number
): number[] | undefined {
  if (!a0 && !a1) return undefined;
  if (!a0) return a1;
  if (!a1) return a0;
  const len = Math.min(a0.length, a1.length);
  const out = new Array<number>(len);
  for (let i = 0; i < len; i++) {
    let val = a0[i] + (a1[i] - a0[i]) * s;
    if (clampMin !== undefined) val = Math.max(clampMin, val);
    if (clampMax !== undefined) val = Math.min(clampMax, val);
    out[i] = val;
  }
  return out;
}

function interpolateRecordHermite(
  rm1?: Record<string, number>,
  r0?: Record<string, number>,
  r1?: Record<string, number>,
  rp2?: Record<string, number>,
  s: number = 0,
  dt: number = 0.01,
  dt0: number = 0.02,
  dt1: number = 0.02
): Record<string, number> | undefined {
  if (!r0 && !r1) return undefined;
  if (!r0) return r1;
  if (!r1) return r0;

  const keys = new Set([...Object.keys(r0), ...Object.keys(r1)]);
  const out: Record<string, number> = {};
  for (const k of keys) {
    const y0 = r0[k] ?? 0;
    const y1 = r1[k] ?? 0;
    const ym1 = rm1 ? (rm1[k] ?? 2 * y0 - y1) : 2 * y0 - y1;
    const yp2 = rp2 ? (rp2[k] ?? 2 * y1 - y0) : 2 * y1 - y0;
    out[k] = Math.max(0, hermiteScalar(ym1, y0, y1, yp2, s, dt, dt0, dt1));
  }
  return out;
}

function interpolateRecordLinear(
  r0?: Record<string, number>,
  r1?: Record<string, number>,
  s: number = 0
): Record<string, number> | undefined {
  if (!r0 && !r1) return undefined;
  if (!r0) return r1;
  if (!r1) return r0;
  const keys = new Set([...Object.keys(r0), ...Object.keys(r1)]);
  const out: Record<string, number> = {};
  for (const k of keys) {
    const v0 = r0[k] ?? 0;
    const v1 = r1[k] ?? 0;
    out[k] = Math.max(0, v0 + (v1 - v0) * s);
  }
  return out;
}

export class TelemetryLoader {
  private dataset: TelemetryDataset | null = null;
  private onLoadedCallbacks: Array<(data: TelemetryDataset) => void> = [];
  private interpolationMode: InterpolationMode = 'hermite';

  constructor() {}

  public setInterpolationMode(mode: InterpolationMode): void {
    this.interpolationMode = mode;
  }

  public getInterpolationMode(): InterpolationMode {
    return this.interpolationMode;
  }

  public getDataset(): TelemetryDataset | null {
    return this.dataset;
  }

  public getDuration(): number {
    return this.dataset?.duration ?? 0;
  }

  public getFrameCount(): number {
    return this.dataset?.frames?.length ?? 0;
  }

  public onDataLoaded(cb: (data: TelemetryDataset) => void): () => void {
    this.onLoadedCallbacks.push(cb);
    if (this.dataset) {
      cb(this.dataset);
    }
    return () => {
      const idx = this.onLoadedCallbacks.indexOf(cb);
      if (idx >= 0) this.onLoadedCallbacks.splice(idx, 1);
    };
  }

  public async load(source: string | File): Promise<TelemetryDataset> {
    if (source instanceof File) {
      const fileName = source.name.toLowerCase();
      if (fileName.endsWith('.bin')) {
        const buffer = await source.arrayBuffer();
        return this.loadBinary(buffer);
      } else {
        const text = await source.text();
        const json = JSON.parse(text);
        return this.loadFromJson(json);
      }
    }

    if (typeof source === 'string') {
      const url = source;
      const lowerUrl = url.toLowerCase();
      if (lowerUrl.endsWith('.bin') || lowerUrl.includes('.bin?')) {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`Failed to fetch binary telemetry from ${url}: status ${res.status}`);
        }
        const buffer = await res.arrayBuffer();
        return this.loadBinary(buffer);
      } else {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`Failed to fetch telemetry from ${url}: status ${res.status}`);
        }
        const contentType = res.headers.get('content-type') || '';
        if (contentType.includes('octet-stream')) {
          const buffer = await res.arrayBuffer();
          return this.loadBinary(buffer);
        }
        const json = await res.json();
        return this.loadFromJson(json);
      }
    }

    throw new Error('Unsupported telemetry source type. Expected string URL or File.');
  }

  public async loadDefault(url: string = './sample_telemetry.json'): Promise<TelemetryDataset> {
    return this.load(url);
  }

  public async loadFromFile(file: File): Promise<TelemetryDataset> {
    return this.load(file);
  }

  public loadFromJson(data: any): TelemetryDataset {
    if (!data || !Array.isArray(data.frames) || data.frames.length === 0) {
      throw new Error('Invalid telemetry schema: Missing frames array');
    }

    const validated: TelemetryDataset = {
      simulation_id: data.simulation_id || 'pyramid_powerplant_sim',
      version: data.version || '1.0.0',
      scenario_name: data.scenario_name || 'default',
      duration: data.duration ?? (data.frames.length > 0 ? data.frames[data.frames.length - 1].time : 0),
      dt_macro: data.dt_macro ?? 0.01,
      dt_micro: data.dt_micro ?? 0.0001,
      total_frames: data.frames.length,
      metadata: data.metadata || {},
      summary: data.summary || {},
      frames: data.frames.map((f: any, idx: number) => this.normalizeFrame(f, idx)),
    };

    this.dataset = validated;

    for (const cb of this.onLoadedCallbacks) {
      cb(validated);
    }

    return validated;
  }

  public loadBinary(buffer: ArrayBuffer): TelemetryDataset {
    if (buffer.byteLength < 4) {
      throw new Error('Binary telemetry buffer is truncated: length < 4 bytes');
    }

    const headerLenView = new DataView(buffer);
    const headerLen = headerLenView.getUint32(0, true);

    if (buffer.byteLength < 4 + headerLen) {
      throw new Error(`Binary telemetry buffer corrupted: buffer length ${buffer.byteLength} < 4 + headerLen ${headerLen}`);
    }

    const headerBytes = new Uint8Array(buffer, 4, headerLen);
    const headerStr = new TextDecoder('utf-8').decode(headerBytes).trim();
    const header = JSON.parse(headerStr);

    const numFrames = Number(header.total_frames || 0);
    const channelsMeta: Record<string, any> = header.channels || {};
    const gasNodes: string[] = header.gas_nodes || [];
    const maserPopKeys: string[] = header.maser_population_keys || [];
    const payloadOffset = 4 + headerLen;

    const binaryChannels: Record<string, Float32Array> = {};

    for (const [name, meta] of Object.entries(channelsMeta)) {
      const offsetBytes = meta.offset_bytes ?? 0;
      const byteOffset = payloadOffset + offsetBytes;
      const shape: number[] = meta.shape || [];
      const count = meta.count ?? (shape.length > 0 ? shape.reduce((a, b) => a * b, 1) : 0);

      if (count > 0 && byteOffset + count * 4 <= buffer.byteLength) {
        if (byteOffset % 4 === 0) {
          binaryChannels[name] = new Float32Array(buffer, byteOffset, count);
        } else {
          const slice = buffer.slice(byteOffset, byteOffset + count * 4);
          binaryChannels[name] = new Float32Array(slice);
        }
      } else {
        binaryChannels[name] = new Float32Array(0);
      }
    }

    const getMatrixRow = (channelName: string, frameIdx: number, cols: number): number[] => {
      const arr = binaryChannels[channelName];
      if (!arr || arr.length < (frameIdx + 1) * cols) return [];
      const start = frameIdx * cols;
      const row = new Array<number>(cols);
      for (let c = 0; c < cols; c++) {
        row[c] = arr[start + c];
      }
      return row;
    };

    const getScalar = (channelName: string, frameIdx: number, fallback: number = 0): number => {
      const arr = binaryChannels[channelName];
      if (!arr || frameIdx >= arr.length) return fallback;
      return arr[frameIdx];
    };

    const galleryZList = binaryChannels['gallery_z'] ? Array.from(binaryChannels['gallery_z']) : [];
    const fftFreqList = binaryChannels['fft_frequencies_hz'] ? Array.from(binaryChannels['fft_frequencies_hz']) : [];

    const pressCols = channelsMeta['acoustic_pressure_profile']?.shape?.[1] ?? 0;
    const velCols = channelsMeta['acoustic_velocity_profile']?.shape?.[1] ?? 0;
    const engCols = channelsMeta['acoustic_energy_density']?.shape?.[1] ?? 0;
    const gasH2Cols = channelsMeta['gas_h2_mole_fractions']?.shape?.[1] ?? 0;
    const gasCCols = channelsMeta['gas_sound_speeds']?.shape?.[1] ?? 0;
    const gasRhoCols = channelsMeta['gas_densities']?.shape?.[1] ?? 0;
    const tierCols = channelsMeta['tier_voltages']?.shape?.[1] ?? 0;
    const beamStressCols = channelsMeta['all_beam_stresses_mpa']?.shape?.[1] ?? 0;
    const beamVoltCols = channelsMeta['all_beam_voltages_v']?.shape?.[1] ?? 0;
    const fftPsdCols = channelsMeta['fft_power_spectral_density_db']?.shape?.[1] ?? 0;

    const tempCols = channelsMeta['chamber_temperatures_k']?.shape?.[1] ?? 0;
    const pressChamberCols = channelsMeta['chamber_pressures_pa']?.shape?.[1] ?? 0;
    const fluxCols = channelsMeta['shaft_poynting_flux_w_m2']?.shape?.[1] ?? 0;
    const maserPopCols = channelsMeta['maser_state_populations']?.shape?.[1] ?? 0;

    const dtMacro = header.dt_macro ?? 0.01;
    const frames: TelemetryFrame[] = [];

    for (let i = 0; i < numFrames; i++) {
      const maserPopMap: Record<string, number> = {};
      if (maserPopCols > 0 && maserPopKeys.length > 0) {
        const arr = binaryChannels['maser_state_populations'];
        if (arr) {
          const start = i * maserPopCols;
          for (let k = 0; k < maserPopKeys.length && k < maserPopCols; k++) {
            maserPopMap[maserPopKeys[k]] = arr[start + k];
          }
        }
      }

      const spatial: TelemetrySpatial = {
        gallery_z: galleryZList,
        acoustic_pressure_profile: pressCols > 0 ? getMatrixRow('acoustic_pressure_profile', i, pressCols) : [],
        acoustic_velocity_profile: velCols > 0 ? getMatrixRow('acoustic_velocity_profile', i, velCols) : [],
        acoustic_energy_density: engCols > 0 ? getMatrixRow('acoustic_energy_density', i, engCols) : [],
        gas_nodes: gasNodes,
        gas_h2_mole_fractions: gasH2Cols > 0 ? getMatrixRow('gas_h2_mole_fractions', i, gasH2Cols) : [],
        gas_sound_speeds: gasCCols > 0 ? getMatrixRow('gas_sound_speeds', i, gasCCols) : [],
        gas_densities: gasRhoCols > 0 ? getMatrixRow('gas_densities', i, gasRhoCols) : [],
        tier_voltages: tierCols > 0 ? getMatrixRow('tier_voltages', i, tierCols) : [],
        all_beam_stresses_mpa: beamStressCols > 0 ? getMatrixRow('all_beam_stresses_mpa', i, beamStressCols) : [],
        all_beam_voltages_v: beamVoltCols > 0 ? getMatrixRow('all_beam_voltages_v', i, beamVoltCols) : [],
        fft_frequencies_hz: fftFreqList,
        fft_power_spectral_density_db: fftPsdCols > 0 ? getMatrixRow('fft_power_spectral_density_db', i, fftPsdCols) : [],
        north_shaft_power: getScalar('north_shaft_power', i, 0),
        south_shaft_power: getScalar('south_shaft_power', i, 0),
      };

      const frame: TelemetryFrame = {
        time: getScalar('time', i, i * dtMacro),
        step_index: Math.round(getScalar('step_index', i, i)),
        bedrock_displacement: getScalar('bedrock_displacement', i, 0),
        bedrock_velocity: getScalar('bedrock_velocity', i, 0),
        bedrock_acceleration: getScalar('bedrock_acceleration', i, 0),
        water_hammer_pressure: getScalar('water_hammer_pressure', i, 0),
        seismic_force: getScalar('seismic_force', i, 0),
        hydraulic_force: getScalar('hydraulic_force', i, 0),
        schumann_excitation: getScalar('schumann_excitation', i, 0),
        acoustic_pressure_sub: getScalar('acoustic_pressure_sub', i, 0),
        h2_mole_fraction_qc: getScalar('h2_mole_fraction_qc', i, 0),
        h2_mole_fraction_kc: getScalar('h2_mole_fraction_kc', i, 0),
        chemical_reaction_rate: getScalar('chemical_reaction_rate', i, 0),
        qc_chamber_temperature_k: getScalar('qc_chamber_temperature_k', i, 293.15),
        cumulative_h2_moles: getScalar('cumulative_h2_moles', i, 0),
        qc_heat_release_w: getScalar('qc_heat_release_w', i, 0),
        chamber_temperatures_k: tempCols > 0 ? getMatrixRow('chamber_temperatures_k', i, tempCols) : undefined,
        chamber_pressures_pa: pressChamberCols > 0 ? getMatrixRow('chamber_pressures_pa', i, pressChamberCols) : undefined,
        gallery_peak_pressure: getScalar('gallery_peak_pressure', i, 0),
        gallery_rms_pressure: getScalar('gallery_rms_pressure', i, 0),
        gallery_sound_speed_avg: getScalar('gallery_sound_speed_avg', i, 343.2),
        gallery_total_acoustic_energy: getScalar('gallery_total_acoustic_energy', i, 0),
        f_sharp_spectral_purity: getScalar('f_sharp_spectral_purity', i, 0),
        top_pressure_kc_entry: getScalar('top_pressure_kc_entry', i, 0),
        antechamber_p_in: getScalar('antechamber_p_in', i, 0),
        antechamber_p_out: getScalar('antechamber_p_out', i, 0),
        antechamber_transmission_loss_db: getScalar('antechamber_transmission_loss_db', i, 0),
        antechamber_p_trans: getScalar('antechamber_p_trans', i, 0),
        total_piezo_voltage: getScalar('total_piezo_voltage', i, 0),
        total_piezo_charge: getScalar('total_piezo_charge', i, 0),
        displacement_current_a: getScalar('displacement_current_a', i, 0),
        beam_array_impedance_ohms: getScalar('beam_array_impedance_ohms', i, 0),
        total_mechanical_energy: getScalar('total_mechanical_energy', i, 0),
        total_electrostatic_energy: getScalar('total_electrostatic_energy', i, 0),
        max_beam_stress_pa: getScalar('max_beam_stress_pa', i, 0),
        spark_triggered: getScalar('spark_triggered', i, 0) > 0.5,
        spark_count: Math.round(getScalar('spark_count', i, 0)),
        ion_density: getScalar('ion_density', i, 0),
        maser_total_radiated_power: getScalar('maser_total_radiated_power', i, 0),
        effective_radiated_power_w: getScalar('effective_radiated_power_w', i, 0),
        maser_population_inversion: getScalar('maser_population_inversion', i, 0),
        maser_photon_energy_density: getScalar('maser_photon_energy_density', i, 0),
        maser_pumping_rate: getScalar('maser_pumping_rate', i, 0),
        maser_is_above_threshold: getScalar('maser_is_above_threshold', i, 0) > 0.5,
        maser_north_beam_power: getScalar('maser_north_beam_power', i, 0),
        maser_south_beam_power: getScalar('maser_south_beam_power', i, 0),
        shaft_poynting_flux_w_m2: fluxCols > 0 ? getMatrixRow('shaft_poynting_flux_w_m2', i, fluxCols) : undefined,
        maser_state_populations: Object.keys(maserPopMap).length > 0 ? maserPopMap : undefined,
        maser_cumulative_radiated_energy: getScalar('maser_cumulative_radiated_energy', i, 0),
        p_total_in: getScalar('p_total_in', i, 0),
        p_total_out: getScalar('p_total_out', i, 0),
        p_total_loss: getScalar('p_total_loss', i, 0),
        cumulative_energy_in: getScalar('cumulative_energy_in', i, 0),
        cumulative_energy_out: getScalar('cumulative_energy_out', i, 0),
        cumulative_energy_loss: getScalar('cumulative_energy_loss', i, 0),
        total_stored_energy: getScalar('total_stored_energy', i, 0),
        delta_stored_energy: getScalar('delta_stored_energy', i, 0),
        net_work: getScalar('net_work', i, 0),
        energy_balance_error: getScalar('energy_balance_error', i, 0),
        relative_energy_error: getScalar('relative_energy_error', i, 0),
        is_energy_conserved: getScalar('is_energy_conserved', i, 1) > 0.5,
        spatial,
      };

      frames.push(frame);
    }

    const validated: TelemetryDataset = {
      simulation_id: header.simulation_id || 'pyramid_powerplant_sim',
      version: header.version || '1.0.0',
      scenario_name: header.scenario_name || 'baseline',
      duration: header.duration ?? (frames.length > 0 ? frames[frames.length - 1].time : 0),
      dt_macro: dtMacro,
      dt_micro: header.dt_micro ?? 0.0001,
      total_frames: frames.length,
      metadata: header.metadata || {},
      summary: header.summary || {},
      frames,
      binaryChannels,
    };

    this.dataset = validated;

    for (const cb of this.onLoadedCallbacks) {
      cb(validated);
    }

    return validated;
  }

  private normalizeFrame(f: any, idx: number): TelemetryFrame {
    const spatial: TelemetrySpatial | undefined = f.spatial
      ? {
          gallery_z: f.spatial.gallery_z,
          acoustic_pressure_profile: f.spatial.acoustic_pressure_profile,
          acoustic_velocity_profile: f.spatial.acoustic_velocity_profile,
          acoustic_energy_density: f.spatial.acoustic_energy_density,
          gas_nodes: f.spatial.gas_nodes,
          gas_h2_mole_fractions: f.spatial.gas_h2_mole_fractions,
          gas_sound_speeds: f.spatial.gas_sound_speeds,
          gas_densities: f.spatial.gas_densities,
          tier_voltages: f.spatial.tier_voltages,
          all_beam_stresses_mpa: f.spatial.all_beam_stresses_mpa,
          all_beam_voltages_v: f.spatial.all_beam_voltages_v,
          fft_frequencies_hz: f.spatial.fft_frequencies_hz,
          fft_power_spectral_density_db: f.spatial.fft_power_spectral_density_db,
          north_shaft_power: f.spatial.north_shaft_power,
          south_shaft_power: f.spatial.south_shaft_power,
        }
      : undefined;

    return {
      time: f.time ?? idx * 0.01,
      step_index: f.step_index ?? idx,
      bedrock_displacement: f.bedrock_displacement ?? 0,
      bedrock_velocity: f.bedrock_velocity ?? 0,
      bedrock_acceleration: f.bedrock_acceleration ?? 0,
      water_hammer_pressure: f.water_hammer_pressure ?? 0,
      seismic_force: f.seismic_force ?? 0,
      hydraulic_force: f.hydraulic_force ?? 0,
      schumann_excitation: f.schumann_excitation ?? 0,
      acoustic_pressure_sub: f.acoustic_pressure_sub ?? 0,
      h2_mole_fraction_qc: f.h2_mole_fraction_qc ?? 0,
      h2_mole_fraction_kc: f.h2_mole_fraction_kc ?? 0,
      chemical_reaction_rate: f.chemical_reaction_rate ?? 0,
      qc_chamber_temperature_k: f.qc_chamber_temperature_k ?? 293.15,
      cumulative_h2_moles: f.cumulative_h2_moles ?? 0,
      qc_heat_release_w: f.qc_heat_release_w ?? 0,
      chamber_temperatures_k: f.chamber_temperatures_k,
      chamber_pressures_pa: f.chamber_pressures_pa,
      gallery_peak_pressure: f.gallery_peak_pressure ?? 0,
      gallery_rms_pressure: f.gallery_rms_pressure ?? 0,
      gallery_sound_speed_avg: f.gallery_sound_speed_avg ?? 343.2,
      gallery_total_acoustic_energy: f.gallery_total_acoustic_energy ?? 0,
      f_sharp_spectral_purity: f.f_sharp_spectral_purity ?? 0,
      top_pressure_kc_entry: f.top_pressure_kc_entry ?? 0,
      antechamber_p_in: f.antechamber_p_in ?? 0,
      antechamber_p_out: f.antechamber_p_out ?? 0,
      antechamber_transmission_loss_db: f.antechamber_transmission_loss_db ?? 0,
      antechamber_p_trans: f.antechamber_p_trans ?? 0,
      total_piezo_voltage: f.total_piezo_voltage ?? 0,
      total_piezo_charge: f.total_piezo_charge ?? 0,
      displacement_current_a: f.displacement_current_a ?? 0,
      beam_array_impedance_ohms: f.beam_array_impedance_ohms ?? 0,
      total_mechanical_energy: f.total_mechanical_energy ?? 0,
      total_electrostatic_energy: f.total_electrostatic_energy ?? 0,
      max_beam_stress_pa: f.max_beam_stress_pa ?? 0,
      spark_triggered: Boolean(f.spark_triggered),
      spark_count: f.spark_count ?? 0,
      ion_density: f.ion_density ?? 0,
      maser_total_radiated_power: f.maser_total_radiated_power ?? 0,
      effective_radiated_power_w: f.effective_radiated_power_w ?? 0,
      maser_population_inversion: f.maser_population_inversion ?? 0,
      maser_photon_energy_density: f.maser_photon_energy_density ?? 0,
      maser_pumping_rate: f.maser_pumping_rate ?? 0,
      maser_is_above_threshold: Boolean(f.maser_is_above_threshold),
      maser_north_beam_power: f.maser_north_beam_power ?? 0,
      maser_south_beam_power: f.maser_south_beam_power ?? 0,
      shaft_poynting_flux_w_m2: f.shaft_poynting_flux_w_m2,
      maser_state_populations: f.maser_state_populations,
      maser_cumulative_radiated_energy: f.maser_cumulative_radiated_energy ?? 0,
      p_total_in: f.p_total_in ?? 0,
      p_total_out: f.p_total_out ?? 0,
      p_total_loss: f.p_total_loss ?? 0,
      cumulative_energy_in: f.cumulative_energy_in ?? 0,
      cumulative_energy_out: f.cumulative_energy_out ?? 0,
      cumulative_energy_loss: f.cumulative_energy_loss ?? 0,
      total_stored_energy: f.total_stored_energy ?? 0,
      delta_stored_energy: f.delta_stored_energy ?? 0,
      net_work: f.net_work ?? 0,
      energy_balance_error: f.energy_balance_error ?? 0,
      relative_energy_error: f.relative_energy_error ?? 0,
      is_energy_conserved: Boolean(f.is_energy_conserved ?? true),
      spatial,
    };
  }

  public getInterpolatedFrame(t: number): TelemetryFrame | null {
    if (!this.dataset || this.dataset.frames.length === 0) return null;
    const frames = this.dataset.frames;
    const n = frames.length;

    if (t <= frames[0].time) return frames[0];
    if (t >= frames[n - 1].time) return frames[n - 1];

    let low = 0;
    let high = n - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (frames[mid].time <= t) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    const k0 = Math.max(0, high);
    const k1 = Math.min(n - 1, k0 + 1);

    const f0 = frames[k0];
    const f1 = frames[k1];
    const dt = f1.time - f0.time;
    if (dt <= 1e-9) return f0;

    const s = Math.max(0, Math.min(1, (t - f0.time) / dt));

    if (this.interpolationMode === 'linear') {
      return this.interpolateLinear(f0, f1, s, t);
    }

    const km1 = Math.max(0, k0 - 1);
    const kp2 = Math.min(n - 1, k1 + 1);
    const fm1 = frames[km1];
    const fp2 = frames[kp2];

    return this.interpolateHermite(fm1, f0, f1, fp2, s, dt, t);
  }

  private interpolateLinear(f0: TelemetryFrame, f1: TelemetryFrame, s: number, t: number): TelemetryFrame {
    const lerp = (a: number, b: number) => a + (b - a) * s;
    const nearest = s < 0.5 ? f0 : f1;

    return {
      time: t,
      step_index: nearest.step_index,
      bedrock_displacement: lerp(f0.bedrock_displacement, f1.bedrock_displacement),
      bedrock_velocity: lerp(f0.bedrock_velocity, f1.bedrock_velocity),
      bedrock_acceleration: lerp(f0.bedrock_acceleration, f1.bedrock_acceleration),
      water_hammer_pressure: lerp(f0.water_hammer_pressure, f1.water_hammer_pressure),
      seismic_force: lerp(f0.seismic_force, f1.seismic_force),
      hydraulic_force: lerp(f0.hydraulic_force, f1.hydraulic_force),
      schumann_excitation: lerp(f0.schumann_excitation, f1.schumann_excitation),
      acoustic_pressure_sub: lerp(f0.acoustic_pressure_sub, f1.acoustic_pressure_sub),
      h2_mole_fraction_qc: lerp(f0.h2_mole_fraction_qc, f1.h2_mole_fraction_qc),
      h2_mole_fraction_kc: lerp(f0.h2_mole_fraction_kc, f1.h2_mole_fraction_kc),
      chemical_reaction_rate: lerp(f0.chemical_reaction_rate, f1.chemical_reaction_rate),
      qc_chamber_temperature_k: lerp(f0.qc_chamber_temperature_k, f1.qc_chamber_temperature_k),
      cumulative_h2_moles: lerp(f0.cumulative_h2_moles, f1.cumulative_h2_moles),
      qc_heat_release_w: lerp(f0.qc_heat_release_w, f1.qc_heat_release_w),
      chamber_temperatures_k: interpolateArrayLinear(f0.chamber_temperatures_k, f1.chamber_temperatures_k, s, 0),
      chamber_pressures_pa: interpolateArrayLinear(f0.chamber_pressures_pa, f1.chamber_pressures_pa, s, 0),
      gallery_peak_pressure: lerp(f0.gallery_peak_pressure, f1.gallery_peak_pressure),
      gallery_rms_pressure: lerp(f0.gallery_rms_pressure, f1.gallery_rms_pressure),
      gallery_sound_speed_avg: lerp(f0.gallery_sound_speed_avg, f1.gallery_sound_speed_avg),
      gallery_total_acoustic_energy: lerp(f0.gallery_total_acoustic_energy, f1.gallery_total_acoustic_energy),
      f_sharp_spectral_purity: lerp(f0.f_sharp_spectral_purity, f1.f_sharp_spectral_purity),
      top_pressure_kc_entry: lerp(f0.top_pressure_kc_entry, f1.top_pressure_kc_entry),
      antechamber_p_in: lerp(f0.antechamber_p_in, f1.antechamber_p_in),
      antechamber_p_out: lerp(f0.antechamber_p_out, f1.antechamber_p_out),
      antechamber_transmission_loss_db: lerp(f0.antechamber_transmission_loss_db, f1.antechamber_transmission_loss_db),
      antechamber_p_trans: lerp(f0.antechamber_p_trans, f1.antechamber_p_trans),
      total_piezo_voltage: lerp(f0.total_piezo_voltage, f1.total_piezo_voltage),
      total_piezo_charge: lerp(f0.total_piezo_charge, f1.total_piezo_charge),
      displacement_current_a: lerp(f0.displacement_current_a ?? 0, f1.displacement_current_a ?? 0),
      beam_array_impedance_ohms: lerp(f0.beam_array_impedance_ohms ?? 0, f1.beam_array_impedance_ohms ?? 0),
      total_mechanical_energy: lerp(f0.total_mechanical_energy, f1.total_mechanical_energy),
      total_electrostatic_energy: lerp(f0.total_electrostatic_energy, f1.total_electrostatic_energy),
      max_beam_stress_pa: lerp(f0.max_beam_stress_pa, f1.max_beam_stress_pa),
      spark_triggered: f0.spark_triggered || f1.spark_triggered,
      spark_count: Math.round(lerp(f0.spark_count, f1.spark_count)),
      ion_density: lerp(f0.ion_density, f1.ion_density),
      maser_total_radiated_power: lerp(f0.maser_total_radiated_power, f1.maser_total_radiated_power),
      effective_radiated_power_w: lerp(f0.effective_radiated_power_w ?? 0, f1.effective_radiated_power_w ?? 0),
      maser_population_inversion: lerp(f0.maser_population_inversion, f1.maser_population_inversion),
      maser_photon_energy_density: lerp(f0.maser_photon_energy_density, f1.maser_photon_energy_density),
      maser_pumping_rate: lerp(f0.maser_pumping_rate, f1.maser_pumping_rate),
      maser_is_above_threshold: nearest.maser_is_above_threshold,
      maser_north_beam_power: lerp(f0.maser_north_beam_power, f1.maser_north_beam_power),
      maser_south_beam_power: lerp(f0.maser_south_beam_power, f1.maser_south_beam_power),
      shaft_poynting_flux_w_m2: interpolateArrayLinear(f0.shaft_poynting_flux_w_m2, f1.shaft_poynting_flux_w_m2, s, 0),
      maser_state_populations: interpolateRecordLinear(f0.maser_state_populations, f1.maser_state_populations, s),
      maser_cumulative_radiated_energy: lerp(f0.maser_cumulative_radiated_energy, f1.maser_cumulative_radiated_energy),
      p_total_in: lerp(f0.p_total_in, f1.p_total_in),
      p_total_out: lerp(f0.p_total_out, f1.p_total_out),
      p_total_loss: lerp(f0.p_total_loss, f1.p_total_loss),
      cumulative_energy_in: lerp(f0.cumulative_energy_in, f1.cumulative_energy_in),
      cumulative_energy_out: lerp(f0.cumulative_energy_out, f1.cumulative_energy_out),
      cumulative_energy_loss: lerp(f0.cumulative_energy_loss, f1.cumulative_energy_loss),
      total_stored_energy: lerp(f0.total_stored_energy, f1.total_stored_energy),
      delta_stored_energy: lerp(f0.delta_stored_energy, f1.delta_stored_energy),
      net_work: lerp(f0.net_work, f1.net_work),
      energy_balance_error: lerp(f0.energy_balance_error, f1.energy_balance_error),
      relative_energy_error: lerp(f0.relative_energy_error, f1.relative_energy_error),
      is_energy_conserved: nearest.is_energy_conserved,
      spatial: this.interpolateSpatialLinear(f0.spatial, f1.spatial, s),
    };
  }

  private interpolateHermite(
    fm1: TelemetryFrame,
    f0: TelemetryFrame,
    f1: TelemetryFrame,
    fp2: TelemetryFrame,
    s: number,
    dt: number,
    t: number
  ): TelemetryFrame {
    const dt0 = f1.time - fm1.time > 1e-9 ? f1.time - fm1.time : dt * 2;
    const dt1 = fp2.time - f0.time > 1e-9 ? fp2.time - f0.time : dt * 2;

    const hermite = (ym1: number, y0: number, y1: number, yp2: number, clampMin?: number, clampMax?: number) =>
      hermiteScalar(ym1, y0, y1, yp2, s, dt, dt0, dt1, clampMin, clampMax);

    const nearest = s < 0.5 ? f0 : f1;

    return {
      time: t,
      step_index: nearest.step_index,
      bedrock_displacement: hermite(fm1.bedrock_displacement, f0.bedrock_displacement, f1.bedrock_displacement, fp2.bedrock_displacement),
      bedrock_velocity: hermite(fm1.bedrock_velocity, f0.bedrock_velocity, f1.bedrock_velocity, fp2.bedrock_velocity),
      bedrock_acceleration: hermite(fm1.bedrock_acceleration, f0.bedrock_acceleration, f1.bedrock_acceleration, fp2.bedrock_acceleration),
      water_hammer_pressure: hermite(fm1.water_hammer_pressure, f0.water_hammer_pressure, f1.water_hammer_pressure, fp2.water_hammer_pressure),
      seismic_force: hermite(fm1.seismic_force, f0.seismic_force, f1.seismic_force, fp2.seismic_force),
      hydraulic_force: hermite(fm1.hydraulic_force, f0.hydraulic_force, f1.hydraulic_force, fp2.hydraulic_force),
      schumann_excitation: hermite(fm1.schumann_excitation, f0.schumann_excitation, f1.schumann_excitation, fp2.schumann_excitation),
      acoustic_pressure_sub: hermite(fm1.acoustic_pressure_sub, f0.acoustic_pressure_sub, f1.acoustic_pressure_sub, fp2.acoustic_pressure_sub),
      h2_mole_fraction_qc: hermite(fm1.h2_mole_fraction_qc, f0.h2_mole_fraction_qc, f1.h2_mole_fraction_qc, fp2.h2_mole_fraction_qc, 0, 1),
      h2_mole_fraction_kc: hermite(fm1.h2_mole_fraction_kc, f0.h2_mole_fraction_kc, f1.h2_mole_fraction_kc, fp2.h2_mole_fraction_kc, 0, 1),
      chemical_reaction_rate: hermite(fm1.chemical_reaction_rate, f0.chemical_reaction_rate, f1.chemical_reaction_rate, fp2.chemical_reaction_rate, 0),
      qc_chamber_temperature_k: hermite(fm1.qc_chamber_temperature_k, f0.qc_chamber_temperature_k, f1.qc_chamber_temperature_k, fp2.qc_chamber_temperature_k, 0),
      cumulative_h2_moles: hermite(fm1.cumulative_h2_moles, f0.cumulative_h2_moles, f1.cumulative_h2_moles, fp2.cumulative_h2_moles, 0),
      qc_heat_release_w: hermite(fm1.qc_heat_release_w, f0.qc_heat_release_w, f1.qc_heat_release_w, fp2.qc_heat_release_w, 0),
      chamber_temperatures_k: interpolateArrayHermite(
        fm1.chamber_temperatures_k,
        f0.chamber_temperatures_k,
        f1.chamber_temperatures_k,
        fp2.chamber_temperatures_k,
        s,
        dt,
        dt0,
        dt1,
        0
      ),
      chamber_pressures_pa: interpolateArrayHermite(
        fm1.chamber_pressures_pa,
        f0.chamber_pressures_pa,
        f1.chamber_pressures_pa,
        fp2.chamber_pressures_pa,
        s,
        dt,
        dt0,
        dt1,
        0
      ),
      gallery_peak_pressure: hermite(fm1.gallery_peak_pressure, f0.gallery_peak_pressure, f1.gallery_peak_pressure, fp2.gallery_peak_pressure, 0),
      gallery_rms_pressure: hermite(fm1.gallery_rms_pressure, f0.gallery_rms_pressure, f1.gallery_rms_pressure, fp2.gallery_rms_pressure, 0),
      gallery_sound_speed_avg: hermite(fm1.gallery_sound_speed_avg, f0.gallery_sound_speed_avg, f1.gallery_sound_speed_avg, fp2.gallery_sound_speed_avg, 0),
      gallery_total_acoustic_energy: hermite(fm1.gallery_total_acoustic_energy, f0.gallery_total_acoustic_energy, f1.gallery_total_acoustic_energy, fp2.gallery_total_acoustic_energy, 0),
      f_sharp_spectral_purity: hermite(fm1.f_sharp_spectral_purity, f0.f_sharp_spectral_purity, f1.f_sharp_spectral_purity, fp2.f_sharp_spectral_purity, 0, 1),
      top_pressure_kc_entry: hermite(fm1.top_pressure_kc_entry, f0.top_pressure_kc_entry, f1.top_pressure_kc_entry, fp2.top_pressure_kc_entry),
      antechamber_p_in: hermite(fm1.antechamber_p_in, f0.antechamber_p_in, f1.antechamber_p_in, fp2.antechamber_p_in),
      antechamber_p_out: hermite(fm1.antechamber_p_out, f0.antechamber_p_out, f1.antechamber_p_out, fp2.antechamber_p_out),
      antechamber_transmission_loss_db: hermite(fm1.antechamber_transmission_loss_db, f0.antechamber_transmission_loss_db, f1.antechamber_transmission_loss_db, fp2.antechamber_transmission_loss_db),
      antechamber_p_trans: hermite(fm1.antechamber_p_trans, f0.antechamber_p_trans, f1.antechamber_p_trans, fp2.antechamber_p_trans),
      total_piezo_voltage: hermite(fm1.total_piezo_voltage, f0.total_piezo_voltage, f1.total_piezo_voltage, fp2.total_piezo_voltage),
      total_piezo_charge: hermite(fm1.total_piezo_charge, f0.total_piezo_charge, f1.total_piezo_charge, fp2.total_piezo_charge),
      displacement_current_a: hermite(fm1.displacement_current_a ?? 0, f0.displacement_current_a ?? 0, f1.displacement_current_a ?? 0, fp2.displacement_current_a ?? 0),
      beam_array_impedance_ohms: hermite(fm1.beam_array_impedance_ohms ?? 0, f0.beam_array_impedance_ohms ?? 0, f1.beam_array_impedance_ohms ?? 0, fp2.beam_array_impedance_ohms ?? 0, 0),
      total_mechanical_energy: hermite(fm1.total_mechanical_energy, f0.total_mechanical_energy, f1.total_mechanical_energy, fp2.total_mechanical_energy, 0),
      total_electrostatic_energy: hermite(fm1.total_electrostatic_energy, f0.total_electrostatic_energy, f1.total_electrostatic_energy, fp2.total_electrostatic_energy, 0),
      max_beam_stress_pa: hermite(fm1.max_beam_stress_pa, f0.max_beam_stress_pa, f1.max_beam_stress_pa, fp2.max_beam_stress_pa),
      spark_triggered: f0.spark_triggered || f1.spark_triggered,
      spark_count: Math.round(hermite(fm1.spark_count, f0.spark_count, f1.spark_count, fp2.spark_count, 0)),
      ion_density: hermite(fm1.ion_density, f0.ion_density, f1.ion_density, fp2.ion_density, 0),
      maser_total_radiated_power: hermite(fm1.maser_total_radiated_power, f0.maser_total_radiated_power, f1.maser_total_radiated_power, fp2.maser_total_radiated_power, 0),
      effective_radiated_power_w: hermite(fm1.effective_radiated_power_w ?? 0, f0.effective_radiated_power_w ?? 0, f1.effective_radiated_power_w ?? 0, fp2.effective_radiated_power_w ?? 0, 0),
      maser_population_inversion: hermite(fm1.maser_population_inversion, f0.maser_population_inversion, f1.maser_population_inversion, fp2.maser_population_inversion),
      maser_photon_energy_density: hermite(fm1.maser_photon_energy_density, f0.maser_photon_energy_density, f1.maser_photon_energy_density, fp2.maser_photon_energy_density, 0),
      maser_pumping_rate: hermite(fm1.maser_pumping_rate, f0.maser_pumping_rate, f1.maser_pumping_rate, fp2.maser_pumping_rate),
      maser_is_above_threshold: nearest.maser_is_above_threshold,
      maser_north_beam_power: hermite(fm1.maser_north_beam_power, f0.maser_north_beam_power, f1.maser_north_beam_power, fp2.maser_north_beam_power, 0),
      maser_south_beam_power: hermite(fm1.maser_south_beam_power, f0.maser_south_beam_power, f1.maser_south_beam_power, fp2.maser_south_beam_power, 0),
      shaft_poynting_flux_w_m2: interpolateArrayHermite(
        fm1.shaft_poynting_flux_w_m2,
        f0.shaft_poynting_flux_w_m2,
        f1.shaft_poynting_flux_w_m2,
        fp2.shaft_poynting_flux_w_m2,
        s,
        dt,
        dt0,
        dt1,
        0
      ),
      maser_state_populations: interpolateRecordHermite(
        fm1.maser_state_populations,
        f0.maser_state_populations,
        f1.maser_state_populations,
        fp2.maser_state_populations,
        s,
        dt,
        dt0,
        dt1
      ),
      maser_cumulative_radiated_energy: hermite(fm1.maser_cumulative_radiated_energy, f0.maser_cumulative_radiated_energy, f1.maser_cumulative_radiated_energy, fp2.maser_cumulative_radiated_energy, 0),
      p_total_in: hermite(fm1.p_total_in, f0.p_total_in, f1.p_total_in, fp2.p_total_in),
      p_total_out: hermite(fm1.p_total_out, f0.p_total_out, f1.p_total_out, fp2.p_total_out),
      p_total_loss: hermite(fm1.p_total_loss, f0.p_total_loss, f1.p_total_loss, fp2.p_total_loss),
      cumulative_energy_in: hermite(fm1.cumulative_energy_in, f0.cumulative_energy_in, f1.cumulative_energy_in, fp2.cumulative_energy_in, 0),
      cumulative_energy_out: hermite(fm1.cumulative_energy_out, f0.cumulative_energy_out, f1.cumulative_energy_out, fp2.cumulative_energy_out, 0),
      cumulative_energy_loss: hermite(fm1.cumulative_energy_loss, f0.cumulative_energy_loss, f1.cumulative_energy_loss, fp2.cumulative_energy_loss, 0),
      total_stored_energy: hermite(fm1.total_stored_energy, f0.total_stored_energy, f1.total_stored_energy, fp2.total_stored_energy),
      delta_stored_energy: hermite(fm1.delta_stored_energy, f0.delta_stored_energy, f1.delta_stored_energy, fp2.delta_stored_energy),
      net_work: hermite(fm1.net_work, f0.net_work, f1.net_work, fp2.net_work),
      energy_balance_error: hermite(fm1.energy_balance_error, f0.energy_balance_error, f1.energy_balance_error, fp2.energy_balance_error),
      relative_energy_error: hermite(fm1.relative_energy_error, f0.relative_energy_error, f1.relative_energy_error, fp2.relative_energy_error),
      is_energy_conserved: nearest.is_energy_conserved,
      spatial: this.interpolateSpatialHermite(fm1.spatial, f0.spatial, f1.spatial, fp2.spatial, s, dt, dt0, dt1),
    };
  }

  private interpolateSpatialLinear(s0?: TelemetrySpatial, s1?: TelemetrySpatial, s: number = 0): TelemetrySpatial | undefined {
    if (!s0 && !s1) return undefined;
    if (!s0) return s1;
    if (!s1) return s0;

    const lerpNum = (a?: number, b?: number): number | undefined => {
      if (a === undefined && b === undefined) return undefined;
      if (a === undefined) return b;
      if (b === undefined) return a;
      return a + (b - a) * s;
    };

    return {
      gallery_z: s0.gallery_z ?? s1.gallery_z,
      acoustic_pressure_profile: interpolateArrayLinear(s0.acoustic_pressure_profile, s1.acoustic_pressure_profile, s),
      acoustic_velocity_profile: interpolateArrayLinear(s0.acoustic_velocity_profile, s1.acoustic_velocity_profile, s),
      acoustic_energy_density: interpolateArrayLinear(s0.acoustic_energy_density, s1.acoustic_energy_density, s, 0),
      gas_nodes: s0.gas_nodes ?? s1.gas_nodes,
      gas_h2_mole_fractions: interpolateArrayLinear(s0.gas_h2_mole_fractions, s1.gas_h2_mole_fractions, s, 0, 1),
      gas_sound_speeds: interpolateArrayLinear(s0.gas_sound_speeds, s1.gas_sound_speeds, s, 0),
      gas_densities: interpolateArrayLinear(s0.gas_densities, s1.gas_densities, s, 0),
      tier_voltages: interpolateArrayLinear(s0.tier_voltages, s1.tier_voltages, s),
      all_beam_stresses_mpa: interpolateArrayLinear(s0.all_beam_stresses_mpa, s1.all_beam_stresses_mpa, s),
      all_beam_voltages_v: interpolateArrayLinear(s0.all_beam_voltages_v, s1.all_beam_voltages_v, s),
      fft_frequencies_hz: s0.fft_frequencies_hz ?? s1.fft_frequencies_hz,
      fft_power_spectral_density_db: interpolateArrayLinear(s0.fft_power_spectral_density_db, s1.fft_power_spectral_density_db, s),
      north_shaft_power: lerpNum(s0.north_shaft_power, s1.north_shaft_power),
      south_shaft_power: lerpNum(s0.south_shaft_power, s1.south_shaft_power),
    };
  }

  private interpolateSpatialHermite(
    sm1?: TelemetrySpatial,
    s0?: TelemetrySpatial,
    s1?: TelemetrySpatial,
    sp2?: TelemetrySpatial,
    s: number = 0,
    dt: number = 0.01,
    dt0: number = 0.02,
    dt1: number = 0.02
  ): TelemetrySpatial | undefined {
    if (!s0 && !s1) return undefined;
    if (!s0) return s1;
    if (!s1) return s0;

    const lerpArr = (am1?: number[], a0?: number[], a1?: number[], ap2?: number[], minVal?: number, maxVal?: number) =>
      interpolateArrayHermite(am1, a0, a1, ap2, s, dt, dt0, dt1, minVal, maxVal);

    const lerpNum = (ym1?: number, y0?: number, y1?: number, yp2?: number, minVal?: number, maxVal?: number): number | undefined => {
      if (y0 === undefined && y1 === undefined) return undefined;
      if (y0 === undefined) return y1;
      if (y1 === undefined) return y0;
      const ym = ym1 ?? 2 * y0 - y1;
      const yp = yp2 ?? 2 * y1 - y0;
      return hermiteScalar(ym, y0, y1, yp, s, dt, dt0, dt1, minVal, maxVal);
    };

    return {
      gallery_z: s0.gallery_z ?? s1.gallery_z,
      acoustic_pressure_profile: lerpArr(
        sm1?.acoustic_pressure_profile,
        s0.acoustic_pressure_profile,
        s1.acoustic_pressure_profile,
        sp2?.acoustic_pressure_profile
      ),
      acoustic_velocity_profile: lerpArr(
        sm1?.acoustic_velocity_profile,
        s0.acoustic_velocity_profile,
        s1.acoustic_velocity_profile,
        sp2?.acoustic_velocity_profile
      ),
      acoustic_energy_density: lerpArr(
        sm1?.acoustic_energy_density,
        s0.acoustic_energy_density,
        s1.acoustic_energy_density,
        sp2?.acoustic_energy_density,
        0
      ),
      gas_nodes: s0.gas_nodes ?? s1.gas_nodes,
      gas_h2_mole_fractions: lerpArr(
        sm1?.gas_h2_mole_fractions,
        s0.gas_h2_mole_fractions,
        s1.gas_h2_mole_fractions,
        sp2?.gas_h2_mole_fractions,
        0,
        1
      ),
      gas_sound_speeds: lerpArr(
        sm1?.gas_sound_speeds,
        s0.gas_sound_speeds,
        s1.gas_sound_speeds,
        sp2?.gas_sound_speeds,
        0
      ),
      gas_densities: lerpArr(
        sm1?.gas_densities,
        s0.gas_densities,
        s1.gas_densities,
        sp2?.gas_densities,
        0
      ),
      tier_voltages: lerpArr(
        sm1?.tier_voltages,
        s0.tier_voltages,
        s1.tier_voltages,
        sp2?.tier_voltages
      ),
      all_beam_stresses_mpa: lerpArr(
        sm1?.all_beam_stresses_mpa,
        s0.all_beam_stresses_mpa,
        s1.all_beam_stresses_mpa,
        sp2?.all_beam_stresses_mpa
      ),
      all_beam_voltages_v: lerpArr(
        sm1?.all_beam_voltages_v,
        s0.all_beam_voltages_v,
        s1.all_beam_voltages_v,
        sp2?.all_beam_voltages_v
      ),
      fft_frequencies_hz: s0.fft_frequencies_hz ?? s1.fft_frequencies_hz,
      fft_power_spectral_density_db: lerpArr(
        sm1?.fft_power_spectral_density_db,
        s0.fft_power_spectral_density_db,
        s1.fft_power_spectral_density_db,
        sp2?.fft_power_spectral_density_db
      ),
      north_shaft_power: lerpNum(sm1?.north_shaft_power, s0.north_shaft_power, s1.north_shaft_power, sp2?.north_shaft_power, 0),
      south_shaft_power: lerpNum(sm1?.south_shaft_power, s0.south_shaft_power, s1.south_shaft_power, sp2?.south_shaft_power, 0),
    };
  }

  public setupDragAndDrop(_targetElement: HTMLElement, overlayElement: HTMLElement): void {
    let dragCounter = 0;

    window.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      overlayElement.classList.add('active');
    });

    window.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        overlayElement.classList.remove('active');
      }
    });

    window.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    window.addEventListener('drop', async (e) => {
      e.preventDefault();
      dragCounter = 0;
      overlayElement.classList.remove('active');

      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        const file = files[0];
        try {
          await this.load(file);
        } catch (err) {
          console.error('Failed to load dropped telemetry file:', err);
        }
      }
    });
  }
}
