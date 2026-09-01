export interface ScenarioInfo {
  id: string;
  name: string;
  description: string;
  duration?: number;
  dt_macro?: number;
  bin_url?: string;
  json_url?: string;
  binUrl?: string;
  jsonUrl?: string;
  tags?: string[];
  thumbnail?: string;
  recommended?: boolean;
  metadata?: Record<string, any>;
}

export interface ScenarioManifest {
  version: string;
  scenarios: ScenarioInfo[];
  defaultScenarioId?: string;
  generated_at?: string;
}

export const DEFAULT_SCENARIOS: ScenarioInfo[] = [
  {
    id: 'sample',
    name: 'Sample Telemetry (JSON)',
    description: 'Pre-computed baseline simulation run with coupled acoustics, piezoelectric stack, and maser emission.',
    duration: 10.0,
    dt_macro: 0.01,
    jsonUrl: './sample_telemetry.json',
    json_url: './sample_telemetry.json',
    tags: ['sample', 'baseline', 'json'],
    recommended: true,
  },
  {
    id: 'baseline',
    name: 'Baseline Resonant Run (10s)',
    description: 'Standard 10-second multi-physics simulation with hydrogen injection, 7.83 Hz infrasonic drive, and quartz excitation.',
    duration: 10.0,
    dt_macro: 0.01,
    binUrl: './scenarios/baseline.bin',
    bin_url: './scenarios/baseline.bin',
    jsonUrl: './scenarios/baseline.json',
    json_url: './scenarios/baseline.json',
    tags: ['baseline', 'resonance', 'maser'],
  },
  {
    id: 'acoustic_peak',
    name: 'Acoustic Peak Mode',
    description: 'High-amplitude acoustic lock in the Grand Gallery resonator rack with steep standing wave pressure gradients.',
    duration: 10.0,
    dt_macro: 0.01,
    binUrl: './scenarios/acoustic_peak.bin',
    bin_url: './scenarios/acoustic_peak.bin',
    jsonUrl: './scenarios/acoustic_peak.json',
    json_url: './scenarios/acoustic_peak.json',
    tags: ['acoustic', 'resonance', 'high-pressure'],
  },
  {
    id: 'full_maser_power',
    name: 'Full Maser Power Output',
    description: 'High hydrogen mole fraction and kilovolt piezoelectric pumping driving saturated 1.4204 GHz maser beam emission.',
    duration: 10.0,
    dt_macro: 0.01,
    binUrl: './scenarios/full_maser_power.bin',
    bin_url: './scenarios/full_maser_power.bin',
    jsonUrl: './scenarios/full_maser_power.json',
    json_url: './scenarios/full_maser_power.json',
    tags: ['maser', 'quantum', 'high-power'],
  },
  {
    id: 'dry_run_no_gas',
    name: 'Dry Run (No Hydrogen Gas)',
    description: 'Acoustic and seismic drive in standard atmospheric air without chemical reaction, demonstrating sub-threshold behavior.',
    duration: 10.0,
    dt_macro: 0.01,
    binUrl: './scenarios/dry_run_no_gas.bin',
    bin_url: './scenarios/dry_run_no_gas.bin',
    jsonUrl: './scenarios/dry_run_no_gas.json',
    json_url: './scenarios/dry_run_no_gas.json',
    tags: ['inert', 'sub-threshold', 'control'],
  },
  {
    id: 'high_seismic',
    name: 'High Seismic Transient',
    description: 'Strong water hammer shock and bedrock acceleration transient testing structural relieving beam stress limits.',
    duration: 10.0,
    dt_macro: 0.01,
    binUrl: './scenarios/high_seismic.bin',
    bin_url: './scenarios/high_seismic.bin',
    jsonUrl: './scenarios/high_seismic.json',
    json_url: './scenarios/high_seismic.json',
    tags: ['seismic', 'water-hammer', 'transient'],
  },
];

export const DEFAULT_MANIFEST: ScenarioManifest = {
  version: '1.0.0',
  defaultScenarioId: 'sample',
  scenarios: DEFAULT_SCENARIOS,
};

export async function fetchScenarioManifest(manifestUrl: string = './scenarios/manifest.json'): Promise<ScenarioManifest> {
  try {
    const res = await fetch(manifestUrl);
    if (!res.ok) {
      console.warn(`Scenario manifest not found at ${manifestUrl} (status ${res.status}), using default manifest.`);
      return DEFAULT_MANIFEST;
    }
    const manifest = (await res.json()) as ScenarioManifest;
    if (!manifest.scenarios || !Array.isArray(manifest.scenarios) || manifest.scenarios.length === 0) {
      return DEFAULT_MANIFEST;
    }
    return manifest;
  } catch (err) {
    console.warn(`Error fetching scenario manifest from ${manifestUrl}:`, err);
    return DEFAULT_MANIFEST;
  }
}
