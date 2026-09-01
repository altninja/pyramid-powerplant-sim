import * as THREE from 'three';
import { PyramidMesh, CutawayMode, PYRAMID_SPECS } from './scene/pyramid_mesh';
import { CameraController } from './scene/camera_controller';
import { FieldRenderers } from './scene/field_renderers';
import { TelemetryLoader, TelemetryDataset, TelemetryFrame } from './data/telemetry_loader';
import { fetchScenarioManifest, ScenarioInfo } from './data/scenario_manifest';
import { TimelineScrubber } from './ui/timeline_scrubber';
import { TelemetryPlots } from './ui/telemetry_plots';
import { ParameterInspector } from './ui/parameter_inspector';

interface ChamberDetails {
  title: string;
  dimensions: string;
  datum: string;
  volume: string;
  physicsRole: string;
  description: string;
}

const CHAMBER_INFO_MAP: Record<string, ChamberDetails> = {
  'Full Pyramid': {
    title: 'Khufu Outer Envelope & Resonator',
    dimensions: `${PYRAMID_SPECS.baseSide} × ${PYRAMID_SPECS.baseSide} m (440 cu)`,
    datum: `Height: ${PYRAMID_SPECS.height} m (280 cu)`,
    volume: 'Solid: 2,593,283 m³',
    physicsRole: 'Schumann Ground Resonator & Mega Acoustic Horn',
    description:
      'Massive limestone megalithic structure tuned to Schumann planetary harmonics (7.83 Hz) and telluric acoustic wave propagation.',
  },
  'Subterranean Chamber': {
    title: 'Subterranean Chamber & Water Hammer Pit',
    dimensions: `${PYRAMID_SPECS.subterranean.widthEW} × ${PYRAMID_SPECS.subterranean.lengthNS} × ${PYRAMID_SPECS.subterranean.height} m`,
    datum: `Datum: ${PYRAMID_SPECS.subterranean.datum.toFixed(2)} m (Bedrock)`,
    volume: 'Survey Volume: 280 m³',
    physicsRole: 'Hydraulic Ram & Low-Frequency Pulse Generator',
    description:
      'Underground bedrock chamber and 3.2m deep pit designed to convert Nile subterranean water flow into high-pressure infrasonic hydraulic hammer impulses.',
  },
  "Queen's Chamber": {
    title: "Queen's Chamber Chemical Reactor",
    dimensions: `${PYRAMID_SPECS.queensChamber.widthEW} × ${PYRAMID_SPECS.queensChamber.lengthNS} × ${PYRAMID_SPECS.queensChamber.heightApex} m`,
    datum: `Datum: +${PYRAMID_SPECS.queensChamber.datum.toFixed(2)} m`,
    volume: 'Volume: 160 m³',
    physicsRole: 'Hydrogen Gas Generation (Zn + 2HCl → ZnCl2 + H2)',
    description:
      'Gabled chemical synthesis chamber fed via Northern & Southern shafts, generating low-density hydrogen gas (cH2 = 1290 m/s) to tune pyramid acoustic speeds.',
  },
  'Grand Gallery': {
    title: 'Grand Gallery Helmholtz Resonator Rack',
    dimensions: `${PYRAMID_SPECS.grandGallery.widthBase}m Base / ${PYRAMID_SPECS.grandGallery.widthRoof}m Roof × ${PYRAMID_SPECS.grandGallery.inclineLength}m`,
    datum: "Slope: 26° 02' 30\" (Height: 8.60 m)",
    volume: 'Cavity: 550 m³',
    physicsRole: '28 Resonator Pairs converting pulses to 438 Hz (F#)',
    description:
      'Corbelled acoustic waveguide acting as an acoustic amplifier and frequency doubler converting low-frequency shockwaves into a coherent 438 Hz acoustic standing wave.',
  },
  "King's Chamber & Relieving Beams": {
    title: "King's Chamber & 5-Tier Granite Relieving Chambers",
    dimensions: `${PYRAMID_SPECS.kingsChamber.widthEW} × ${PYRAMID_SPECS.kingsChamber.lengthNS} × ${PYRAMID_SPECS.kingsChamber.height} m`,
    datum: `Datum: +${PYRAMID_SPECS.kingsChamber.datum.toFixed(2)} m (43 Beams)`,
    volume: 'Chamber + 5 Tiers: ~1,200 m³',
    physicsRole: 'Piezoelectric Quartz Transduction (Acoustic → Multi-kV)',
    description:
      '43 monolithic rose granite beams (28.5% quartz content) arranged in 5 relieving tiers, vibrating in bending modes under 438 Hz drive to generate high kilovolt potentials.',
  },
  'Shaft Beaming': {
    title: 'Shaft Waveguide Maser Antenna Beams',
    dimensions: 'North: 32° 28\' (65.0m) / South: 45° 00\' (53.3m)',
    datum: 'Directivity Gain: G0 ≈ 9.80 dBi (9.55 linear)',
    volume: 'Aperture: 0.21 × 0.21 m',
    physicsRole: '1.4204 GHz Hydrogen Microwave Maser Emission',
    description:
      'Granite-lined directional waveguide shafts channeling stimulated 21.1 cm microwave maser radiation skyward toward celestial targets with kilowatt-level ERP.',
  },
};

export class ViewerApp {
  private canvas: HTMLCanvasElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private cameraController: CameraController;
  private pyramid: PyramidMesh;
  private fieldRenderers: FieldRenderers;
  private telemetryLoader: TelemetryLoader;
  private timelineScrubber: TimelineScrubber;
  private telemetryPlots: TelemetryPlots;
  private parameterInspector: ParameterInspector;

  private currentDataset: TelemetryDataset | null = null;
  private activePreset: string = 'Full Pyramid';
  private wireframeVisible: boolean = false;
  private lastAnimTime: number = 0;
  private availableScenarios: ScenarioInfo[] = [];

  constructor() {
    this.canvas = document.getElementById('webgl-canvas') as HTMLCanvasElement;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07080c);
    this.scene.fog = new THREE.FogExp2(0x07080c, 0.0022);

    this.camera = new THREE.PerspectiveCamera(
      45,
      window.innerWidth / window.innerHeight,
      0.5,
      1500
    );

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.setupLighting();

    this.pyramid = new PyramidMesh();
    this.scene.add(this.pyramid.group);

    this.fieldRenderers = new FieldRenderers();
    this.scene.add(this.fieldRenderers.group);
    this.fieldRenderers.setCutawayPlanes(this.pyramid.getActiveClippingPlanes());

    this.cameraController = new CameraController(this.camera, this.renderer.domElement);

    this.telemetryLoader = new TelemetryLoader();
    this.timelineScrubber = new TimelineScrubber();

    const plotsContainer = document.getElementById('plots-drawer') as HTMLElement;
    this.telemetryPlots = new TelemetryPlots(plotsContainer);

    const inspectorContainer = document.getElementById('inspector-container') as HTMLElement;
    this.parameterInspector = new ParameterInspector(inspectorContainer);

    this.setupTelemetry();
    this.setupUI();
    this.setupEvents();

    requestAnimationFrame(this.animate);
  }

  private setupLighting(): void {
    const ambientLight = new THREE.AmbientLight(0xd9cbb0, 0.65);
    this.scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xffeedd, 0x111625, 0.85);
    hemiLight.position.set(0, 200, 0);
    this.scene.add(hemiLight);

    const sunLight = new THREE.DirectionalLight(0xfff8eb, 2.4);
    sunLight.position.set(160, 220, 180);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 10;
    sunLight.shadow.camera.far = 600;
    sunLight.shadow.camera.left = -180;
    sunLight.shadow.camera.right = 180;
    sunLight.shadow.camera.top = 180;
    sunLight.shadow.camera.bottom = -180;
    sunLight.shadow.bias = -0.0005;
    this.scene.add(sunLight);

    const fillLight = new THREE.DirectionalLight(0x4a6fa5, 0.7);
    fillLight.position.set(-180, 80, -160);
    this.scene.add(fillLight);
  }

  private setupTelemetry(): void {
    this.timelineScrubber.bindElements({
      playPauseBtn: document.getElementById('btn-timeline-play') as HTMLButtonElement,
      reverseBtn: document.getElementById('btn-timeline-reverse') as HTMLButtonElement,
      forwardBtn: document.getElementById('btn-timeline-forward') as HTMLButtonElement,
      stepBackBtn: document.getElementById('btn-timeline-step-back') as HTMLButtonElement,
      stepForwardBtn: document.getElementById('btn-timeline-step-fwd') as HTMLButtonElement,
      resetBtn: document.getElementById('btn-timeline-reset') as HTMLButtonElement,
      loopBtn: document.getElementById('btn-timeline-loop') as HTMLButtonElement,
      loopRangeBtn: document.getElementById('btn-timeline-loop-range') as HTMLButtonElement,
      setABtn: document.getElementById('btn-range-set-a') as HTMLButtonElement,
      setBBtn: document.getElementById('btn-range-set-b') as HTMLButtonElement,
      rangeTimeA: document.getElementById('range-time-a'),
      rangeTimeB: document.getElementById('range-time-b'),
      loopRangeBar: document.getElementById('loop-range-bar'),
      slider: document.getElementById('timeline-slider') as HTMLInputElement,
      timeReadout: document.getElementById('timeline-time-text'),
      frameReadout: document.getElementById('timeline-frame-text'),
      statusBadge: document.getElementById('status-badge'),
      speedPillsContainer: document.getElementById('speed-pills-container'),
    });

    this.timelineScrubber.onSeek((t: number) => {
      const frame = this.telemetryLoader.getInterpolatedFrame(t);
      const allFrames = this.currentDataset?.frames ?? [];
      this.fieldRenderers.update(frame, 0.016);
      this.telemetryPlots.update(t, frame, allFrames);
      this.parameterInspector.update(frame);
    });

    this.telemetryLoader.onDataLoaded((dataset: TelemetryDataset) => {
      this.currentDataset = dataset;
      this.timelineScrubber.setDuration(dataset.duration, dataset.total_frames);
      this.timelineScrubber.seekTo(0, false);
      const initialFrame = this.telemetryLoader.getInterpolatedFrame(0);
      this.fieldRenderers.update(initialFrame, 0.016);
      this.telemetryPlots.update(0, initialFrame, dataset.frames);
      this.parameterInspector.update(initialFrame);
    });

    const dropOverlay = document.getElementById('drop-overlay') as HTMLElement;
    if (dropOverlay) {
      this.telemetryLoader.setupDragAndDrop(document.body, dropOverlay);
    }

    this.telemetryLoader.loadDefault('./sample_telemetry.json').catch((err) => {
      console.warn('Could not auto-load default telemetry dataset:', err);
    });
  }

  private setupUI(): void {
    const scenarioSelect = document.getElementById('scenario-selector') as HTMLSelectElement;
    if (scenarioSelect) {
      fetchScenarioManifest('./scenarios/manifest.json').then((manifest) => {
        this.availableScenarios = manifest.scenarios;
        scenarioSelect.innerHTML = '';
        manifest.scenarios.forEach((sc) => {
          const opt = document.createElement('option');
          opt.value = sc.id;
          opt.textContent = sc.name;
          scenarioSelect.appendChild(opt);
        });

        if (manifest.defaultScenarioId) {
          scenarioSelect.value = manifest.defaultScenarioId;
        }
      });

      scenarioSelect.addEventListener('change', async () => {
        const scenarioId = scenarioSelect.value;
        const sc = this.availableScenarios.find((s) => s.id === scenarioId);
        if (!sc) return;

        const targetUrl = sc.binUrl || sc.bin_url || sc.jsonUrl || sc.json_url || './sample_telemetry.json';
        const badge = document.getElementById('status-badge');
        if (badge) {
          badge.className = 'badge paused';
          badge.innerHTML = `<span class="badge-dot"></span><span>LOADING ${sc.name}...</span>`;
        }

        try {
          await this.telemetryLoader.load(targetUrl);
          this.timelineScrubber.reset();
          this.timelineScrubber.playForward();
        } catch (err) {
          console.warn(`Failed loading scenario ${sc.name} from ${targetUrl}:`, err);
          if (badge) {
            badge.className = 'badge paused';
            badge.innerHTML = `<span class="badge-dot" style="background:#ef4444"></span><span>SCENARIO NOT FOUND</span>`;
          }
        }
      });
    }

    const bookmarkBtns = [
      { id: 'bm-rxn', event: 'rxn' },
      { id: 'bm-acoustic', event: 'acoustic' },
      { id: 'bm-piezo', event: 'piezo' },
      { id: 'bm-maser', event: 'maser' },
      { id: 'bm-spark', event: 'spark' },
    ];

    bookmarkBtns.forEach(({ id, event }) => {
      const btn = document.getElementById(id);
      btn?.addEventListener('click', () => {
        this.jumpToEvent(event as 'rxn' | 'acoustic' | 'piezo' | 'maser' | 'spark');
        document.querySelectorAll('.bookmark-pill').forEach((b) => b.classList.remove('active-bookmark'));
        btn.classList.add('active-bookmark');
      });
    });

    const btnCutEast = document.getElementById('btn-cut-east');
    const btnCutSouth = document.getElementById('btn-cut-south');
    const btnCutQuad = document.getElementById('btn-cut-quad');
    const btnCutNone = document.getElementById('btn-cut-none');
    const cutButtons = [btnCutEast, btnCutSouth, btnCutQuad, btnCutNone];

    const setCutBtnActive = (activeBtn: HTMLElement | null) => {
      cutButtons.forEach((b) => b?.classList.remove('active'));
      activeBtn?.classList.add('active');
    };

    btnCutEast?.addEventListener('click', () => {
      this.pyramid.setCutawayMode('east');
      this.fieldRenderers.setCutawayPlanes(this.pyramid.getActiveClippingPlanes());
      setCutBtnActive(btnCutEast);
    });

    btnCutSouth?.addEventListener('click', () => {
      this.pyramid.setCutawayMode('south');
      this.fieldRenderers.setCutawayPlanes(this.pyramid.getActiveClippingPlanes());
      setCutBtnActive(btnCutSouth);
    });

    btnCutQuad?.addEventListener('click', () => {
      this.pyramid.setCutawayMode('quadrant');
      this.fieldRenderers.setCutawayPlanes(this.pyramid.getActiveClippingPlanes());
      setCutBtnActive(btnCutQuad);
    });

    btnCutNone?.addEventListener('click', () => {
      this.pyramid.setCutawayMode('none');
      this.fieldRenderers.setCutawayPlanes(this.pyramid.getActiveClippingPlanes());
      setCutBtnActive(btnCutNone);
    });

    const sliderOpacity = document.getElementById('slider-opacity') as HTMLInputElement;
    const opacityVal = document.getElementById('opacity-val');
    sliderOpacity?.addEventListener('input', () => {
      const val = parseFloat(sliderOpacity.value) / 100.0;
      if (opacityVal) opacityVal.textContent = `${sliderOpacity.value}%`;
      this.pyramid.setOuterCasingOpacity(val);
    });

    const btnToggleWireframe = document.getElementById('btn-toggle-wireframe');
    btnToggleWireframe?.addEventListener('click', () => {
      this.wireframeVisible = !this.wireframeVisible;
      this.pyramid.setWireframeVisibility(this.wireframeVisible);
      btnToggleWireframe.classList.toggle('active', this.wireframeVisible);
    });

    const layerAcoustic = document.getElementById('layer-acoustic') as HTMLInputElement;
    layerAcoustic?.addEventListener('change', () => {
      this.fieldRenderers.setAcousticVisible(layerAcoustic.checked);
    });

    const layerHydrogen = document.getElementById('layer-hydrogen') as HTMLInputElement;
    layerHydrogen?.addEventListener('change', () => {
      this.fieldRenderers.setHydrogenVisible(layerHydrogen.checked);
    });

    const layerPiezo = document.getElementById('layer-piezo') as HTMLInputElement;
    layerPiezo?.addEventListener('change', () => {
      this.fieldRenderers.setPiezoVisible(layerPiezo.checked);
    });

    const layerMicrowave = document.getElementById('layer-microwave') as HTMLInputElement;
    layerMicrowave?.addEventListener('change', () => {
      this.fieldRenderers.setMicrowaveVisible(layerMicrowave.checked);
    });

    const layerHydraulic = document.getElementById('layer-hydraulic') as HTMLInputElement;
    layerHydraulic?.addEventListener('change', () => {
      this.fieldRenderers.setHydraulicVisible(layerHydraulic.checked);
    });

    document.querySelectorAll('[data-preset]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const preset = btn.getAttribute('data-preset');
        if (preset) this.switchPreset(preset);
      });
    });

    const fileInput = document.getElementById('file-input-json') as HTMLInputElement;
    const btnUpload = document.getElementById('btn-upload-json');
    btnUpload?.addEventListener('click', () => {
      fileInput?.click();
    });

    fileInput?.addEventListener('change', async () => {
      if (fileInput.files && fileInput.files.length > 0) {
        try {
          await this.telemetryLoader.load(fileInput.files[0]);
        } catch (err) {
          console.error('Failed to load file:', err);
        }
      }
    });

    const btnTogglePlots = document.getElementById('btn-toggle-oscilloscopes');
    btnTogglePlots?.addEventListener('click', () => {
      this.telemetryPlots.toggleCollapsed();
      btnTogglePlots.classList.toggle('active', this.telemetryPlots.isVisible());
    });

    const btnToggleInspector = document.getElementById('btn-toggle-inspector');
    btnToggleInspector?.addEventListener('click', () => {
      this.parameterInspector.toggleCollapsed();
      btnToggleInspector.classList.toggle('active', this.parameterInspector.isVisible());
    });

    const btnToggleInterp = document.getElementById('btn-toggle-interp');
    const interpLabel = document.getElementById('interp-label');
    btnToggleInterp?.addEventListener('click', () => {
      const cur = this.telemetryLoader.getInterpolationMode();
      const next = cur === 'hermite' ? 'linear' : 'hermite';
      this.telemetryLoader.setInterpolationMode(next);
      if (interpLabel) {
        interpLabel.textContent = next === 'hermite' ? 'Hermite Sub-frame' : 'Linear Sub-frame';
      }
      btnToggleInterp.classList.toggle('btn-cyan', next === 'hermite');
    });

    const btnInfoClose = document.getElementById('btn-info-close');
    const infoPanel = document.getElementById('info-panel');
    btnInfoClose?.addEventListener('click', () => {
      if (infoPanel) infoPanel.style.display = 'none';
    });
  }

  public jumpToEvent(eventName: 'rxn' | 'acoustic' | 'piezo' | 'maser' | 'spark'): void {
    const frames = this.currentDataset?.frames ?? [];
    let targetTime = 0.0;
    let targetPreset = 'Full Pyramid';

    if (eventName === 'rxn') {
      targetPreset = "Queen's Chamber";
      targetTime = 0.1;
      for (const f of frames) {
        if (f.chemical_reaction_rate > 0 || f.h2_mole_fraction_qc > 0.0001) {
          targetTime = f.time;
          break;
        }
      }
    } else if (eventName === 'acoustic') {
      targetPreset = 'Grand Gallery';
      targetTime = 1.0;
      let maxP = 0;
      for (const f of frames) {
        if (f.gallery_peak_pressure > maxP) {
          maxP = f.gallery_peak_pressure;
          targetTime = f.time;
        }
      }
    } else if (eventName === 'piezo') {
      targetPreset = "King's Chamber & Relieving Beams";
      targetTime = 1.5;
      let maxV = 0;
      for (const f of frames) {
        const v = Math.abs(f.total_piezo_voltage);
        if (v > maxV) {
          maxV = v;
          targetTime = f.time;
        }
      }
    } else if (eventName === 'maser') {
      targetPreset = 'Shaft Beaming';
      targetTime = 2.0;
      let foundThreshold = false;
      for (const f of frames) {
        if (f.maser_is_above_threshold || f.maser_total_radiated_power > 0.001) {
          targetTime = f.time;
          foundThreshold = true;
          break;
        }
      }
      if (!foundThreshold) {
        let maxPwr = 0;
        for (const f of frames) {
          if (f.maser_total_radiated_power > maxPwr) {
            maxPwr = f.maser_total_radiated_power;
            targetTime = f.time;
          }
        }
      }
    } else if (eventName === 'spark') {
      targetPreset = "King's Chamber & Relieving Beams";
      targetTime = 2.5;
      let foundSpark = false;
      for (const f of frames) {
        if (f.spark_triggered) {
          targetTime = f.time;
          foundSpark = true;
          break;
        }
      }
      if (!foundSpark) {
        let maxV = 0;
        for (const f of frames) {
          const v = Math.abs(f.total_piezo_voltage);
          if (v > maxV) {
            maxV = v;
            targetTime = f.time;
          }
        }
      }
    }

    this.switchPreset(targetPreset);
    this.timelineScrubber.seekTo(targetTime, false);
  }

  public getActivePreset(): string {
    return this.activePreset;
  }

  public switchPreset(presetName: string): void {
    if (this.cameraController.flyToPreset(presetName)) {
      this.activePreset = presetName;

      document.querySelectorAll('[data-preset]').forEach((b) => {
        if (b.getAttribute('data-preset') === presetName) {
          b.classList.add('active');
        } else {
          b.classList.remove('active');
        }
      });

      this.updateInfoPanel(presetName);
    }
  }

  private updateInfoPanel(presetName: string): void {
    const info = CHAMBER_INFO_MAP[presetName];
    if (!info) return;

    const heading = document.getElementById('info-heading');
    const desc = document.getElementById('info-description');
    const v0 = document.getElementById('info-val-0');
    const v1 = document.getElementById('info-val-1');
    const v2 = document.getElementById('info-val-2');
    const v3 = document.getElementById('info-val-3');
    const panel = document.getElementById('info-panel');

    if (heading) heading.textContent = info.title;
    if (desc) desc.textContent = info.description;
    if (v0) v0.textContent = info.dimensions;
    if (v1) v1.textContent = info.datum;
    if (v2) v2.textContent = info.physicsRole;
    if (v3) v3.textContent = info.volume;
    if (panel) panel.style.display = 'flex';
  }

  private setupEvents(): void {
    window.addEventListener('resize', () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    });

    window.addEventListener('keydown', (e) => {
      const target = e.target as HTMLElement;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) return;

      const key = e.key;
      if (key === '1') this.switchPreset('Full Pyramid');
      else if (key === '2') this.switchPreset('Subterranean Chamber');
      else if (key === '3') this.switchPreset("Queen's Chamber");
      else if (key === '4') this.switchPreset('Grand Gallery');
      else if (key === '5') this.switchPreset("King's Chamber & Relieving Beams");
      else if (key === '6') this.switchPreset('Shaft Beaming');
      else if (key.toLowerCase() === 'c') {
        const currentPlanes = this.pyramid.getActiveClippingPlanes().length;
        const nextMode: CutawayMode = currentPlanes === 1 ? 'south' : currentPlanes === 2 ? 'none' : 'east';
        this.pyramid.setCutawayMode(nextMode);
        this.fieldRenderers.setCutawayPlanes(this.pyramid.getActiveClippingPlanes());
      } else if (key.toLowerCase() === 'o') {
        this.telemetryPlots.toggleCollapsed();
      } else if (key.toLowerCase() === 'i') {
        this.parameterInspector.toggleCollapsed();
      } else if (key.toLowerCase() === 'w') {
        this.wireframeVisible = !this.wireframeVisible;
        this.pyramid.setWireframeVisibility(this.wireframeVisible);
      } else if (key.toLowerCase() === 'h') {
        const panel = document.getElementById('info-panel');
        if (panel) {
          panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
        }
      }
    });
  }

  private animate = (now: number): void => {
    requestAnimationFrame(this.animate);

    const animTime = now * 0.001;
    const dt = Math.min(this.lastAnimTime > 0 ? animTime - this.lastAnimTime : 0.016, 0.1);
    this.lastAnimTime = animTime;

    this.cameraController.update();

    const t = this.timelineScrubber.tick(dt);
    const frame: TelemetryFrame | null = this.telemetryLoader.getInterpolatedFrame(t);

    this.fieldRenderers.update(frame, dt);

    const allFrames = this.currentDataset?.frames ?? [];
    this.telemetryPlots.update(t, frame, allFrames);
    this.parameterInspector.update(frame);

    this.renderer.render(this.scene, this.camera);
  };
}

window.addEventListener('DOMContentLoaded', () => {
  new ViewerApp();
});
