import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface CameraPreset {
  name: string;
  position: THREE.Vector3;
  target: THREE.Vector3;
  fov?: number;
  description?: string;
}

export class CameraController {
  public camera: THREE.PerspectiveCamera;
  public controls: OrbitControls;
  public presets: Map<string, CameraPreset> = new Map();

  private isTransitioning: boolean = false;
  private transitionStart: number = 0;
  private transitionDuration: number = 1.2;

  private startPos: THREE.Vector3 = new THREE.Vector3();
  private targetPos: THREE.Vector3 = new THREE.Vector3();
  private startTarget: THREE.Vector3 = new THREE.Vector3();
  private targetLookAt: THREE.Vector3 = new THREE.Vector3();

  private onTransitionCompleteCallback?: () => void;

  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement) {
    this.camera = camera;
    this.controls = new OrbitControls(camera, domElement);

    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 2.0;
    this.controls.maxDistance = 600.0;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.15;

    this.setupPresets();

    const defaultPreset = this.presets.get('Full Pyramid');
    if (defaultPreset) {
      this.camera.position.copy(defaultPreset.position);
      this.controls.target.copy(defaultPreset.target);
      this.controls.update();
    }
  }

  private setupPresets(): void {
    this.presets.set('Full Pyramid', {
      name: 'Full Pyramid',
      position: new THREE.Vector3(180.0, 150.0, 220.0),
      target: new THREE.Vector3(0.0, 45.0, 0.0),
      fov: 45,
      description: 'Isometric overview of Khufu Pyramid and internal network',
    });

    this.presets.set('Subterranean Chamber', {
      name: 'Subterranean Chamber',
      position: new THREE.Vector3(22.0, -18.0, -12.0),
      target: new THREE.Vector3(0.0, -28.0, -27.4),
      fov: 50,
      description: 'Hydraulic ram water hammer and excavated bedrock pit',
    });

    this.presets.set("Queen's Chamber", {
      name: "Queen's Chamber",
      position: new THREE.Vector3(14.0, 27.0, 14.0),
      target: new THREE.Vector3(0.0, 24.0, 0.5),
      fov: 45,
      description: 'Chemical hydrogen reactor and 5-tier corbel niche',
    });

    this.presets.set('Grand Gallery', {
      name: 'Grand Gallery',
      position: new THREE.Vector3(16.0, 36.0, 2.0),
      target: new THREE.Vector3(0.0, 30.0, 16.0),
      fov: 45,
      description: '7-step corbelled acoustic waveguide and 28 resonator slots',
    });

    this.presets.set("King's Chamber & Relieving Beams", {
      name: "King's Chamber & Relieving Beams",
      position: new THREE.Vector3(26.0, 54.0, 28.0),
      target: new THREE.Vector3(0.0, 50.0, 15.0),
      fov: 45,
      description: 'Piezoelectric quartz transducers and 43 granite beams',
    });

    this.presets.set('Shaft Beaming', {
      name: 'Shaft Beaming',
      position: new THREE.Vector3(80.0, 130.0, -130.0),
      target: new THREE.Vector3(0.0, 65.0, 0.0),
      fov: 50,
      description: 'Microwave maser stimulated emission and skyward horn beaming',
    });
  }

  public getPresetNames(): string[] {
    return Array.from(this.presets.keys());
  }

  public getPreset(name: string): CameraPreset | undefined {
    return this.presets.get(name);
  }

  public flyToPreset(
    presetName: string,
    durationSec: number = 1.2,
    onComplete?: () => void
  ): boolean {
    const preset = this.presets.get(presetName);
    if (!preset) return false;

    this.flyTo(preset.position, preset.target, durationSec, onComplete);
    return true;
  }

  public flyTo(
    targetPosition: THREE.Vector3,
    targetLookAt: THREE.Vector3,
    durationSec: number = 1.2,
    onComplete?: () => void
  ): void {
    this.startPos.copy(this.camera.position);
    this.targetPos.copy(targetPosition);

    this.startTarget.copy(this.controls.target);
    this.targetLookAt.copy(targetLookAt);

    this.transitionDuration = Math.max(0.1, durationSec);
    this.transitionStart = performance.now() / 1000.0;
    this.isTransitioning = true;
    this.onTransitionCompleteCallback = onComplete;

    this.controls.enabled = false;
  }

  private easeInOutCubic(t: number): number {
    return t < 0.5 ? 4.0 * t * t * t : 1.0 - Math.pow(-2.0 * t + 2.0, 3.0) / 2.0;
  }

  public update(): void {
    if (this.isTransitioning) {
      const now = performance.now() / 1000.0;
      const elapsed = now - this.transitionStart;
      const progress = Math.min(1.0, elapsed / this.transitionDuration);
      const easeT = this.easeInOutCubic(progress);

      this.camera.position.lerpVectors(this.startPos, this.targetPos, easeT);
      this.controls.target.lerpVectors(this.startTarget, this.targetLookAt, easeT);

      if (progress >= 1.0) {
        this.isTransitioning = false;
        this.controls.enabled = true;
        if (this.onTransitionCompleteCallback) {
          this.onTransitionCompleteCallback();
          this.onTransitionCompleteCallback = undefined;
        }
      }
    }

    this.controls.update();
  }

  public reset(): void {
    this.flyToPreset('Full Pyramid', 0.8);
  }

  public dispose(): void {
    this.controls.dispose();
  }
}
