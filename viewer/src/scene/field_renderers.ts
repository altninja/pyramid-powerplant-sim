import * as THREE from 'three';
import { PYRAMID_SPECS, engineToThree } from './pyramid_mesh';

const ACOUSTIC_HEATMAP_VERTEX = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying float vAlongZ;

  uniform float uTime;
  uniform float uAcousticEnergy;
  uniform float uPeakPressure;

  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vAlongZ = uv.y;

    float wavePhase = uv.y * 37.699 + uTime * 27.5;
    float displacement = sin(wavePhase) * (uPeakPressure / 5.0e6) * 0.06;
    vec3 displacedPosition = position + normal * displacement;

    vec4 worldPos = modelMatrix * vec4(displacedPosition, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const ACOUSTIC_HEATMAP_FRAGMENT = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying float vAlongZ;

  uniform float uTime;
  uniform float uAcousticEnergy;
  uniform float uPeakPressure;
  uniform float uRmsPressure;
  uniform float uSoundSpeed;
  uniform vec3 uColorNode;
  uniform vec3 uColorMid;
  uniform vec3 uColorAntinode;

  vec3 pressureColormap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.02, 0.15, 0.65);
    vec3 c1 = vec3(0.0, 0.85, 0.95);
    vec3 c2 = vec3(0.1, 0.95, 0.35);
    vec3 c3 = vec3(1.0, 0.78, 0.05);
    vec3 c4 = vec3(1.0, 0.12, 0.02);

    if (t < 0.25) return mix(c0, c1, t / 0.25);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    return mix(c3, c4, (t - 0.75) / 0.25);
  }

  void main() {
    if (uPeakPressure <= 0.001) {
      discard;
    }

    float k = 37.69911;
    float omega = 27.52;
    float standingWave = sin(vAlongZ * k) * cos(uTime * omega);
    float pressureMag = abs(standingWave);

    float normalizedIntensity = clamp(uPeakPressure / 2.5e6, 0.0, 1.0);
    float heatValue = pressureMag * (0.35 + 0.65 * normalizedIntensity);

    vec3 baseColor = pressureColormap(heatValue);

    float isobars = sin(pressureMag * 31.4159);
    float fringe = smoothstep(0.7, 0.95, isobars) * 0.35;
    baseColor += vec3(fringe);

    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
    float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 2.2);

    float alpha = (0.28 + 0.62 * pressureMag + fresnel * 0.4) * normalizedIntensity;
    alpha = clamp(alpha, 0.0, 0.92);

    gl_FragColor = vec4(baseColor, alpha);
  }
`;

const HYDROGEN_POINTS_VERTEX = `
  attribute float aSize;
  attribute float aAlpha;
  attribute vec3 aColor;
  attribute float aSeed;

  varying vec3 vColor;
  varying float vAlpha;

  uniform float uTime;
  uniform float uReactionRate;
  uniform float uGlobalAlpha;

  void main() {
    vColor = aColor;
    
    vec3 pos = position;
    float jitterSpeed = 3.0 + uReactionRate * 0.15;
    pos.x += sin(uTime * jitterSpeed + aSeed * 6.28) * 0.08;
    pos.z += cos(uTime * jitterSpeed * 0.9 + aSeed * 12.56) * 0.08;

    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    vAlpha = aAlpha * uGlobalAlpha;

    gl_PointSize = aSize * (260.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const HYDROGEN_POINTS_FRAGMENT = `
  varying vec3 vColor;
  varying float vAlpha;

  void main() {
    if (vAlpha <= 0.005) discard;

    vec2 centerCoord = gl_PointCoord - vec2(0.5);
    float distSq = dot(centerCoord, centerCoord);
    if (distSq > 0.25) discard;

    float glow = exp(-distSq * 14.0);
    float core = smoothstep(0.18, 0.0, distSq);

    vec3 finalColor = vColor + vec3(core * 0.45);
    float finalAlpha = vAlpha * glow;

    gl_FragColor = vec4(finalColor, finalAlpha);
  }
`;

const PIEZO_CORONA_VERTEX = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPos;

  uniform float uTime;
  uniform float uVoltage;

  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    
    float pulse = sin(uTime * 35.0 + position.x * 2.0 + position.y * 3.0) * 0.04;
    float voltScale = clamp(uVoltage / 20000.0, 0.0, 1.5);
    vec3 displaced = position + normal * (pulse * voltScale);

    vec4 worldPos = modelMatrix * vec4(displaced, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const PIEZO_CORONA_FRAGMENT = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPos;

  uniform float uTime;
  uniform float uVoltage;
  uniform float uStress;
  uniform vec3 uColorBase;
  uniform vec3 uColorArc;

  void main() {
    if (uVoltage <= 10.0) discard;

    float normVolt = clamp(uVoltage / 25000.0, 0.0, 1.2);
    
    float noise1 = sin(vUv.x * 40.0 + uTime * 45.0) * cos(vUv.y * 30.0 - uTime * 38.0);
    float noise2 = sin(vWorldPos.x * 8.0 + vWorldPos.y * 12.0 + uTime * 50.0);
    float arcStreak = smoothstep(0.65, 0.98, abs(noise1 * noise2));

    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    float fresnel = pow(1.0 - abs(dot(viewDir, vNormal)), 2.5);

    vec3 color = mix(uColorBase, uColorArc, arcStreak * 0.85);
    color += vec3(fresnel * 0.6) * uColorArc;

    float alpha = (fresnel * 0.75 + arcStreak * 0.55) * normVolt;
    alpha = clamp(alpha, 0.0, 0.88);

    gl_FragColor = vec4(color, alpha);
  }
`;

const MICROWAVE_BEAM_VERTEX = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  varying float vAlongRay;

  uniform float uTime;
  uniform float uBeamPower;

  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vAlongRay = uv.y;

    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const MICROWAVE_BEAM_FRAGMENT = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  varying float vAlongRay;

  uniform float uTime;
  uniform float uBeamPower;
  uniform vec3 uBeamColorCore;
  uniform vec3 uBeamColorSheath;

  void main() {
    if (uBeamPower <= 1e-15) discard;

    float normPower = clamp(uBeamPower > 0.0 ? (log10(max(uBeamPower, 1e-20)) + 20.0) / 20.0 : 0.0, 0.0, 1.0);
    if (uBeamPower > 0.001) {
      normPower = clamp(0.5 + 0.5 * (uBeamPower / 1000.0), 0.5, 1.0);
    }
    normPower = max(normPower, clamp(uBeamPower * 100.0, 0.0, 1.0));

    float wavePhase = vAlongRay * 142.0 - uTime * 48.0;
    float phaseRings = sin(wavePhase);
    float ringIntensity = smoothstep(0.4, 0.98, phaseRings);

    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    float edgeGlow = pow(1.0 - abs(dot(viewDir, vNormal)), 2.8);

    float coreLine = exp(-pow((vUv.x - 0.5) * 6.0, 2.0));

    vec3 color = mix(uBeamColorSheath, uBeamColorCore, coreLine);
    color += vec3(ringIntensity * 0.4) * uBeamColorCore;

    float fadeSky = smoothstep(1.0, 0.15, vAlongRay);
    float alpha = (coreLine * 0.85 + edgeGlow * 0.5 + ringIntensity * 0.35) * fadeSky * normPower;
    alpha = clamp(alpha, 0.0, 0.95);

    gl_FragColor = vec4(color, alpha);
  }
`;

const HYDRAULIC_SHOCKWAVE_VERTEX = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPos;

  uniform float uTime;
  uniform float uPressure;
  uniform float uRadius;

  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const HYDRAULIC_SHOCKWAVE_FRAGMENT = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPos;

  uniform float uTime;
  uniform float uPressure;
  uniform float uDisplacement;
  uniform vec3 uWaveColor;

  void main() {
    if (uPressure <= 10.0 && abs(uDisplacement) <= 1e-7) discard;

    float normPress = clamp(uPressure / 2.0e6, 0.0, 1.0);
    if (normPress < 0.05 && abs(uDisplacement) > 1e-6) {
      normPress = clamp(abs(uDisplacement) * 5000.0, 0.05, 0.8);
    }

    float distCenter = length(vUv - vec2(0.5)) * 2.0;
    if (distCenter > 1.0) discard;

    float rings = sin(distCenter * 25.1327 - uTime * 18.0);
    float ringCrest = smoothstep(0.4, 0.95, rings);

    float edgeFade = smoothstep(1.0, 0.2, distCenter);
    float alpha = ringCrest * edgeFade * normPress * 0.75;

    vec3 color = uWaveColor + vec3(ringCrest * 0.35);

    gl_FragColor = vec4(color, alpha);
  }
`;

export class AcousticStandingWaveRenderer {
  public group: THREE.Group;
  private galleryRibbonMesh: THREE.Mesh;
  private galleryUniforms: Record<string, THREE.IUniform>;
  private kingsChamberMesh: THREE.Mesh;
  private kingsUniforms: Record<string, THREE.IUniform>;
  private kingsPointLight: THREE.PointLight;
  private slotNodesGroup: THREE.Group;
  private slotNodeMeshes: THREE.Mesh[] = [];

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'AcousticStandingWaveGroup';

    const gg = PYRAMID_SPECS.grandGallery;
    const ribbonLength = gg.inclineLength;
    const ribbonWidth = gg.centralTrenchWidth * 1.8;
    const ribbonHeight = gg.verticalHeight * 0.75;

    const ribbonGeo = new THREE.PlaneGeometry(ribbonWidth, ribbonLength, 32, 128);
    ribbonGeo.rotateX(Math.PI / 2);

    this.galleryUniforms = {
      uTime: { value: 0.0 },
      uAcousticEnergy: { value: 0.0 },
      uPeakPressure: { value: 0.0 },
      uRmsPressure: { value: 0.0 },
      uSoundSpeed: { value: 343.2 },
      uColorNode: { value: new THREE.Color('#0033cc') },
      uColorMid: { value: new THREE.Color('#00ffaa') },
      uColorAntinode: { value: new THREE.Color('#ff3300') },
    };

    const galleryMat = new THREE.ShaderMaterial({
      vertexShader: ACOUSTIC_HEATMAP_VERTEX,
      fragmentShader: ACOUSTIC_HEATMAP_FRAGMENT,
      uniforms: this.galleryUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    this.galleryRibbonMesh = new THREE.Mesh(ribbonGeo, galleryMat);
    this.galleryRibbonMesh.name = 'GrandGalleryAcousticRibbon';

    const ggStart = engineToThree(gg.start.x, gg.start.ns, gg.start.elev);
    const rad = THREE.MathUtils.degToRad(gg.slopeAngleDeg);

    const galleryContainer = new THREE.Group();
    galleryContainer.position.copy(ggStart);
    galleryContainer.rotation.x = rad;

    this.galleryRibbonMesh.position.set(0, ribbonHeight * 0.5, ribbonLength / 2.0);
    galleryContainer.add(this.galleryRibbonMesh);

    const verticalSliceGeo = new THREE.PlaneGeometry(ribbonHeight, ribbonLength, 16, 128);
    verticalSliceGeo.rotateY(Math.PI / 2);
    verticalSliceGeo.rotateZ(Math.PI / 2);
    const vertSlice = new THREE.Mesh(verticalSliceGeo, galleryMat);
    vertSlice.position.set(0, ribbonHeight * 0.5, ribbonLength / 2.0);
    galleryContainer.add(vertSlice);

    this.slotNodesGroup = new THREE.Group();
    const nodeSphereGeo = new THREE.SphereGeometry(0.18, 12, 12);
    const nodeMat = new THREE.MeshBasicMaterial({
      color: 0x00f2fe,
      transparent: true,
      opacity: 0.7,
      blending: THREE.AdditiveBlending,
    });

    for (let i = 0; i < gg.slotPairs; i++) {
      const sPos = i * gg.slotSpacing + 0.8;
      if (sPos > ribbonLength - 0.5) break;

      const westNode = new THREE.Mesh(nodeSphereGeo, nodeMat.clone());
      westNode.position.set(-gg.centralTrenchWidth / 2.0 - gg.sideRampsWidth / 2.0, 0.25, sPos);
      this.slotNodesGroup.add(westNode);
      this.slotNodeMeshes.push(westNode);

      const eastNode = new THREE.Mesh(nodeSphereGeo, nodeMat.clone());
      eastNode.position.set(gg.centralTrenchWidth / 2.0 + gg.sideRampsWidth / 2.0, 0.25, sPos);
      this.slotNodesGroup.add(eastNode);
      this.slotNodeMeshes.push(eastNode);
    }
    galleryContainer.add(this.slotNodesGroup);

    this.group.add(galleryContainer);

    const kc = PYRAMID_SPECS.kingsChamber;
    const kcCenter = engineToThree(kc.centerX, kc.centerNS, kc.datum + kc.height / 2.0);
    const kcBoxGeo = new THREE.BoxGeometry(kc.widthEW * 0.96, kc.height * 0.92, kc.lengthNS * 0.96);

    this.kingsUniforms = {
      uTime: { value: 0.0 },
      uAcousticEnergy: { value: 0.0 },
      uPeakPressure: { value: 0.0 },
      uRmsPressure: { value: 0.0 },
      uSoundSpeed: { value: 343.2 },
      uColorNode: { value: new THREE.Color('#002288') },
      uColorMid: { value: new THREE.Color('#00e5ff') },
      uColorAntinode: { value: new THREE.Color('#ffaa00') },
    };

    const kcMat = new THREE.ShaderMaterial({
      vertexShader: ACOUSTIC_HEATMAP_VERTEX,
      fragmentShader: ACOUSTIC_HEATMAP_FRAGMENT,
      uniforms: this.kingsUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
    });

    this.kingsChamberMesh = new THREE.Mesh(kcBoxGeo, kcMat);
    this.kingsChamberMesh.position.copy(kcCenter);
    this.kingsChamberMesh.name = 'KingsChamberAcousticVolume';
    this.group.add(this.kingsChamberMesh);

    this.kingsPointLight = new THREE.PointLight(0xffaa44, 0.0, 20.0, 1.5);
    this.kingsPointLight.position.copy(kcCenter);
    this.group.add(this.kingsPointLight);
  }

  public update(frame: any, _deltaTime: number): void {
    const time = frame?.time ?? 0.0;
    const peakP = frame?.gallery_peak_pressure ?? 0.0;
    const rmsP = frame?.gallery_rms_pressure ?? 0.0;
    const energy = frame?.gallery_total_acoustic_energy ?? 0.0;
    const soundSpeed = frame?.gallery_sound_speed_avg ?? 343.2;
    const topKcEntryP = frame?.top_pressure_kc_entry ?? frame?.antechamber_p_trans ?? peakP * 0.05;

    this.galleryUniforms.uTime.value = time;
    this.galleryUniforms.uPeakPressure.value = peakP;
    this.galleryUniforms.uRmsPressure.value = rmsP;
    this.galleryUniforms.uAcousticEnergy.value = energy;
    this.galleryUniforms.uSoundSpeed.value = soundSpeed;

    this.kingsUniforms.uTime.value = time;
    this.kingsUniforms.uPeakPressure.value = Math.max(topKcEntryP, peakP * 0.02);
    this.kingsUniforms.uRmsPressure.value = rmsP * 0.05;
    this.kingsUniforms.uAcousticEnergy.value = energy * 0.1;
    this.kingsUniforms.uSoundSpeed.value = soundSpeed;

    const normP = Math.min(peakP / 3.0e6, 1.0);
    this.kingsPointLight.intensity = normP * 3.5;
    this.kingsPointLight.color.setHSL(0.08 + 0.06 * (1.0 - normP), 0.95, 0.55);

    const nodeCount = this.slotNodeMeshes.length;
    for (let i = 0; i < nodeCount; i++) {
      const mesh = this.slotNodeMeshes[i];
      const mat = mesh.material as THREE.MeshBasicMaterial;
      const phase = (i / nodeCount) * 12.566 + time * 27.5;
      const antinodeAmp = Math.abs(Math.sin(phase)) * normP;
      mat.opacity = 0.15 + antinodeAmp * 0.8;
      const s = 1.0 + antinodeAmp * 1.5;
      mesh.scale.set(s, s, s);
    }
  }

  public setCutawayPlanes(planes: THREE.Plane[]): void {
    (this.galleryRibbonMesh.material as THREE.ShaderMaterial).clippingPlanes = planes;
    (this.kingsChamberMesh.material as THREE.ShaderMaterial).clippingPlanes = planes;
  }

  public dispose(): void {
    this.galleryRibbonMesh.geometry.dispose();
    (this.galleryRibbonMesh.material as THREE.Material).dispose();
    this.kingsChamberMesh.geometry.dispose();
    (this.kingsChamberMesh.material as THREE.Material).dispose();
    this.slotNodeMeshes.forEach((m) => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
  }
}

export class HydrogenGasVisualizer {
  public group: THREE.Group;
  private particleCount: number = 2200;
  private points: THREE.Points;
  private pointsGeo: THREE.BufferGeometry;
  private pointsMat: THREE.ShaderMaterial;
  private positions: Float32Array;
  private colors: Float32Array;
  private sizes: Float32Array;
  private alphas: Float32Array;
  private seeds: Float32Array;

  private particleRegions: Uint8Array;
  private particleProgress: Float32Array;
  private particleSpeed: Float32Array;
  private particleOffsets: Float32Array;

  private qcMistMesh: THREE.Mesh;
  private qcMistMat: THREE.MeshBasicMaterial;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'HydrogenGasGroup';

    const count = this.particleCount;
    this.positions = new Float32Array(count * 3);
    this.colors = new Float32Array(count * 3);
    this.sizes = new Float32Array(count);
    this.alphas = new Float32Array(count);
    this.seeds = new Float32Array(count);

    this.particleRegions = new Uint8Array(count);
    this.particleProgress = new Float32Array(count);
    this.particleSpeed = new Float32Array(count);
    this.particleOffsets = new Float32Array(count * 3);

    const qc = PYRAMID_SPECS.queensChamber;

    const baseColorCyan = new THREE.Color('#00f2fe');
    const baseColorEmerald = new THREE.Color('#00ffb4');

    for (let i = 0; i < count; i++) {
      this.seeds[i] = Math.random();
      this.particleProgress[i] = Math.random();
      this.particleSpeed[i] = 0.2 + Math.random() * 0.8;
      this.sizes[i] = 0.4 + Math.random() * 0.8;
      this.alphas[i] = 0.2 + Math.random() * 0.7;

      const r = Math.random();
      if (r < 0.45) {
        this.particleRegions[i] = 0;
      } else if (r < 0.60) {
        this.particleRegions[i] = 1;
      } else if (r < 0.80) {
        this.particleRegions[i] = 2;
      } else if (r < 0.95) {
        this.particleRegions[i] = 3;
      } else {
        this.particleRegions[i] = 4;
      }

      this.particleOffsets[i * 3 + 0] = (Math.random() - 0.5) * 2.0;
      this.particleOffsets[i * 3 + 1] = (Math.random() - 0.5) * 1.5;
      this.particleOffsets[i * 3 + 2] = (Math.random() - 0.5) * 2.0;

      const col = Math.random() > 0.4 ? baseColorEmerald : baseColorCyan;
      this.colors[i * 3 + 0] = col.r;
      this.colors[i * 3 + 1] = col.g;
      this.colors[i * 3 + 2] = col.b;
    }

    this.pointsGeo = new THREE.BufferGeometry();
    this.pointsGeo.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.pointsGeo.setAttribute('aColor', new THREE.BufferAttribute(this.colors, 3));
    this.pointsGeo.setAttribute('aSize', new THREE.BufferAttribute(this.sizes, 1));
    this.pointsGeo.setAttribute('aAlpha', new THREE.BufferAttribute(this.alphas, 1));
    this.pointsGeo.setAttribute('aSeed', new THREE.BufferAttribute(this.seeds, 1));

    this.pointsMat = new THREE.ShaderMaterial({
      vertexShader: HYDROGEN_POINTS_VERTEX,
      fragmentShader: HYDROGEN_POINTS_FRAGMENT,
      uniforms: {
        uTime: { value: 0.0 },
        uReactionRate: { value: 0.0 },
        uGlobalAlpha: { value: 0.0 },
      },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.points = new THREE.Points(this.pointsGeo, this.pointsMat);
    this.points.name = 'HydrogenParticlesPoints';
    this.group.add(this.points);

    const qcMistGeo = new THREE.BoxGeometry(qc.widthEW * 0.9, qc.heightApex * 0.8, qc.lengthNS * 0.9);
    this.qcMistMat = new THREE.MeshBasicMaterial({
      color: 0x00ffb4,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.qcMistMesh = new THREE.Mesh(qcMistGeo, this.qcMistMat);
    const qcCenter = engineToThree(qc.centerX, qc.centerNS, qc.datum + qc.heightApex / 2.0);
    this.qcMistMesh.position.copy(qcCenter);
    this.qcMistMesh.name = 'QueensChamberHydrogenMist';
    this.group.add(this.qcMistMesh);

    this.evaluateParticlePositions(0.01, 1.0);
  }

  private evaluateParticlePositions(dt: number, reactionRate: number): void {
    const qc = PYRAMID_SPECS.queensChamber;
    const ap = PYRAMID_SPECS.ascendingPassage;
    const gg = PYRAMID_SPECS.grandGallery;

    const count = this.particleCount;
    const pos = this.positions;

    const speedScale = 0.4 + Math.min(reactionRate / 10.0, 2.5);

    for (let i = 0; i < count; i++) {
      let p = this.particleProgress[i];
      p += dt * this.particleSpeed[i] * speedScale * 0.15;
      if (p > 1.0) p -= 1.0;
      this.particleProgress[i] = p;

      const region = this.particleRegions[i];
      const ox = this.particleOffsets[i * 3 + 0];
      const oy = this.particleOffsets[i * 3 + 1];
      const oz = this.particleOffsets[i * 3 + 2];

      const idx = i * 3;

      if (region === 0) {
        const x = qc.centerX + ox * (qc.widthEW * 0.4);
        const y = qc.datum + p * (qc.heightApex * 0.9) + 0.4;
        const z = qc.centerNS + oz * (qc.lengthNS * 0.4);
        pos[idx + 0] = x;
        pos[idx + 1] = y;
        pos[idx + 2] = z;
      } else if (region === 1) {
        const x = ox * 0.35;
        const y = qc.datum + 0.6 + oy * 0.35;
        const z = qc.centerNS + p * (-2.88 - qc.centerNS);
        pos[idx + 0] = x;
        pos[idx + 1] = y;
        pos[idx + 2] = z;
      } else if (region === 2) {
        const startX = ap.start.x;
        const startY = ap.start.elev;
        const startZ = ap.start.ns;
        const endX = ap.end.x;
        const endY = ap.end.elev;
        const endZ = ap.end.ns;

        pos[idx + 0] = THREE.MathUtils.lerp(startX, endX, p) + ox * 0.4;
        pos[idx + 1] = THREE.MathUtils.lerp(startY, endY, p) + 0.6 + oy * 0.3;
        pos[idx + 2] = THREE.MathUtils.lerp(startZ, endZ, p) + oz * 0.3;
      } else if (region === 3) {
        const ggRad = THREE.MathUtils.degToRad(gg.slopeAngleDeg);
        const startY = gg.start.elev;
        const startZ = gg.start.ns;

        const distAlong = p * gg.inclineLength;
        const y = startY + Math.sin(ggRad) * distAlong + 1.2 + oy * 1.5;
        const z = startZ + Math.cos(ggRad) * distAlong + oz * 0.5;
        const x = ox * (gg.centralTrenchWidth * 0.6);

        pos[idx + 0] = x;
        pos[idx + 1] = y;
        pos[idx + 2] = z;
      } else {
        const shaftAngle = THREE.MathUtils.degToRad(qc.shaftSouth.angleDeg);
        const sDist = p * qc.shaftSouth.length * 0.6;
        pos[idx + 0] = ox * 0.1;
        pos[idx + 1] = 22.0 + Math.sin(shaftAngle) * sDist;
        pos[idx + 2] = 3.115 + Math.cos(shaftAngle) * sDist;
      }
    }

    this.pointsGeo.attributes.position.needsUpdate = true;
  }

  public update(frame: any, deltaTime: number): void {
    const time = frame?.time ?? 0.0;
    const h2Qc = frame?.h2_mole_fraction_qc ?? 0.0;
    const h2Kc = frame?.h2_mole_fraction_kc ?? 0.0;
    const rxnRate = frame?.chemical_reaction_rate ?? 0.0;
    const heatW = frame?.qc_heat_release_w ?? 0.0;

    const maxH2 = Math.max(h2Qc, h2Kc * 100.0);
    let globalAlpha = 0.0;
    if (maxH2 > 1e-6) {
      globalAlpha = Math.min(0.2 + (maxH2 / 0.015) * 0.75, 0.95);
    } else if (rxnRate > 0.1) {
      globalAlpha = Math.min(rxnRate / 20.0, 0.6);
    }

    this.pointsMat.uniforms.uTime.value = time;
    this.pointsMat.uniforms.uReactionRate.value = rxnRate;
    this.pointsMat.uniforms.uGlobalAlpha.value = globalAlpha;

    this.evaluateParticlePositions(deltaTime, rxnRate);

    if (globalAlpha > 0.01) {
      this.qcMistMat.opacity = globalAlpha * 0.35;
      const heatIntensity = Math.min(heatW / 3.0e6, 1.0);
      this.qcMistMat.color.setHSL(0.42 - heatIntensity * 0.1, 0.95, 0.5 + heatIntensity * 0.2);
    } else {
      this.qcMistMat.opacity = 0.0;
    }
  }

  public setCutawayPlanes(planes: THREE.Plane[]): void {
    this.pointsMat.clippingPlanes = planes;
    this.qcMistMat.clippingPlanes = planes;
  }

  public dispose(): void {
    this.pointsGeo.dispose();
    this.pointsMat.dispose();
    this.qcMistMesh.geometry.dispose();
    this.qcMistMat.dispose();
  }
}

export class PiezoElectricVisualizer {
  public group: THREE.Group;
  private coronaUniforms: Record<string, THREE.IUniform>;
  private beamHaloMeshes: THREE.Mesh[] = [];
  private cofferSparkGroup: THREE.Group;
  private sparkLight: THREE.PointLight;
  private sparkArcs: THREE.LineSegments[] = [];
  private sparkSphereMesh: THREE.Mesh;
  private sparkSphereMat: THREE.MeshBasicMaterial;
  private sparkFlashDecay: number = 0.0;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'PiezoElectricGroup';

    const kc = PYRAMID_SPECS.kingsChamber;
    const tiers = kc.relievingTiers;
    const tierElevations = [48.87, 50.8, 52.7, 54.6, 56.5];
    const beamsPerTier = [9, 9, 9, 9, 7];

    this.coronaUniforms = {
      uTime: { value: 0.0 },
      uVoltage: { value: 0.0 },
      uStress: { value: 0.0 },
      uColorBase: { value: new THREE.Color('#7928ca') },
      uColorArc: { value: new THREE.Color('#38bdf8') },
    };

    const coronaMat = new THREE.ShaderMaterial({
      vertexShader: PIEZO_CORONA_VERTEX,
      fragmentShader: PIEZO_CORONA_FRAGMENT,
      uniforms: this.coronaUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    tierElevations.forEach((elev, tIdx) => {
      const numBeams = beamsPerTier[tIdx];
      const beamSpan = tiers.beamSpan * 1.08;
      const beamD = tiers.beamDepth * 1.15;
      const spacingEW = kc.widthEW / numBeams;

      for (let b = 0; b < numBeams; b++) {
        const bX = -kc.widthEW / 2.0 + (b + 0.5) * spacingEW;
        const bY = elev + beamD / 2.0;
        const bZ = kc.centerNS;

        const haloGeo = new THREE.BoxGeometry(spacingEW * 0.98, beamD, beamSpan);
        const haloMesh = new THREE.Mesh(haloGeo, coronaMat);
        haloMesh.position.set(bX, bY, bZ);
        haloMesh.name = `PiezoHalo_Tier${tIdx + 1}_Beam${b + 1}`;
        this.group.add(haloMesh);
        this.beamHaloMeshes.push(haloMesh);
      }
    });

    this.cofferSparkGroup = new THREE.Group();
    const cof = kc.coffer;
    const cofferCenter = new THREE.Vector3(cof.posX, cof.posElev + cof.extHeight / 2.0, cof.posNS);
    this.cofferSparkGroup.position.copy(cofferCenter);

    this.sparkLight = new THREE.PointLight(0xf0abfc, 0.0, 18.0, 2.0);
    this.cofferSparkGroup.add(this.sparkLight);

    const sphereGeo = new THREE.SphereGeometry(0.8, 16, 16);
    this.sparkSphereMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
    });
    this.sparkSphereMesh = new THREE.Mesh(sphereGeo, this.sparkSphereMat);
    this.cofferSparkGroup.add(this.sparkSphereMesh);

    for (let a = 0; a < 4; a++) {
      const arcPoints: THREE.Vector3[] = [];
      const segs = 10;
      for (let s = 0; s <= segs; s++) {
        arcPoints.push(new THREE.Vector3(0, 0, 0));
      }
      const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPoints);
      const arcMat = new THREE.LineBasicMaterial({
        color: 0xd946ef,
        transparent: true,
        opacity: 0.0,
        blending: THREE.AdditiveBlending,
        linewidth: 2,
      });
      const arcLine = new THREE.LineSegments(arcGeo, arcMat);
      this.cofferSparkGroup.add(arcLine);
      this.sparkArcs.push(arcLine);
    }

    this.group.add(this.cofferSparkGroup);
  }

  private triggerSparkArcFlash(): void {
    this.sparkFlashDecay = 1.0;

    const cof = PYRAMID_SPECS.kingsChamber.coffer;
    for (let a = 0; a < this.sparkArcs.length; a++) {
      const line = this.sparkArcs[a];
      const posAttr = line.geometry.attributes.position as THREE.BufferAttribute;
      const segs = posAttr.count;

      const start = new THREE.Vector3((Math.random() - 0.5) * cof.extLength, 0, (Math.random() - 0.5) * cof.extWidth);
      const target = new THREE.Vector3(
        (Math.random() - 0.5) * cof.extLength * 1.5,
        1.2 + Math.random() * 1.8,
        (Math.random() - 0.5) * cof.extWidth * 1.5
      );

      for (let s = 0; s < segs; s++) {
        const frac = s / (segs - 1);
        const cur = new THREE.Vector3().lerpVectors(start, target, frac);
        if (s > 0 && s < segs - 1) {
          cur.x += (Math.random() - 0.5) * 0.35;
          cur.y += (Math.random() - 0.5) * 0.35;
          cur.z += (Math.random() - 0.5) * 0.35;
        }
        posAttr.setXYZ(s, cur.x, cur.y, cur.z);
      }
      posAttr.needsUpdate = true;
    }
  }

  public update(frame: any, deltaTime: number): void {
    const time = frame?.time ?? 0.0;
    const voltage = frame?.total_piezo_voltage ?? 0.0;
    const stress = frame?.max_beam_stress_pa ?? 0.0;
    const sparkTriggered = frame?.spark_triggered ?? false;
    const ionDensity = frame?.ion_density ?? 0.0;

    this.coronaUniforms.uTime.value = time;
    this.coronaUniforms.uVoltage.value = voltage;
    this.coronaUniforms.uStress.value = stress;

    if (sparkTriggered) {
      this.triggerSparkArcFlash();
    }

    if (this.sparkFlashDecay > 0.001) {
      this.sparkFlashDecay = Math.max(0.0, this.sparkFlashDecay - deltaTime * 4.5);
      const f = this.sparkFlashDecay;

      this.sparkLight.intensity = f * 15.0;
      this.sparkSphereMat.opacity = f * 0.85;
      const sphereScale = 1.0 + (1.0 - f) * 2.5;
      this.sparkSphereMesh.scale.set(sphereScale, sphereScale, sphereScale);

      for (let a = 0; a < this.sparkArcs.length; a++) {
        const arcMat = this.sparkArcs[a].material as THREE.LineBasicMaterial;
        arcMat.opacity = f * 0.9;
      }
    } else {
      this.sparkLight.intensity = 0.0;
      this.sparkSphereMat.opacity = 0.0;
      for (let a = 0; a < this.sparkArcs.length; a++) {
        (this.sparkArcs[a].material as THREE.LineBasicMaterial).opacity = 0.0;
      }
    }

    if (ionDensity > 1e12 && this.sparkFlashDecay <= 0.001) {
      const ionGlow = Math.min(ionDensity / 1e16, 1.0) * 0.3;
      this.sparkSphereMat.opacity = ionGlow;
    }
  }

  public setCutawayPlanes(_planes: THREE.Plane[]): void {}

  public dispose(): void {
    this.beamHaloMeshes.forEach((m) => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
    this.sparkSphereMesh.geometry.dispose();
    this.sparkSphereMat.dispose();
    this.sparkArcs.forEach((a) => {
      a.geometry.dispose();
      (a.material as THREE.Material).dispose();
    });
  }
}

export class MicrowaveBeamVisualizer {
  public group: THREE.Group;
  private northUniforms: Record<string, THREE.IUniform>;
  private southUniforms: Record<string, THREE.IUniform>;
  private northBeamMesh: THREE.Mesh;
  private southBeamMesh: THREE.Mesh;
  private northCoreLine: THREE.Line;
  private southCoreLine: THREE.Line;
  private northApertureLight: THREE.PointLight;
  private southApertureLight: THREE.PointLight;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'MicrowaveBeamGroup';

    const kc = PYRAMID_SPECS.kingsChamber;
    const beamLength = 260.0;

    this.northUniforms = {
      uTime: { value: 0.0 },
      uBeamPower: { value: 0.0 },
      uBeamColorCore: { value: new THREE.Color('#00f5ff') },
      uBeamColorSheath: { value: new THREE.Color('#0066ff') },
    };

    const northBeamMat = new THREE.ShaderMaterial({
      vertexShader: MICROWAVE_BEAM_VERTEX,
      fragmentShader: MICROWAVE_BEAM_FRAGMENT,
      uniforms: this.northUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    const beamRadiusStart = 0.22;
    const beamRadiusEnd = 2.4;
    const beamGeo = new THREE.CylinderGeometry(beamRadiusEnd, beamRadiusStart, beamLength, 32, 64, true);
    beamGeo.rotateX(Math.PI / 2);
    beamGeo.translate(0, 0, beamLength / 2.0);

    this.northBeamMesh = new THREE.Mesh(beamGeo, northBeamMat);

    const ksn = kc.shaftNorth;
    const ksnStart = engineToThree(ksn.start.x, ksn.start.ns, ksn.start.elev);
    const radN = THREE.MathUtils.degToRad(ksn.angleDeg);
    const ksnDir = new THREE.Vector3(0, Math.sin(radN), -Math.cos(radN)).normalize();

    this.northBeamMesh.position.copy(ksnStart);
    this.northBeamMesh.lookAt(ksnStart.clone().add(ksnDir));
    this.group.add(this.northBeamMesh);

    const coreLineGeoN = new THREE.BufferGeometry().setFromPoints([
      ksnStart.clone(),
      ksnStart.clone().add(ksnDir.clone().multiplyScalar(beamLength)),
    ]);
    const coreLineMatN = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      linewidth: 2,
    });
    this.northCoreLine = new THREE.Line(coreLineGeoN, coreLineMatN);
    this.group.add(this.northCoreLine);

    this.northApertureLight = new THREE.PointLight(0x00f5ff, 0.0, 35.0, 1.8);
    this.northApertureLight.position.copy(ksnStart);
    this.group.add(this.northApertureLight);

    this.southUniforms = {
      uTime: { value: 0.0 },
      uBeamPower: { value: 0.0 },
      uBeamColorCore: { value: new THREE.Color('#00f5ff') },
      uBeamColorSheath: { value: new THREE.Color('#0066ff') },
    };

    const southBeamMat = new THREE.ShaderMaterial({
      vertexShader: MICROWAVE_BEAM_VERTEX,
      fragmentShader: MICROWAVE_BEAM_FRAGMENT,
      uniforms: this.southUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    this.southBeamMesh = new THREE.Mesh(beamGeo.clone(), southBeamMat);

    const kss = kc.shaftSouth;
    const kssStart = engineToThree(kss.start.x, kss.start.ns, kss.start.elev);
    const radS = THREE.MathUtils.degToRad(kss.angleDeg);
    const kssDir = new THREE.Vector3(0, Math.sin(radS), Math.cos(radS)).normalize();

    this.southBeamMesh.position.copy(kssStart);
    this.southBeamMesh.lookAt(kssStart.clone().add(kssDir));
    this.group.add(this.southBeamMesh);

    const coreLineGeoS = new THREE.BufferGeometry().setFromPoints([
      kssStart.clone(),
      kssStart.clone().add(kssDir.clone().multiplyScalar(beamLength)),
    ]);
    const coreLineMatS = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.0,
      blending: THREE.AdditiveBlending,
      linewidth: 2,
    });
    this.southCoreLine = new THREE.Line(coreLineGeoS, coreLineMatS);
    this.group.add(this.southCoreLine);

    this.southApertureLight = new THREE.PointLight(0x00f5ff, 0.0, 35.0, 1.8);
    this.southApertureLight.position.copy(kssStart);
    this.group.add(this.southApertureLight);
  }

  public update(frame: any, _deltaTime: number): void {
    const time = frame?.time ?? 0.0;
    const totalPower = frame?.maser_total_radiated_power ?? 0.0;
    const northPower = frame?.maser_north_beam_power ?? frame?.spatial?.north_shaft_power ?? totalPower * 0.5;
    const southPower = frame?.maser_south_beam_power ?? frame?.spatial?.south_shaft_power ?? totalPower * 0.5;

    this.northUniforms.uTime.value = time;
    this.northUniforms.uBeamPower.value = northPower;

    this.southUniforms.uTime.value = time;
    this.southUniforms.uBeamPower.value = southPower;

    const normN = northPower > 1e-15 ? Math.min(1.0, (log10Safe(northPower) + 20) / 20) : 0.0;
    const normS = southPower > 1e-15 ? Math.min(1.0, (log10Safe(southPower) + 20) / 20) : 0.0;

    (this.northCoreLine.material as THREE.LineBasicMaterial).opacity = normN * 0.9;
    (this.southCoreLine.material as THREE.LineBasicMaterial).opacity = normS * 0.9;

    this.northApertureLight.intensity = normN * 4.0;
    this.southApertureLight.intensity = normS * 4.0;
  }

  public setCutawayPlanes(_planes: THREE.Plane[]): void {}

  public dispose(): void {
    this.northBeamMesh.geometry.dispose();
    (this.northBeamMesh.material as THREE.Material).dispose();
    this.southBeamMesh.geometry.dispose();
    (this.southBeamMesh.material as THREE.Material).dispose();
    this.northCoreLine.geometry.dispose();
    (this.northCoreLine.material as THREE.Material).dispose();
    this.southCoreLine.geometry.dispose();
    (this.southCoreLine.material as THREE.Material).dispose();
  }
}

function log10Safe(val: number): number {
  return Math.log(Math.max(val, 1e-30)) / Math.LN10;
}

export class HydraulicShockwaveVisualizer {
  public group: THREE.Group;
  private shockwaveUniforms: Record<string, THREE.IUniform>;
  private ringMeshes: THREE.Mesh[] = [];
  private ringCount: number = 6;
  private ringRadii: Float32Array;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'HydraulicShockwaveGroup';

    const sub = PYRAMID_SPECS.subterranean;
    const subCenter = engineToThree(sub.centerX, sub.centerNS, sub.datum);

    this.ringRadii = new Float32Array(this.ringCount);
    for (let r = 0; r < this.ringCount; r++) {
      this.ringRadii[r] = (r / this.ringCount) * 80.0;
    }

    this.shockwaveUniforms = {
      uTime: { value: 0.0 },
      uPressure: { value: 0.0 },
      uDisplacement: { value: 0.0 },
      uWaveColor: { value: new THREE.Color('#00f2fe') },
    };

    const shockMat = new THREE.ShaderMaterial({
      vertexShader: HYDRAULIC_SHOCKWAVE_VERTEX,
      fragmentShader: HYDRAULIC_SHOCKWAVE_FRAGMENT,
      uniforms: this.shockwaveUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    const discGeo = new THREE.PlaneGeometry(100.0, 100.0, 32, 32);
    discGeo.rotateX(-Math.PI / 2);

    for (let r = 0; r < this.ringCount; r++) {
      const ring = new THREE.Mesh(discGeo, shockMat);
      ring.position.copy(subCenter);
      ring.name = `HydraulicShockwaveRing_${r + 1}`;
      this.group.add(ring);
      this.ringMeshes.push(ring);
    }
  }

  public update(frame: any, deltaTime: number): void {
    const time = frame?.time ?? 0.0;
    const pSub = frame?.acoustic_pressure_sub ?? frame?.water_hammer_pressure ?? 0.0;
    const bedrockDisp = frame?.bedrock_displacement ?? 0.0;

    this.shockwaveUniforms.uTime.value = time;
    this.shockwaveUniforms.uPressure.value = pSub;
    this.shockwaveUniforms.uDisplacement.value = bedrockDisp;

    const sub = PYRAMID_SPECS.subterranean;
    const subY = sub.datum;
    const maxRadius = 110.0;
    const expansionSpeed = 35.0;

    const normPressure = Math.min(pSub / 2.0e6, 1.0);

    for (let r = 0; r < this.ringCount; r++) {
      let radius = this.ringRadii[r];
      radius += deltaTime * expansionSpeed * (0.8 + normPressure * 0.6);
      if (radius > maxRadius) {
        radius -= maxRadius;
      }
      this.ringRadii[r] = radius;

      const ring = this.ringMeshes[r];
      const scale = radius / 50.0;
      ring.scale.set(scale, scale, scale);

      const elevProgress = radius / maxRadius;
      const curY = THREE.MathUtils.lerp(subY, 15.0, elevProgress);
      ring.position.y = curY;
    }
  }

  public setCutawayPlanes(planes: THREE.Plane[]): void {
    (this.ringMeshes[0].material as THREE.ShaderMaterial).clippingPlanes = planes;
  }

  public dispose(): void {
    this.ringMeshes.forEach((m) => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
  }
}

export class FieldRenderers {
  public group: THREE.Group;

  public acousticRenderer: AcousticStandingWaveRenderer;
  public hydrogenRenderer: HydrogenGasVisualizer;
  public piezoRenderer: PiezoElectricVisualizer;
  public microwaveRenderer: MicrowaveBeamVisualizer;
  public hydraulicRenderer: HydraulicShockwaveVisualizer;

  private isAcousticVisible: boolean = true;
  private isHydrogenVisible: boolean = true;
  private isPiezoVisible: boolean = true;
  private isMicrowaveVisible: boolean = true;
  private isHydraulicVisible: boolean = true;

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'MasterFieldRenderersGroup';

    this.acousticRenderer = new AcousticStandingWaveRenderer();
    this.hydrogenRenderer = new HydrogenGasVisualizer();
    this.piezoRenderer = new PiezoElectricVisualizer();
    this.microwaveRenderer = new MicrowaveBeamVisualizer();
    this.hydraulicRenderer = new HydraulicShockwaveVisualizer();

    this.group.add(
      this.acousticRenderer.group,
      this.hydrogenRenderer.group,
      this.piezoRenderer.group,
      this.microwaveRenderer.group,
      this.hydraulicRenderer.group
    );
  }

  public update(frame: any, deltaTime: number): void {
    if (this.isAcousticVisible) {
      this.acousticRenderer.update(frame, deltaTime);
    }
    if (this.isHydrogenVisible) {
      this.hydrogenRenderer.update(frame, deltaTime);
    }
    if (this.isPiezoVisible) {
      this.piezoRenderer.update(frame, deltaTime);
    }
    if (this.isMicrowaveVisible) {
      this.microwaveRenderer.update(frame, deltaTime);
    }
    if (this.isHydraulicVisible) {
      this.hydraulicRenderer.update(frame, deltaTime);
    }
  }

  public setCutawayPlanes(planes: THREE.Plane[]): void {
    this.acousticRenderer.setCutawayPlanes(planes);
    this.hydrogenRenderer.setCutawayPlanes(planes);
    this.piezoRenderer.setCutawayPlanes(planes);
    this.microwaveRenderer.setCutawayPlanes(planes);
    this.hydraulicRenderer.setCutawayPlanes(planes);
  }

  public setAcousticVisible(visible: boolean): void {
    this.isAcousticVisible = visible;
    this.acousticRenderer.group.visible = visible;
  }

  public setHydrogenVisible(visible: boolean): void {
    this.isHydrogenVisible = visible;
    this.hydrogenRenderer.group.visible = visible;
  }

  public setPiezoVisible(visible: boolean): void {
    this.isPiezoVisible = visible;
    this.piezoRenderer.group.visible = visible;
  }

  public setMicrowaveVisible(visible: boolean): void {
    this.isMicrowaveVisible = visible;
    this.microwaveRenderer.group.visible = visible;
  }

  public setHydraulicVisible(visible: boolean): void {
    this.isHydraulicVisible = visible;
    this.hydraulicRenderer.group.visible = visible;
  }

  public dispose(): void {
    this.acousticRenderer.dispose();
    this.hydrogenRenderer.dispose();
    this.piezoRenderer.dispose();
    this.microwaveRenderer.dispose();
    this.hydraulicRenderer.dispose();
  }
}
