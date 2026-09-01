import * as THREE from 'three';

export const PYRAMID_SPECS = {
  baseSide: 230.364,
  height: 146.58,
  slopeAngleDeg: 51.84444444444445,
  royalCubit: 0.5236,
  baseElevation: 0.0,

  subterranean: {
    datum: -30.0,
    widthEW: 14.07,
    lengthNS: 8.35,
    height: 3.52,
    centerX: 0.0,
    centerNS: -27.4,
    centerElev: -30.0,
    pitDepth: 3.2,
    pitWidth: 2.0,
    pitLength: 2.0,
    blindPassageLength: 16.38,
    blindPassageWidth: 0.74,
    blindPassageHeight: 0.74,
  },

  descendingPassage: {
    start: { x: 0.0, elev: 17.0, ns: -56.5 },
    end: { x: 0.0, elev: -30.0, ns: -27.4 },
    length: 105.23,
    inclineAngleDeg: 26.523055555555554,
    width: 1.05,
    height: 1.2,
  },
  ascendingPassage: {
    start: { x: 0.0, elev: 0.0, ns: -38.2 },
    end: { x: 0.0, elev: 21.2, ns: -2.88 },
    length: 39.28,
    inclineAngleDeg: 26.041666666666668,
    width: 1.05,
    height: 1.2,
  },

  queensChamber: {
    datum: 21.2,
    widthEW: 5.75,
    lengthNS: 5.23,
    heightApex: 6.23,
    wallHeight: 4.69,
    centerX: 0.0,
    centerNS: 0.5,
    niche: {
      height: 4.67,
      baseWidth: 1.57,
      depth: 1.04,
      tiers: 5,
    },
    shaftNorth: {
      start: { x: 0.0, elev: 22.0, ns: -2.115 },
      angleDeg: 39.11666666666667,
      length: 65.0,
      width: 0.21,
      height: 0.21,
      heading: 'north' as const,
    },
    shaftSouth: {
      start: { x: 0.0, elev: 22.0, ns: 3.115 },
      angleDeg: 39.60777777777778,
      length: 63.6,
      width: 0.21,
      height: 0.21,
      heading: 'south' as const,
    },
  },

  grandGallery: {
    start: { x: 0.0, elev: 21.2, ns: -2.88 },
    inclineLength: 46.61,
    slopeAngleDeg: 26.041666666666668,
    verticalHeight: 8.6,
    widthBase: 2.09,
    widthRoof: 1.05,
    corbelSteps: 7,
    centralTrenchWidth: 1.05,
    sideRampsWidth: 0.52,
    slotPairs: 28,
    slotSpacing: 1.68,
    slotLength: 0.54,
    slotWidth: 0.16,
    slotDepth: 0.28,
  },

  antechamber: {
    datum: 43.03,
    widthEW: 1.75,
    lengthNS: 2.95,
    height: 3.81,
    centerX: 0.0,
    centerNS: 12.5,
    graniteLeafThickness: 0.41,
    groovesCount: 4,
  },

  kingsChamber: {
    datum: 43.03,
    widthEW: 10.47,
    lengthNS: 5.235,
    height: 5.84,
    centerX: 0.0,
    centerNS: 15.0,
    coffer: {
      extLength: 2.278,
      extWidth: 0.977,
      extHeight: 1.048,
      intLength: 1.977,
      intWidth: 0.677,
      intHeight: 0.872,
      posX: -3.4,
      posNS: 15.0,
      posElev: 43.03,
    },
    relievingTiers: {
      count: 5,
      names: [
        "Davison's Chamber",
        "Wellington's Chamber",
        "Nelson's Chamber",
        "Lady Arbuthnot's Chamber",
        "Campbell's Chamber",
      ],
      totalBeams: 43,
      beamSpan: 6.5,
      beamWidth: 1.2,
      beamDepth: 1.5,
    },
    shaftNorth: {
      start: { x: 0.0, elev: 44.0, ns: 12.38 },
      angleDeg: 32.46666666666667,
      length: 71.0,
      width: 0.22,
      height: 0.22,
      heading: 'north' as const,
    },
    shaftSouth: {
      start: { x: 0.0, elev: 44.0, ns: 17.62 },
      angleDeg: 45.0,
      length: 53.0,
      width: 0.22,
      height: 0.22,
      heading: 'south' as const,
    },
  },
};

export function engineToThree(x: number, northSouth: number, elevation: number): THREE.Vector3 {
  return new THREE.Vector3(x, elevation, northSouth);
}

export type CutawayMode = 'none' | 'east' | 'south' | 'quadrant' | 'custom';

export class PyramidMesh {
  public group: THREE.Group;
  public envelopeGroup: THREE.Group;
  public subterraneanGroup: THREE.Group;
  public passagesGroup: THREE.Group;
  public queensGroup: THREE.Group;
  public grandGalleryGroup: THREE.Group;
  public antechamberGroup: THREE.Group;
  public kingsGroup: THREE.Group;
  public groundGroup: THREE.Group;
  public lightsGroup: THREE.Group;

  private outerCasingMaterial: THREE.MeshPhysicalMaterial;
  private outerCasingWireframeMaterial: THREE.LineBasicMaterial;
  private outerCasingMesh!: THREE.Mesh;
  private outerCasingWireframe!: THREE.LineSegments;
  private limestoneMaterial: THREE.MeshStandardMaterial;
  private limestoneDarkMaterial: THREE.MeshStandardMaterial;
  private roseGraniteMaterial: THREE.MeshStandardMaterial;
  private polishedGraniteMaterial: THREE.MeshStandardMaterial;
  private shaftMaterial: THREE.MeshStandardMaterial;
  private gantenbrinkDoorMaterial: THREE.MeshStandardMaterial;

  private cutawayPlaneEast: THREE.Plane;
  private cutawayPlaneSouth: THREE.Plane;
  private activeClippingPlanes: THREE.Plane[] = [];

  public relievingBeams: THREE.Mesh[] = [];
  public chamberCenters: Map<string, THREE.Vector3> = new Map();

  constructor() {
    this.group = new THREE.Group();
    this.group.name = 'KhufuPyramidHierarchy';

    this.envelopeGroup = new THREE.Group();
    this.envelopeGroup.name = 'EnvelopeGroup';

    this.subterraneanGroup = new THREE.Group();
    this.subterraneanGroup.name = 'SubterraneanGroup';

    this.passagesGroup = new THREE.Group();
    this.passagesGroup.name = 'PassagesGroup';

    this.queensGroup = new THREE.Group();
    this.queensGroup.name = 'QueensGroup';

    this.grandGalleryGroup = new THREE.Group();
    this.grandGalleryGroup.name = 'GrandGalleryGroup';

    this.antechamberGroup = new THREE.Group();
    this.antechamberGroup.name = 'AntechamberGroup';

    this.kingsGroup = new THREE.Group();
    this.kingsGroup.name = 'KingsGroup';

    this.groundGroup = new THREE.Group();
    this.groundGroup.name = 'GroundGroup';

    this.lightsGroup = new THREE.Group();
    this.lightsGroup.name = 'ChamberLightsGroup';

    this.group.add(
      this.groundGroup,
      this.envelopeGroup,
      this.subterraneanGroup,
      this.passagesGroup,
      this.queensGroup,
      this.grandGalleryGroup,
      this.antechamberGroup,
      this.kingsGroup,
      this.lightsGroup
    );

    this.cutawayPlaneEast = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
    this.cutawayPlaneSouth = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);

    this.outerCasingMaterial = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color('#d9cbb0'),
      transparent: true,
      opacity: 0.18,
      roughness: 0.35,
      metalness: 0.05,
      transmission: 0.82,
      ior: 1.52,
      thickness: 1.5,
      side: THREE.DoubleSide,
      depthWrite: false,
      clippingPlanes: this.activeClippingPlanes,
      clipShadows: true,
    });

    this.outerCasingWireframeMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color('#e5c07b'),
      transparent: true,
      opacity: 0.45,
      linewidth: 1,
      clippingPlanes: this.activeClippingPlanes,
    });

    this.limestoneMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#cfbe9b'),
      roughness: 0.88,
      metalness: 0.02,
      side: THREE.DoubleSide,
      clippingPlanes: this.activeClippingPlanes,
    });

    this.limestoneDarkMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#8a7f6c'),
      roughness: 0.92,
      metalness: 0.02,
      side: THREE.DoubleSide,
      clippingPlanes: this.activeClippingPlanes,
    });

    this.roseGraniteMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#b3543d'),
      roughness: 0.58,
      metalness: 0.15,
      side: THREE.DoubleSide,
      clippingPlanes: this.activeClippingPlanes,
    });

    this.polishedGraniteMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#8c3826'),
      roughness: 0.32,
      metalness: 0.25,
      side: THREE.DoubleSide,
      clippingPlanes: this.activeClippingPlanes,
    });

    this.shaftMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#5098b8'),
      transparent: true,
      opacity: 0.75,
      roughness: 0.4,
      metalness: 0.3,
      side: THREE.DoubleSide,
      clippingPlanes: this.activeClippingPlanes,
    });

    this.gantenbrinkDoorMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#e0b553'),
      roughness: 0.4,
      metalness: 0.6,
      side: THREE.DoubleSide,
      clippingPlanes: this.activeClippingPlanes,
    });

    this.buildGroundAndBedrock();
    this.buildOuterEnvelope();
    this.buildSubterraneanChamber();
    this.buildPassages();
    this.buildQueensChamber();
    this.buildGrandGallery();
    this.buildAntechamber();
    this.buildKingsChamberAndBeams();
    this.setupChamberLights();
    this.registerChamberCenters();

    this.setCutawayMode('east');
  }

  private buildGroundAndBedrock(): void {
    const groundSize = 340.0;
    const groundGeo = new THREE.PlaneGeometry(groundSize, groundSize, 32, 32);
    groundGeo.rotateX(-Math.PI / 2);

    const groundMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#1e222d'),
      roughness: 0.95,
      metalness: 0.05,
      wireframe: false,
    });

    const groundMesh = new THREE.Mesh(groundGeo, groundMat);
    groundMesh.position.set(0, 0, 0);
    groundMesh.receiveShadow = true;
    this.groundGroup.add(groundMesh);

    const grid = new THREE.GridHelper(groundSize, 68, 0xd4af37, 0x2c3e50);
    grid.position.y = 0.05;
    this.groundGroup.add(grid);

    const subBedrockGeo = new THREE.BoxGeometry(40, 36, 60);
    const subBedrockEdges = new THREE.EdgesGeometry(subBedrockGeo);
    const subBedrockLine = new THREE.LineSegments(
      subBedrockEdges,
      new THREE.LineBasicMaterial({ color: 0x34495e, transparent: true, opacity: 0.3 })
    );
    subBedrockLine.position.set(0, -18, -25);
    this.groundGroup.add(subBedrockLine);
  }

  private buildOuterEnvelope(): void {
    const halfBase = PYRAMID_SPECS.baseSide / 2.0;
    const h = PYRAMID_SPECS.height;

    const vertices = new Float32Array([
      0, h, 0,
      halfBase, 0, -halfBase,
      halfBase, 0, halfBase,
      -halfBase, 0, halfBase,
      -halfBase, 0, -halfBase,
    ]);

    const indices = [
      0, 2, 1,
      0, 3, 2,
      0, 4, 3,
      0, 1, 4,
      3, 4, 1,
      3, 1, 2,
    ];

    const pyramidGeo = new THREE.BufferGeometry();
    pyramidGeo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    pyramidGeo.setIndex(indices);
    pyramidGeo.computeVertexNormals();

    this.outerCasingMesh = new THREE.Mesh(pyramidGeo, this.outerCasingMaterial);
    this.outerCasingMesh.castShadow = true;
    this.outerCasingMesh.receiveShadow = true;
    this.envelopeGroup.add(this.outerCasingMesh);

    const wireGeo = new THREE.EdgesGeometry(pyramidGeo, 20);
    this.outerCasingWireframe = new THREE.LineSegments(wireGeo, this.outerCasingWireframeMaterial);
    this.envelopeGroup.add(this.outerCasingWireframe);
  }

  private buildSubterraneanChamber(): void {
    const s = PYRAMID_SPECS.subterranean;

    const chamberGeo = new THREE.BoxGeometry(s.widthEW, s.height, s.lengthNS);
    const chamberMesh = new THREE.Mesh(chamberGeo, this.limestoneDarkMaterial);
    const chamberY = s.datum + s.height / 2.0;
    chamberMesh.position.set(s.centerX, chamberY, s.centerNS);
    this.subterraneanGroup.add(chamberMesh);

    const pitGeo = new THREE.BoxGeometry(s.pitWidth, s.pitDepth, s.pitLength);
    const pitMesh = new THREE.Mesh(pitGeo, this.limestoneDarkMaterial);
    pitMesh.position.set(s.centerX + 2.0, s.datum - s.pitDepth / 2.0, s.centerNS + 1.0);
    this.subterraneanGroup.add(pitMesh);

    const blindGeo = new THREE.BoxGeometry(s.blindPassageWidth, s.blindPassageHeight, s.blindPassageLength);
    const blindMesh = new THREE.Mesh(blindGeo, this.limestoneDarkMaterial);
    const blindZ = s.centerNS + s.lengthNS / 2.0 + s.blindPassageLength / 2.0;
    blindMesh.position.set(s.centerX, s.datum + s.blindPassageHeight / 2.0, blindZ);
    this.subterraneanGroup.add(blindMesh);
  }

  private buildPassages(): void {
    const dp = PYRAMID_SPECS.descendingPassage;
    const dpStart = engineToThree(dp.start.x, dp.start.ns, dp.start.elev);
    const dpEnd = engineToThree(dp.end.x, dp.end.ns, dp.end.elev);
    const dpMesh = this.createPassageBetween(dpStart, dpEnd, dp.width, dp.height, this.limestoneMaterial);
    this.passagesGroup.add(dpMesh);

    const ap = PYRAMID_SPECS.ascendingPassage;
    const apStart = engineToThree(ap.start.x, ap.start.ns, ap.start.elev);
    const apEnd = engineToThree(ap.end.x, ap.end.ns, ap.end.elev);
    const apMesh = this.createPassageBetween(apStart, apEnd, ap.width, ap.height, this.limestoneMaterial);
    this.passagesGroup.add(apMesh);

    const qpHStart = engineToThree(0.0, -2.88, 21.2);
    const qpHEnd = engineToThree(0.0, -2.115, 21.2);
    const qpHMesh = this.createPassageBetween(qpHStart, qpHEnd, 1.05, 1.15, this.limestoneMaterial);
    this.passagesGroup.add(qpHMesh);
  }

  private buildQueensChamber(): void {
    const qc = PYRAMID_SPECS.queensChamber;
    const baseY = qc.datum;
    const wallH = qc.wallHeight;
    const apexH = qc.heightApex;
    const halfW = qc.widthEW / 2.0;
    const halfL = qc.lengthNS / 2.0;
    const centerZ = qc.centerNS;

    const vertices = new Float32Array([
      halfW, baseY, centerZ - halfL,
      halfW, baseY, centerZ + halfL,
      -halfW, baseY, centerZ + halfL,
      -halfW, baseY, centerZ - halfL,

      halfW, baseY + wallH, centerZ - halfL,
      halfW, baseY + wallH, centerZ + halfL,
      -halfW, baseY + wallH, centerZ + halfL,
      -halfW, baseY + wallH, centerZ - halfL,

      0, baseY + apexH, centerZ - halfL,
      0, baseY + apexH, centerZ + halfL,
    ]);

    const indices = [
      2, 1, 0, 3, 2, 0,
      0, 1, 5, 0, 5, 4,
      3, 6, 2, 3, 7, 6,
      0, 4, 7, 0, 7, 3,
      4, 8, 7,
      2, 6, 5, 2, 5, 1,
      5, 6, 9,
      4, 5, 9, 4, 9, 8,
      7, 8, 9, 7, 9, 6,
    ];

    const qcGeo = new THREE.BufferGeometry();
    qcGeo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    qcGeo.setIndex(indices);
    qcGeo.computeVertexNormals();

    const qcMesh = new THREE.Mesh(qcGeo, this.limestoneMaterial);
    this.queensGroup.add(qcMesh);

    const nicheTiers = qc.niche.tiers;
    const nicheTotalH = qc.niche.height;
    const tierH = nicheTotalH / nicheTiers;
    const nicheBaseW = qc.niche.baseWidth;
    const nicheDepth = qc.niche.depth;

    for (let i = 0; i < nicheTiers; i++) {
      const tierW = nicheBaseW * (1.0 - i * 0.12);
      const tierGeo = new THREE.BoxGeometry(nicheDepth, tierH, tierW);
      const tierMesh = new THREE.Mesh(tierGeo, this.limestoneDarkMaterial);
      const tierY = baseY + i * tierH + tierH / 2.0;
      tierMesh.position.set(halfW + nicheDepth / 2.0 - 0.05, tierY, centerZ);
      this.queensGroup.add(tierMesh);
    }

    const qsn = qc.shaftNorth;
    const qsnStart = engineToThree(qsn.start.x, qsn.start.ns, qsn.start.elev);
    const radN = THREE.MathUtils.degToRad(qsn.angleDeg);
    const qsnDir = new THREE.Vector3(0, Math.sin(radN), -Math.cos(radN)).normalize();
    const qsnEnd = qsnStart.clone().add(qsnDir.clone().multiplyScalar(qsn.length));
    const qsnMesh = this.createPassageBetween(qsnStart, qsnEnd, qsn.width, qsn.height, this.shaftMaterial);
    this.queensGroup.add(qsnMesh);

    const qss = qc.shaftSouth;
    const qssStart = engineToThree(qss.start.x, qss.start.ns, qss.start.elev);
    const radS = THREE.MathUtils.degToRad(qss.angleDeg);
    const qssDir = new THREE.Vector3(0, Math.sin(radS), Math.cos(radS)).normalize();
    const qssEnd = qssStart.clone().add(qssDir.clone().multiplyScalar(qss.length));
    const qssMesh = this.createPassageBetween(qssStart, qssEnd, qss.width, qss.height, this.shaftMaterial);
    this.queensGroup.add(qssMesh);

    const doorGeo = new THREE.BoxGeometry(0.25, 0.25, 0.15);
    const doorMesh = new THREE.Mesh(doorGeo, this.gantenbrinkDoorMaterial);
    doorMesh.position.copy(qssEnd);
    doorMesh.lookAt(qssEnd.clone().add(qssDir));
    this.queensGroup.add(doorMesh);
  }

  private buildGrandGallery(): void {
    const gg = PYRAMID_SPECS.grandGallery;
    const rad = THREE.MathUtils.degToRad(gg.slopeAngleDeg);

    const start = engineToThree(gg.start.x, gg.start.ns, gg.start.elev);
    const length = gg.inclineLength;

    const galleryGroup = new THREE.Group();
    galleryGroup.position.copy(start);
    galleryGroup.rotation.x = rad;

    const trenchW = gg.centralTrenchWidth;
    const rampW = gg.sideRampsWidth;
    const totalW = gg.widthBase;

    const floorGeo = new THREE.BoxGeometry(totalW, 0.4, length);
    const floorMesh = new THREE.Mesh(floorGeo, this.limestoneMaterial);
    floorMesh.position.set(0, -0.2, length / 2.0);
    galleryGroup.add(floorMesh);

    const steps = gg.corbelSteps;
    const stepH = gg.verticalHeight / (steps + 1);
    const stepInward = (gg.widthBase - gg.widthRoof) / (2.0 * steps);

    for (let s = 0; s < steps; s++) {
      const curHalfW = totalW / 2.0 - s * stepInward;
      const curH = stepH;
      const curY = s * stepH + stepH / 2.0;

      const eastStepGeo = new THREE.BoxGeometry(stepInward + 0.02, curH, length);
      const eastStepMesh = new THREE.Mesh(eastStepGeo, this.limestoneMaterial);
      eastStepMesh.position.set(curHalfW - stepInward / 2.0, curY, length / 2.0);
      galleryGroup.add(eastStepMesh);

      const westStepGeo = new THREE.BoxGeometry(stepInward + 0.02, curH, length);
      const westStepMesh = new THREE.Mesh(westStepGeo, this.limestoneMaterial);
      westStepMesh.position.set(-curHalfW + stepInward / 2.0, curY, length / 2.0);
      galleryGroup.add(westStepMesh);
    }

    const roofGeo = new THREE.BoxGeometry(gg.widthRoof, 0.4, length);
    const roofMesh = new THREE.Mesh(roofGeo, this.limestoneMaterial);
    roofMesh.position.set(0, gg.verticalHeight, length / 2.0);
    galleryGroup.add(roofMesh);

    const slotGeo = new THREE.BoxGeometry(gg.slotWidth, gg.slotDepth, gg.slotLength);
    const slotMat = this.roseGraniteMaterial;

    for (let i = 0; i < gg.slotPairs; i++) {
      const sPos = i * gg.slotSpacing + 0.8;
      if (sPos > length - 0.5) break;

      const westSlot = new THREE.Mesh(slotGeo, slotMat);
      westSlot.position.set(-trenchW / 2.0 - rampW / 2.0, 0.1, sPos);
      galleryGroup.add(westSlot);

      const eastSlot = new THREE.Mesh(slotGeo, slotMat);
      eastSlot.position.set(trenchW / 2.0 + rampW / 2.0, 0.1, sPos);
      galleryGroup.add(eastSlot);
    }

    this.grandGalleryGroup.add(galleryGroup);
  }

  private buildAntechamber(): void {
    const ac = PYRAMID_SPECS.antechamber;
    const acCenter = engineToThree(ac.centerX, ac.centerNS, ac.datum + ac.height / 2.0);

    const acGeo = new THREE.BoxGeometry(ac.widthEW, ac.height, ac.lengthNS);
    const acMesh = new THREE.Mesh(acGeo, this.roseGraniteMaterial);
    acMesh.position.copy(acCenter);
    this.antechamberGroup.add(acMesh);

    const grooveDepth = 0.12;
    const grooveW = 0.22;
    const grooveSpacing = ac.lengthNS / (ac.groovesCount + 1);

    for (let g = 1; g <= ac.groovesCount; g++) {
      const gZ = acCenter.z - ac.lengthNS / 2.0 + g * grooveSpacing;

      const wGrooveGeo = new THREE.BoxGeometry(grooveDepth, ac.height, grooveW);
      const wGroove = new THREE.Mesh(wGrooveGeo, this.polishedGraniteMaterial);
      wGroove.position.set(acCenter.x - ac.widthEW / 2.0 + grooveDepth / 2.0, acCenter.y, gZ);
      this.antechamberGroup.add(wGroove);

      const eGrooveGeo = new THREE.BoxGeometry(grooveDepth, ac.height, grooveW);
      const eGroove = new THREE.Mesh(eGrooveGeo, this.polishedGraniteMaterial);
      eGroove.position.set(acCenter.x + ac.widthEW / 2.0 - grooveDepth / 2.0, acCenter.y, gZ);
      this.antechamberGroup.add(eGroove);

      if (g === 1) {
        const leafH = 2.4;
        const leafGeo = new THREE.BoxGeometry(ac.widthEW - 0.05, leafH, ac.graniteLeafThickness);
        const leafMesh = new THREE.Mesh(leafGeo, this.polishedGraniteMaterial);
        const leafY = ac.datum + ac.height - leafH / 2.0;
        leafMesh.position.set(acCenter.x, leafY, gZ);
        this.antechamberGroup.add(leafMesh);
      }
    }
  }

  private buildKingsChamberAndBeams(): void {
    const kc = PYRAMID_SPECS.kingsChamber;
    const chamberCenter = engineToThree(kc.centerX, kc.centerNS, kc.datum + kc.height / 2.0);

    const roomGeo = new THREE.BoxGeometry(kc.widthEW, kc.height, kc.lengthNS);
    const roomMesh = new THREE.Mesh(roomGeo, this.roseGraniteMaterial);
    roomMesh.position.copy(chamberCenter);
    this.kingsGroup.add(roomMesh);

    const cof = kc.coffer;
    const cofferGroup = new THREE.Group();
    const cofferExtGeo = new THREE.BoxGeometry(cof.extLength, cof.extHeight, cof.extWidth);
    const cofferExtMesh = new THREE.Mesh(cofferExtGeo, this.polishedGraniteMaterial);
    cofferExtMesh.position.set(cof.posX, cof.posElev + cof.extHeight / 2.0, cof.posNS);
    cofferGroup.add(cofferExtMesh);

    const cofferIntGeo = new THREE.BoxGeometry(cof.intLength, cof.intHeight, cof.intWidth);
    const cofferIntMesh = new THREE.Mesh(cofferIntGeo, this.limestoneDarkMaterial);
    cofferIntMesh.position.set(cof.posX, cof.posElev + cof.extHeight - cof.intHeight / 2.0 + 0.01, cof.posNS);
    cofferGroup.add(cofferIntMesh);
    this.kingsGroup.add(cofferGroup);

    const tiers = kc.relievingTiers;
    const tierElevations = [48.87, 50.8, 52.7, 54.6, 56.5];
    const beamsPerTier = [9, 9, 9, 9, 7];

    let beamIndex = 0;
    tierElevations.forEach((elev, tIdx) => {
      const numBeams = beamsPerTier[tIdx];
      const tierGroup = new THREE.Group();
      tierGroup.name = tiers.names[tIdx];

      const beamSpan = tiers.beamSpan;
      const beamD = tiers.beamDepth;
      const spacingEW = kc.widthEW / numBeams;

      for (let b = 0; b < numBeams; b++) {
        const bX = -kc.widthEW / 2.0 + (b + 0.5) * spacingEW;
        const bY = elev + beamD / 2.0;
        const bZ = kc.centerNS;

        if (tIdx === 4 && (b === 0 || b === numBeams - 1)) {
          const gabledBeamGeo = new THREE.BoxGeometry(spacingEW * 0.92, beamD * 1.3, beamSpan);
          const gabledBeamMesh = new THREE.Mesh(gabledBeamGeo, this.roseGraniteMaterial);
          gabledBeamMesh.position.set(bX, bY + 0.4, bZ);
          gabledBeamMesh.rotation.z = b === 0 ? 0.25 : -0.25;
          gabledBeamMesh.name = `RelievingBeam_${beamIndex + 1}_${tiers.names[tIdx]}`;
          this.relievingBeams.push(gabledBeamMesh);
          tierGroup.add(gabledBeamMesh);
        } else {
          const beamGeo = new THREE.BoxGeometry(spacingEW * 0.9, beamD, beamSpan);
          const beamMesh = new THREE.Mesh(beamGeo, this.roseGraniteMaterial);
          beamMesh.position.set(bX, bY, bZ);
          beamMesh.name = `RelievingBeam_${beamIndex + 1}_${tiers.names[tIdx]}`;
          this.relievingBeams.push(beamMesh);
          tierGroup.add(beamMesh);
        }
        beamIndex++;
      }

      this.kingsGroup.add(tierGroup);
    });

    const ksn = kc.shaftNorth;
    const ksnStart = engineToThree(ksn.start.x, ksn.start.ns, ksn.start.elev);
    const radN = THREE.MathUtils.degToRad(ksn.angleDeg);
    const ksnDir = new THREE.Vector3(0, Math.sin(radN), -Math.cos(radN)).normalize();
    const ksnEnd = ksnStart.clone().add(ksnDir.clone().multiplyScalar(ksn.length));
    const ksnMesh = this.createPassageBetween(ksnStart, ksnEnd, ksn.width, ksn.height, this.shaftMaterial);
    this.kingsGroup.add(ksnMesh);

    const kss = kc.shaftSouth;
    const kssStart = engineToThree(kss.start.x, kss.start.ns, kss.start.elev);
    const radS = THREE.MathUtils.degToRad(kss.angleDeg);
    const kssDir = new THREE.Vector3(0, Math.sin(radS), Math.cos(radS)).normalize();
    const kssEnd = kssStart.clone().add(kssDir.clone().multiplyScalar(kss.length));
    const kssMesh = this.createPassageBetween(kssStart, kssEnd, kss.width, kss.height, this.shaftMaterial);
    this.kingsGroup.add(kssMesh);
  }

  private setupChamberLights(): void {
    const subLight = new THREE.PointLight(0x00f2fe, 1.5, 45, 1.2);
    subLight.position.set(0, -28.0, -27.4);
    this.lightsGroup.add(subLight);

    const qcLight = new THREE.PointLight(0x00ffaa, 2.0, 30, 1.2);
    qcLight.position.set(0, 24.5, 0.5);
    this.lightsGroup.add(qcLight);

    const ggLight = new THREE.PointLight(0xffaa44, 2.2, 55, 1.0);
    ggLight.position.set(0, 32.0, 18.0);
    this.lightsGroup.add(ggLight);

    const kcLight = new THREE.PointLight(0xd4af37, 2.5, 35, 1.2);
    kcLight.position.set(0, 46.0, 15.0);
    this.lightsGroup.add(kcLight);

    const beamLight = new THREE.PointLight(0xbf55ec, 2.0, 25, 1.5);
    beamLight.position.set(0, 53.0, 15.0);
    this.lightsGroup.add(beamLight);
  }

  private registerChamberCenters(): void {
    this.chamberCenters.set('Full Pyramid', new THREE.Vector3(0, 45.0, 0));
    this.chamberCenters.set(
      'Subterranean Chamber',
      engineToThree(
        PYRAMID_SPECS.subterranean.centerX,
        PYRAMID_SPECS.subterranean.centerNS,
        PYRAMID_SPECS.subterranean.datum + 2.0
      )
    );
    this.chamberCenters.set(
      "Queen's Chamber",
      engineToThree(
        PYRAMID_SPECS.queensChamber.centerX,
        PYRAMID_SPECS.queensChamber.centerNS,
        PYRAMID_SPECS.queensChamber.datum + 3.0
      )
    );
    this.chamberCenters.set('Grand Gallery', engineToThree(0.0, 18.0, 32.0));
    this.chamberCenters.set(
      "King's Chamber & Relieving Beams",
      engineToThree(
        PYRAMID_SPECS.kingsChamber.centerX,
        PYRAMID_SPECS.kingsChamber.centerNS,
        PYRAMID_SPECS.kingsChamber.datum + 5.0
      )
    );
    this.chamberCenters.set('Shaft Beaming', new THREE.Vector3(0, 75.0, 0));
  }

  private createPassageBetween(
    start: THREE.Vector3,
    end: THREE.Vector3,
    width: number,
    height: number,
    material: THREE.Material
  ): THREE.Mesh {
    const dir = new THREE.Vector3().subVectors(end, start);
    const length = dir.length();
    const center = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

    const geo = new THREE.BoxGeometry(width, height, length);
    const mesh = new THREE.Mesh(geo, material);
    mesh.position.copy(center);

    const up = new THREE.Vector3(0, 1, 0);
    const rotAxis = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), dir.clone().normalize()).normalize();
    const angle = new THREE.Vector3(0, 0, 1).angleTo(dir.clone().normalize());

    if (rotAxis.lengthSq() > 0.0001) {
      mesh.quaternion.setFromAxisAngle(rotAxis, angle);
    } else if (dir.z < 0) {
      mesh.quaternion.setFromAxisAngle(up, Math.PI);
    }

    return mesh;
  }

  public setCutawayMode(mode: CutawayMode, customPlanes?: THREE.Plane[]): void {
    if (customPlanes && customPlanes.length > 0) {
      this.activeClippingPlanes = customPlanes;
    } else {
      switch (mode) {
        case 'none':
          this.activeClippingPlanes = [];
          break;
        case 'east':
          this.activeClippingPlanes = [this.cutawayPlaneEast];
          break;
        case 'south':
          this.activeClippingPlanes = [this.cutawayPlaneSouth];
          break;
        case 'quadrant':
          this.activeClippingPlanes = [this.cutawayPlaneEast, this.cutawayPlaneSouth];
          break;
        default:
          this.activeClippingPlanes = [];
      }
    }

    this.outerCasingMaterial.clippingPlanes = this.activeClippingPlanes;
    this.outerCasingWireframeMaterial.clippingPlanes = this.activeClippingPlanes;
    this.outerCasingMaterial.needsUpdate = true;
    this.outerCasingWireframeMaterial.needsUpdate = true;
  }

  public getActiveClippingPlanes(): THREE.Plane[] {
    return this.activeClippingPlanes;
  }

  public setOuterCasingOpacity(opacity: number): void {
    this.outerCasingMaterial.opacity = Math.max(0.0, Math.min(1.0, opacity));
    this.outerCasingMaterial.needsUpdate = true;
  }

  public setWireframeVisibility(visible: boolean): void {
    this.outerCasingWireframe.visible = visible;
  }

  public getChamberCenter(name: string): THREE.Vector3 | undefined {
    return this.chamberCenters.get(name);
  }

  public getAllChamberCenters(): Map<string, THREE.Vector3> {
    return this.chamberCenters;
  }
}
