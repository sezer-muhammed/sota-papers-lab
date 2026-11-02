/* The 3D viewer: three.js + Spark (sparkjs.dev, MIT) on one canvas, orbit
   controls, camera frustums. Loaded with a dynamic import by viewer.tsx, so
   the rest of the shell does not pay for three.js.

   Scenes arrive in COLMAP world axes (y down, z forward); one group turned
   half a turn about x shows them y-up, splats and frustums alike. */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { frustumScale, type Frustum } from "./cameras";

export type SplatFormat = "ply" | "spz";

/** An HSL token triplet ("212 100% 48%") as a three.js colour. */
function tokenColor(name: string, fallback: THREE.Color): THREE.Color {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(`--${name}-value`).trim();
  const m = raw.match(/^([\d.]+)(?:deg)?[ ,]+([\d.]+)%[ ,]+([\d.]+)%/);
  if (!m) return fallback;
  return new THREE.Color().setHSL(Number(m[1]) / 360, Number(m[2]) / 100, Number(m[3]) / 100, THREE.SRGBColorSpace);
}

export class SplatEngine {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(55, 1, 0.01, 2000);
  private readonly controls: OrbitControls;
  private readonly world = new THREE.Group();
  private readonly resize: ResizeObserver;
  private readonly spark: SparkRenderer;
  private mesh?: SplatMesh;
  private frustums?: THREE.LineSegments;
  private cameras: Frustum[] = [];
  private framed = false;
  /** Frames drawn, for tests and the stats line. */
  frames = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);
    this.world.rotation.x = Math.PI;
    this.scene.add(this.world);
    this.camera.position.set(0, 0, 3);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.applyTheme();
    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(canvas);
    this.fit();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.frames += 1;
    });
  }

  /** Background from the page's sunken surface token; call on theme change. */
  applyTheme() {
    this.renderer.setClearColor(tokenColor("ds-background-200", new THREE.Color(0.05, 0.05, 0.05)));
    if (this.frustums) (this.frustums.material as THREE.LineBasicMaterial).color = this.frustumColor();
  }

  private frustumColor() {
    return tokenColor("ds-blue-700", new THREE.Color(0.1, 0.45, 1));
  }

  private fit() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Swaps in a splat file. Keeps the view once one has been framed (live
      snapshots of the same run), unless `reframe`. Resolves with the count. */
  async load(bytes: ArrayBuffer, format: SplatFormat, reframe = false): Promise<number> {
    const mesh = new SplatMesh({
      fileBytes: new Uint8Array(bytes),
      fileType: format === "spz" ? SplatFileType.SPZ : SplatFileType.PLY,
    });
    await mesh.initialized;
    if (this.mesh) {
      this.world.remove(this.mesh);
      this.mesh.dispose();
    }
    this.mesh = mesh;
    this.world.add(mesh);
    if (reframe || !this.framed) this.frame();
    return mesh.packedSplats?.numSplats ?? 0;
  }

  clear() {
    if (this.mesh) {
      this.world.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = undefined;
      // Spark keeps drawing its last sorted splats until told otherwise.
      this.spark.clearSplats();
    }
    this.setCameras([]);
    this.framed = false;
  }

  setCameras(cameras: Frustum[]) {
    this.cameras = cameras;
    if (this.frustums) {
      this.world.remove(this.frustums);
      this.frustums.geometry.dispose();
      (this.frustums.material as THREE.Material).dispose();
      this.frustums = undefined;
    }
    if (!cameras.length) return;
    const s = frustumScale(cameras);
    const pts: number[] = [];
    for (const { center: c, corners } of cameras) {
      const q = corners.map((p) => [c[0] + (p[0] - c[0]) * s, c[1] + (p[1] - c[1]) * s, c[2] + (p[2] - c[2]) * s]);
      q.forEach((corner, i) => pts.push(...c, ...corner, ...corner, ...q[(i + 1) % 4]!));
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    this.frustums = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: this.frustumColor() }));
    this.world.add(this.frustums);
    if (!this.framed && !this.mesh) this.frame();
  }

  setFrustumsVisible(visible: boolean) {
    if (this.frustums) this.frustums.visible = visible;
  }

  /** Orbit around the splats' centre; look from the first camera when there
      is one, else from a distance that fits the centres' box. */
  frame() {
    this.world.updateMatrixWorld(true);
    const box = new THREE.Box3();
    if (this.mesh) box.copy(this.mesh.getBoundingBox(true)).applyMatrix4(this.world.matrixWorld);
    for (const cam of this.cameras) box.expandByPoint(new THREE.Vector3(...cam.center).applyMatrix4(this.world.matrixWorld));
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
    const first = this.cameras[0];
    if (first) {
      this.camera.position.copy(new THREE.Vector3(...first.center).applyMatrix4(this.world.matrixWorld));
    } else {
      this.camera.position.copy(center).add(new THREE.Vector3(0, radius * 0.3, radius * 1.2));
    }
    this.camera.near = radius / 1000;
    this.camera.far = radius * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
    this.framed = true;
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this.resize.disconnect();
    this.controls.dispose();
    this.clear();
    this.renderer.dispose();
  }
}
