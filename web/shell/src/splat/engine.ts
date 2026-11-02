/* The 3D viewer: three.js + Spark (sparkjs.dev, MIT) on one canvas, orbit
   controls, camera frustums. Loaded with a dynamic import by viewer.tsx, so
   the rest of the shell does not pay for three.js.

   Scenes arrive in COLMAP world axes (y down, z forward); one group turned
   half a turn about x shows them y-up, splats and frustums alike. */
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  SparkRenderer,
  SplatEdit,
  SplatEditRgbaBlendMode,
  SplatEditSdf,
  SplatEditSdfType,
  SplatFileType,
  SplatMesh,
} from "@sparkjsdev/spark";
import { cameraView, focusPoint, frustumScale, meanUp, type Frustum, type Vec3 } from "./cameras";

/** An axis-aligned box in the viewer's axes (y up, after the scene turn). */
export type Box = { center: Vec3; half: Vec3 };

const DEFAULT_FOV = 55;
const FLY_MS = 700;

export type SplatFormat = "ply" | "spz";

/** An HSL token triplet ("212 100% 48%") as a three.js colour. */
function tokenColor(name: string, fallback: THREE.Color): THREE.Color {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(`--${name}-value`).trim();
  const m = raw.match(/^([\d.]+)(?:deg)?[ ,]+([\d.]+)%[ ,]+([\d.]+)%/);
  if (!m) return fallback;
  return new THREE.Color().setHSL(Number(m[1]) / 360, Number(m[2]) / 100, Number(m[3]) / 100, THREE.SRGBColorSpace);
}

/** SPZ stores positions in RUB (x right, y up, z back: COLMAP with y and z
    negated) and Spark keeps them so; turn such a mesh back into the
    group's COLMAP axes, so it lines up with PLY files and the cameras. */
function toColmap(mesh: SplatMesh) {
  mesh.rotation.x = Math.PI;
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
  private points?: SplatMesh;
  private showPoints = true;
  private showFrustums = true;
  private frustums?: THREE.LineSegments;
  private readonly thumbs = new THREE.Group();
  private crop?: { box: Box; outline: THREE.LineSegments };
  private fly?: {
    from: { position: THREE.Vector3; quaternion: THREE.Quaternion; fov: number };
    to: { position: THREE.Vector3; quaternion: THREE.Quaternion; fov: number; target: THREE.Vector3 };
    start: number;
  };
  /** Looking through camera `viewing`: orbit is off until the user drags. */
  private viewing?: number;
  /** Called when the view leaves a camera (the user orbits away). */
  onLeaveCamera?: () => void;
  /** Called when a drag stops the turntable. */
  onTurntableStop?: () => void;
  private cameras: Frustum[] = [];
  private framed = false;
  /** Frames drawn, for tests and the stats line. */
  frames = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.camera.fov = DEFAULT_FOV;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);
    this.world.rotation.x = Math.PI;
    this.scene.add(this.world);
    this.world.add(this.thumbs);
    this.camera.position.set(0, 0, 3);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.applyTheme();
    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(canvas);
    this.fit();
    this.controls.addEventListener("start", () => {
      this.leaveCamera();
      if (!this.controls.autoRotate) return;
      this.controls.autoRotate = false;
      this.onTurntableStop?.();
    });
    this.renderer.setAnimationLoop((time) => {
      if (this.fly) this.step(time);
      else if (this.viewing === undefined) this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.frames += 1;
    });
  }

  /** Background from the page's sunken surface token; call on theme change. */
  applyTheme() {
    this.renderer.setClearColor(tokenColor("ds-background-200", new THREE.Color(0.05, 0.05, 0.05)));
    if (this.frustums) (this.frustums.material as THREE.LineBasicMaterial).color = this.frustumColor();
    if (this.crop) (this.crop.outline.material as THREE.LineBasicMaterial).color = this.frustumColor();
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
    if (format === "spz") toColmap(mesh);
    if (this.mesh) {
      this.world.remove(this.mesh);
      this.mesh.dispose();
    }
    this.mesh = mesh;
    this.world.add(mesh);
    this.cropMesh(mesh);
    this.syncPoints();
    if (reframe || !this.framed) this.frame();
    return mesh.packedSplats?.numSplats ?? 0;
  }

  /** The posing stage's coloured point cloud (an .spz of small splats);
      shown until splats arrive, then only when asked for. */
  async loadPoints(bytes: ArrayBuffer): Promise<number> {
    const mesh = new SplatMesh({ fileBytes: new Uint8Array(bytes), fileType: SplatFileType.SPZ });
    await mesh.initialized;
    toColmap(mesh);
    this.dropPoints();
    this.points = mesh;
    this.world.add(mesh);
    this.cropMesh(mesh);
    this.syncPoints();
    if (!this.framed) this.frame();
    return mesh.packedSplats?.numSplats ?? 0;
  }

  setPointsVisible(visible: boolean) {
    this.showPoints = visible;
    this.syncPoints();
  }

  private syncPoints() {
    if (this.points) this.points.visible = this.showPoints || !this.mesh;
  }

  private dropPoints() {
    if (!this.points) return;
    this.world.remove(this.points);
    this.points.dispose();
    this.points = undefined;
  }

  clear() {
    if (this.mesh) {
      this.world.remove(this.mesh);
      this.mesh.dispose();
      this.mesh = undefined;
      // Spark keeps drawing its last sorted splats until told otherwise.
      this.spark.clearSplats();
    }
    this.dropPoints();
    this.setCameras([]);
    this.setCrop(undefined);
    this.leaveCamera();
    this.framed = false;
  }

  setCameras(cameras: Frustum[]) {
    this.cameras = cameras;
    this.clearThumbs();
    // Turn the scene so the cameras' up is the orbit's up; COLMAP's −y
    // without cameras.
    const up = meanUp(cameras);
    if (up) this.world.quaternion.setFromUnitVectors(new THREE.Vector3(...up), new THREE.Vector3(0, 1, 0));
    else this.world.rotation.set(Math.PI, 0, 0);
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
    this.syncFrustums();
    if (!this.framed && !this.mesh) this.frame();
  }

  setFrustumsVisible(visible: boolean) {
    this.showFrustums = visible;
    this.syncFrustums();
  }

  /** Hidden while looking through a camera, so its own photo plane does not
      cover the render. */
  private syncFrustums() {
    const visible = this.showFrustums && this.viewing === undefined;
    if (this.frustums) this.frustums.visible = visible;
    this.thumbs.visible = visible;
  }

  /** Shows a photo on camera `index`'s image plane. */
  setThumbnail(index: number, image: TexImageSource) {
    const cam = this.cameras[index];
    if (!cam) return;
    const s = frustumScale(this.cameras);
    const c = cam.center;
    const q = cam.corners.map((p) => [c[0] + (p[0] - c[0]) * s, c[1] + (p[1] - c[1]) * s, c[2] + (p[2] - c[2]) * s]);
    const geometry = new THREE.BufferGeometry();
    // Corners run top-left, top-right, bottom-right, bottom-left.
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(q.flat(), 3));
    // ImageBitmaps ignore flipY, so v runs down the image like its rows.
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
    geometry.setIndex([0, 3, 1, 1, 3, 2]);
    const texture = new THREE.Texture(image as THREE.Texture["image"]);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.flipY = false;
    texture.needsUpdate = true;
    const quad = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide, transparent: true, opacity: 0.85, depthWrite: false }),
    );
    quad.userData.camera = index;
    this.thumbs.add(quad);
  }

  private clearThumbs() {
    for (const child of [...this.thumbs.children]) {
      const quad = child as THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>;
      quad.geometry.dispose();
      quad.material.map?.dispose();
      quad.material.dispose();
      this.thumbs.remove(quad);
    }
  }

  /** The camera whose frustum is drawn nearest to a click (within 28 px of
      its centre or image plane), or undefined. */
  pickCamera(clientX: number, clientY: number): number | undefined {
    if (!this.showFrustums || !this.frustums || !this.cameras.length) return undefined;
    const rect = this.canvas.getBoundingClientRect();
    const s = frustumScale(this.cameras);
    this.world.updateMatrixWorld(true);
    let best: { i: number; d: number } | undefined;
    this.cameras.forEach((cam, i) => {
      const c = cam.center;
      const mid = cam.corners
        .reduce<Vec3>((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4, a[2] + p[2] / 4], [0, 0, 0])
        .map((v, k) => c[k]! + (v - c[k]!) * s) as Vec3;
      for (const p of [c, mid]) {
        const v = new THREE.Vector3(...p).applyMatrix4(this.world.matrixWorld).project(this.camera);
        if (v.z < -1 || v.z > 1) continue;
        const x = rect.left + ((v.x + 1) / 2) * rect.width;
        const y = rect.top + ((1 - v.y) / 2) * rect.height;
        const d = Math.hypot(x - clientX, y - clientY);
        if (d < 28 && (!best || d < best.d)) best = { i, d };
      }
    });
    return best?.i;
  }

  /** Flies to camera `index` and looks through it with its field of view,
      so the render lines up with the photo. */
  flyTo(index: number) {
    const cam = this.cameras[index];
    if (!cam) return;
    this.world.updateMatrixWorld(true);
    const m = this.world.matrixWorld;
    const view = cameraView(cam);
    const position = new THREE.Vector3(...cam.center).applyMatrix4(m);
    const forward = new THREE.Vector3(...view.forward).transformDirection(m);
    const up = new THREE.Vector3(...view.up).transformDirection(m);
    const look = new THREE.Matrix4().lookAt(position, position.clone().add(forward), up);
    const distance = Math.max(this.controls.target.distanceTo(this.camera.position), 1e-3);
    this.fly = {
      from: { position: this.camera.position.clone(), quaternion: this.camera.quaternion.clone(), fov: this.camera.fov },
      to: {
        position,
        quaternion: new THREE.Quaternion().setFromRotationMatrix(look),
        fov: view.fovY,
        target: position.clone().add(forward.multiplyScalar(distance)),
      },
      start: performance.now(),
    };
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) this.fly.start = -Infinity;
    this.viewing = index;
    this.syncFrustums();
    this.controls.enabled = false;
    this.controls.autoRotate = false;
  }

  private step(time: number) {
    const fly = this.fly!;
    const t = Math.min(1, (time - fly.start) / FLY_MS);
    const e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
    this.camera.position.lerpVectors(fly.from.position, fly.to.position, e);
    this.camera.quaternion.slerpQuaternions(fly.from.quaternion, fly.to.quaternion, e);
    this.camera.fov = fly.from.fov + (fly.to.fov - fly.from.fov) * e;
    this.camera.updateProjectionMatrix();
    if (t >= 1) {
      this.controls.target.copy(fly.to.target);
      this.fly = undefined;
    }
  }

  /** Back to orbiting (keeps the position; restores the field of view). */
  leaveCamera() {
    if (this.viewing === undefined) return;
    this.viewing = undefined;
    this.fly = undefined;
    this.syncFrustums();
    this.camera.fov = DEFAULT_FOV;
    this.camera.updateProjectionMatrix();
    this.controls.enabled = true;
    this.onLeaveCamera?.();
  }

  setTurntable(on: boolean, speed = 2) {
    if (on) this.leaveCamera();
    this.controls.autoRotate = on;
    this.controls.autoRotateSpeed = speed;
  }

  /** A starting crop box in view axes (y up): around the orbit centre,
      as wide as the cameras stand from it, or around what is shown. */
  bounds(): Box | undefined {
    this.world.updateMatrixWorld(true);
    const target = this.controls.target;
    if (this.cameras.length >= 2) {
      const d = this.cameras
        .map((cam) => new THREE.Vector3(...cam.center).applyMatrix4(this.world.matrixWorld).distanceTo(target))
        .sort((x, y) => x - y);
      const r = d[d.length >> 1]!;
      return { center: [target.x, target.y, target.z], half: [r, r, r] };
    }
    const shown = this.mesh ?? this.points;
    if (!shown) return undefined;
    const box = shown.getBoundingBox(true).clone().applyMatrix4(shown.matrixWorld);
    const c = box.getCenter(new THREE.Vector3());
    const h = box.getSize(new THREE.Vector3()).multiplyScalar(0.5);
    return { center: [c.x, c.y, c.z], half: [h.x, h.y, h.z] };
  }

  /** Hides every splat outside `box` (view axes) and outlines it;
      undefined shows everything again. Display only: downloads are whole. */
  setCrop(box: Box | undefined) {
    if (!box) {
      if (this.crop) {
        this.scene.remove(this.crop.outline);
        this.crop.outline.geometry.dispose();
        (this.crop.outline.material as THREE.Material).dispose();
        this.crop = undefined;
      }
    } else {
      if (!this.crop) {
        const outline = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
          new THREE.LineBasicMaterial({ color: this.frustumColor() }),
        );
        this.scene.add(outline);
        this.crop = { box, outline };
      }
      this.crop.box = box;
      this.crop.outline.position.set(...box.center);
      this.crop.outline.scale.set(...box.half);
    }
    for (const mesh of [this.mesh, this.points]) if (mesh) this.cropMesh(mesh);
  }

  /* Spark evaluates edit shapes against a mesh's own (local) splat
     positions, so each mesh gets its own edit with the box in its frame. */
  private cropMesh(mesh: SplatMesh) {
    if (!this.crop) {
      mesh.edits = null;
      return;
    }
    this.world.updateMatrixWorld(true);
    const { center, half } = this.crop.box;
    const box = new THREE.Matrix4().compose(new THREE.Vector3(...center), new THREE.Quaternion(), new THREE.Vector3(...half));
    const local = mesh.matrixWorld.clone().invert().multiply(box);
    // One edit per mesh, moved in place: a new edit object costs a rebuild.
    let edit = mesh.edits?.[0];
    if (!edit) {
      const sdf = new SplatEditSdf({ type: SplatEditSdfType.BOX, opacity: 0 });
      // Multiply alpha by 0 outside the box.
      edit = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY, invert: true, sdfs: [sdf] });
      mesh.edits = [edit];
    }
    const sdf = edit.sdfs![0]!;
    local.decompose(sdf.position, sdf.quaternion, sdf.scale);
    sdf.updateMatrixWorld();
  }



  /** Orbit around the splats' centre; look from the first camera when there
      is one, else from a distance that fits the centres' box. */
  frame() {
    this.world.updateMatrixWorld(true);
    const box = new THREE.Box3();
    const shown = this.mesh ?? this.points;
    if (shown) box.copy(shown.getBoundingBox(true)).applyMatrix4(shown.matrixWorld);
    for (const cam of this.cameras) box.expandByPoint(new THREE.Vector3(...cam.center).applyMatrix4(this.world.matrixWorld));
    if (box.isEmpty()) return;
    this.leaveCamera();
    const focus = focusPoint(this.cameras);
    const center = focus ? new THREE.Vector3(...focus).applyMatrix4(this.world.matrixWorld) : box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
    const first = this.cameras[0];
    if (first) {
      // Behind the first camera, so the cameras show around the scene.
      const from = new THREE.Vector3(...first.center).applyMatrix4(this.world.matrixWorld);
      this.camera.position.copy(center).add(from.sub(center).multiplyScalar(1.6));
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
