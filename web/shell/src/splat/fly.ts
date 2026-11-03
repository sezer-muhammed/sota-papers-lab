/* Free-fly (game-style) camera for the 3D viewer: WASD to move, Q/E or
   C/Space down/up, drag to look, Shift for speed, wheel for fly speed.
   Touch: one finger looks, two fingers move.
   Owned by SplatEngine, which only calls `update` while flying. */
import * as THREE from "three";

const MOVE_KEYS: Record<string, [number, number, number]> = {
  KeyW: [0, 0, -1],
  ArrowUp: [0, 0, -1],
  KeyS: [0, 0, 1],
  ArrowDown: [0, 0, 1],
  KeyA: [-1, 0, 0],
  ArrowLeft: [-1, 0, 0],
  KeyD: [1, 0, 0],
  ArrowRight: [1, 0, 0],
  KeyE: [0, 1, 0],
  Space: [0, 1, 0],
  KeyQ: [0, -1, 0],
  KeyC: [0, -1, 0],
};

/** Radians per pixel of drag. */
const LOOK = 0.004;
const MAX_PITCH = Math.PI / 2 - 0.01;

export class FlyControls {
  enabled = false;
  /** Scene units per second at speed factor 1 (set from the scene size). */
  baseSpeed = 1;
  /** Wheel-adjusted multiplier. */
  speedFactor = 1;
  /** Called on any input that moves the camera (leaves a camera view). */
  onInput?: () => void;
  /** Called when the wheel changes the speed. */
  onSpeed?: (factor: number) => void;
  private readonly held = new Set<string>();
  private fast = false;
  private drag?: { id: number; x: number; y: number };
  private readonly touches = new Map<number, { x: number; y: number }>();
  private readonly euler = new THREE.Euler(0, 0, 0, "YXZ");

  constructor(
    private readonly camera: THREE.PerspectiveCamera,
    private readonly dom: HTMLElement,
  ) {
    dom.addEventListener("keydown", this.keyDown);
    dom.addEventListener("keyup", this.keyUp);
    dom.addEventListener("blur", this.release);
    dom.addEventListener("pointerdown", this.pointerDown);
    dom.addEventListener("pointermove", this.pointerMove);
    dom.addEventListener("pointerup", this.pointerUp);
    dom.addEventListener("pointercancel", this.pointerUp);
    dom.addEventListener("wheel", this.wheel, { passive: false });
  }

  private keyDown = (e: KeyboardEvent) => {
    if (!this.enabled || e.metaKey || e.ctrlKey || e.altKey) return;
    this.fast = e.shiftKey;
    if (!MOVE_KEYS[e.code]) return;
    e.preventDefault();
    this.held.add(e.code);
    this.onInput?.();
  };

  private keyUp = (e: KeyboardEvent) => {
    this.fast = e.shiftKey;
    this.held.delete(e.code);
  };

  private release = () => {
    this.held.clear();
    this.fast = false;
  };

  private pointerDown = (e: PointerEvent) => {
    if (!this.enabled) return;
    if (e.pointerType === "touch") {
      this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      return;
    }
    this.drag = { id: e.pointerId, x: e.clientX, y: e.clientY };
  };

  private pointerMove = (e: PointerEvent) => {
    if (!this.enabled) return;
    if (e.pointerType === "touch") return this.touchDrag(e);
    const d = this.drag;
    if (!d || d.id !== e.pointerId) return;
    this.look(e.clientX - d.x, e.clientY - d.y);
    d.x = e.clientX;
    d.y = e.clientY;
  };

  private pointerUp = (e: PointerEvent) => {
    this.touches.delete(e.pointerId);
    if (this.drag?.id === e.pointerId) this.drag = undefined;
  };

  private touchDrag(e: PointerEvent) {
    const prev = this.touches.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    if (this.touches.size === 1) {
      this.look(dx, dy);
    } else {
      // Two fingers: drag up/down moves forward/back, sideways strafes;
      // 250 px of drag (both fingers) is a second of flight.
      const step = (this.speed() * 0.004) / this.touches.size;
      this.translate(new THREE.Vector3(dx * step, 0, dy * step));
      this.onInput?.();
    }
    prev.x = e.clientX;
    prev.y = e.clientY;
  }

  private wheel = (e: WheelEvent) => {
    if (!this.enabled) return;
    e.preventDefault();
    this.speedFactor = THREE.MathUtils.clamp(this.speedFactor * Math.exp(-e.deltaY * 0.002), 1 / 32, 32);
    this.onSpeed?.(this.speedFactor);
  };

  private look(dx: number, dy: number) {
    if (!dx && !dy) return;
    this.onInput?.();
    this.euler.setFromQuaternion(this.camera.quaternion);
    this.euler.y -= dx * LOOK;
    this.euler.x = THREE.MathUtils.clamp(this.euler.x - dy * LOOK, -MAX_PITCH, MAX_PITCH);
    this.euler.z = 0;
    this.camera.quaternion.setFromEuler(this.euler);
  }

  private speed() {
    return this.baseSpeed * this.speedFactor * (this.fast ? 4 : 1);
  }

  /** Moves by `v` in camera axes for x/z (horizontal heading) and world y. */
  private translate(v: THREE.Vector3) {
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const move = right.multiplyScalar(v.x).addScaledVector(forward, -v.z);
    move.y += v.y;
    this.camera.position.add(move);
  }

  /** Moves for the held keys; `dt` in seconds. */
  update(dt: number) {
    if (!this.enabled || !this.held.size) return;
    const dir = new THREE.Vector3();
    for (const code of this.held) dir.add(new THREE.Vector3(...MOVE_KEYS[code]!));
    if (dir.lengthSq() === 0) return;
    dir.normalize().multiplyScalar(this.speed() * Math.min(dt, 0.1));
    this.translate(dir);
  }

  dispose() {
    const dom = this.dom;
    dom.removeEventListener("keydown", this.keyDown);
    dom.removeEventListener("keyup", this.keyUp);
    dom.removeEventListener("blur", this.release);
    dom.removeEventListener("pointerdown", this.pointerDown);
    dom.removeEventListener("pointermove", this.pointerMove);
    dom.removeEventListener("pointerup", this.pointerUp);
    dom.removeEventListener("pointercancel", this.pointerUp);
    dom.removeEventListener("wheel", this.wheel);
  }
}
