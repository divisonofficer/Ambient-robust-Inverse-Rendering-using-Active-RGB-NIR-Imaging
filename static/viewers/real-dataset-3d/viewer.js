import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SH_C0 = 0.28209479177387814;
const IMG_EXT = "webp";

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const sigmoid = (value) => 1 / (1 + Math.exp(-value));

export function mountDatasetViewer({ container, basePath }) {
  const viewer = new DatasetViewer(container);
  if (basePath) viewer.loadScene(basePath);
  return viewer;
}

class DatasetViewer {
  constructor(container) {
    this.container = container;
    this.modality = "rgb";
    this.basePath = "";
    this.frames = [];
    this.textures = new Map();
    this.textureRecords = new Map();
    this.textureQueue = [];
    this.textureLoader = new THREE.TextureLoader();
    this.loadingTextureCount = 0;
    this.maxConcurrentTextureLoads = 3;
    this.lastTextureByModality = { rgb: null, nir: null };
    this.raf = 0;
    this.currentFrameId = -1;
    this.depthFrameId = -1;
    this.sceneToken = 0;
    this.ready = false;
    this.paused = false;
    this.splitTop = null;

    this.container.innerHTML = "";
    this.container.classList.add("relative", "overflow-hidden", "bg-black");

    this.canvas = document.createElement("canvas");
    this.canvas.className = "absolute inset-0 h-full w-full";
    this.canvas.tabIndex = 0;
    this.container.appendChild(this.canvas);

    this.hud = document.createElement("div");
    this.hud.className =
      "hidden";
    this.hud.textContent = "loading...";
    this.container.appendChild(this.hud);

    this.error = document.createElement("div");
    this.error.className =
      "absolute inset-0 hidden items-center justify-center bg-black px-8 text-center text-sm leading-6 text-red-200";
    this.container.appendChild(this.error);

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x080808);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.7;
    this.controls.enableRotate = true;

    this.rgbPlane = null;
    this.nirPlane = null;
    this.rgbDepthMesh = null;
    this.nirDepthMesh = null;
    this.depthGeometry = null;
    this.frustumGroup = null;
    this.showFrustums = false;
    this.points = null;
    this.pointData = null;
    this.center = new THREE.Vector3();
    this.radius = 1;
    this.objectRadius = 1;
    this.fovX = 0.6;
    this.aspectFromData = 1;
    this.depthWidth = 160;
    this.depthStrength = 0.25;
    this.prefetchRadius = 10;
    this.prefetchedFrameIds = new Set();
    this.sharedOrbit = null;
    this.textureToken = 0;

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.resume();
  }

  async loadScene(basePath) {
    const token = ++this.sceneToken;
    this.basePath = basePath.endsWith("/") ? basePath : `${basePath}/`;
    this.currentFrameId = -1;
    this.depthFrameId = -1;
    this.ready = false;
    this.showError("");
    this.setHud("loading scene...");
    this.clearScene();

    if (window.location.protocol !== "http:" && window.location.protocol !== "https:") {
      this.showError("The integrated 3D viewer requires HTTP/HTTPS static hosting.");
      return;
    }

    try {
      const [train, test] = await Promise.all([
        fetch(`${this.basePath}transforms_train.json`).then((response) => response.json()),
        fetch(`${this.basePath}transforms_test.json`)
          .then((response) => (response.ok ? response.json() : null))
          .catch(() => null),
      ]);
      if (token !== this.sceneToken) return;

      this.setupFrames(train, test);

      try {
        const ply = await fetch(`${this.basePath}point_cloud.ply`).then((response) => {
          if (!response.ok) throw new Error("point_cloud.ply not found");
          return response.arrayBuffer();
        });
        if (token !== this.sceneToken) return;
        this.buildPointCloud(parsePly(ply));
      } catch (error) {
        console.warn("[dataset-viewer] point cloud unavailable, using flat image fallback:", error);
      }

      this.finishCameraSetup();
      this.buildFrustums();
      this.ready = true;
      this.updateNearestFrame(true);
      this.resize();
      this.resume();
    } catch (error) {
      console.error(error);
      this.showError(`Failed to load dataset viewer assets: ${error.message || error}`);
    }
  }

  setModality(modality) {
    this.modality = modality === "nir" ? "nir" : "rgb";
    this.currentFrameId = -1;
    this.depthFrameId = -1;
    this.updateNearestFrame(true);
    this.updateLayerState();
  }

  setRgbNirSplit(splitTop) {
    this.splitTop = Number.isFinite(splitTop) ? clamp(splitTop, 0, 1) : null;
    this.updateLayerState();
  }

  pause() {
    this.paused = true;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
  }

  resume() {
    if (!this.paused && this.raf) return;
    this.paused = false;
    this.animate();
  }

  dispose() {
    this.pause();
    this.resizeObserver.disconnect();
    this.clearScene();
    this.controls.dispose();
    this.renderer.dispose();
    this.container.innerHTML = "";
  }

  clearScene() {
    for (const child of [...this.scene.children]) {
      this.scene.remove(child);
      disposeObject(child);
    }
    this.textures.clear();
    for (const record of this.textureRecords.values()) record.texture?.dispose?.();
    this.textureRecords.clear();
    this.textureQueue = [];
    this.loadingTextureCount = 0;
    this.lastTextureByModality = { rgb: null, nir: null };
    this.textureToken += 1;
    this.frames = [];
    this.rgbPlane = null;
    this.nirPlane = null;
    this.rgbDepthMesh = null;
    this.nirDepthMesh = null;
    this.depthGeometry = null;
    this.frustumGroup = null;
    this.showFrustums = false;
    this.points = null;
    this.pointData = null;
    this.ready = false;
    this.prefetchedFrameIds.clear();
  }

  setupFrames(train, test) {
    this.fovX = train.camera_angle_x || 0.6;
    this.aspectFromData = train.h && train.w ? train.h / train.w : 1;
    const frames = [];
    const pushFrames = (data, split) => {
      if (!data?.frames) return;
      for (const frame of data.frames) {
        const matrix = frame.transform_matrix;
        const pos = columnVector(matrix, 3);
        frames.push({
          id: frames.length,
          split,
          stem: frame.file_path.split("/").pop(),
          posV: pos,
          dirV: columnVector(matrix, 2).multiplyScalar(-1).normalize(),
          rightV: columnVector(matrix, 0).normalize(),
          upV: columnVector(matrix, 1).normalize(),
        });
      }
    };

    pushFrames(train, "train");
    pushFrames(test, "test");
    this.frames = frames;
    this.center = new THREE.Vector3();
    for (const frame of this.frames) this.center.add(frame.posV);
    if (this.frames.length) this.center.multiplyScalar(1 / this.frames.length);
  }

  buildPointCloud(parsed) {
    this.pointData = parsed;
    if (parsed.centroid) this.center.copy(parsed.centroid);
    this.objectRadius = parsed.radius || this.objectRadius;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(parsed.positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(parsed.colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.006,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      sizeAttenuation: true,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.visible = false;
    this.scene.add(this.points);
  }

  finishCameraSetup() {
    const up = new THREE.Vector3();
    for (const frame of this.frames) up.add(frame.posV.clone().sub(this.center));
    up.normalize();
    if (up.lengthSq() > 0) this.camera.up.copy(up);

    this.radius = 0;
    for (const frame of this.frames) this.radius += frame.posV.distanceTo(this.center);
    this.radius = this.frames.length ? this.radius / this.frames.length : 1;

    for (const frame of this.frames) {
      frame.fromCenter = frame.posV.clone().sub(this.center).normalize();
      frame.toCenter = Math.max(0.1, this.center.clone().sub(frame.posV).dot(frame.dirV));
    }

    const first = this.frames[0];
    if (first) {
      const startDirection = first.posV.clone().sub(this.center).normalize();
      const framingRadius = Math.max(this.objectRadius || 0, this.radius * 0.12);
      const startDistance = Math.max(
        framingRadius / Math.tan(THREE.MathUtils.degToRad(this.camera.fov) * 0.5) * 1.35,
        this.radius * 0.16
      );
      this.camera.position.copy(this.center).addScaledVector(startDirection, startDistance);
    } else {
      this.camera.position.set(0, 0, this.radius * 0.8);
    }

    this.controls.target.copy(this.center);
    this.controls.minDistance = Math.max((this.objectRadius || this.radius * 0.12) * 0.7, this.radius * 0.04);
    this.controls.maxDistance = Math.max(this.radius * 2.5, this.objectRadius * 8);
    this.controls.update();
    if (this.sharedOrbit) this.setSharedOrbit(this.sharedOrbit.yaw, this.sharedOrbit.pitch);

    if (this.points?.material) this.points.material.size = Math.max(this.radius * 0.004, 0.002);
  }

  setSharedOrbit(yaw, pitch) {
    this.sharedOrbit = { yaw, pitch };
    const distance = this.camera.position.distanceTo(this.center);
    const phi = clamp(pitch, 0.08, Math.PI - 0.08);
    const sinPhi = Math.sin(phi);
    const direction = new THREE.Vector3(
      Math.sin(yaw) * sinPhi,
      Math.cos(phi),
      Math.cos(yaw) * sinPhi
    );

    this.camera.position.copy(this.center).addScaledVector(direction, distance);
    this.camera.lookAt(this.center);
    this.controls.target.copy(this.center);
    this.controls.update();
    this.updateNearestFrame(true);
  }

  buildFrustums() {
    this.frustumGroup = new THREE.Group();
    this.frustumGroup.visible = this.showFrustums;
    const halfX = Math.tan(this.fovX / 2);
    const depth = this.radius * 0.12;

    for (const frame of this.frames) {
      const halfW = depth * halfX;
      const halfH = halfW * this.aspectFromData;
      const apex = frame.posV;
      const center = frame.dirV.clone().multiplyScalar(depth);
      const right = frame.rightV.clone().multiplyScalar(halfW);
      const up = frame.upV.clone().multiplyScalar(halfH);
      const tl = apex.clone().add(center).add(up).sub(right);
      const tr = apex.clone().add(center).add(up).add(right);
      const br = apex.clone().add(center).sub(up).add(right);
      const bl = apex.clone().add(center).sub(up).sub(right);
      const points = [apex, tl, apex, tr, apex, br, apex, bl, tl, tr, tr, br, br, bl, bl, tl];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const color = frame.split === "test" ? 0xff6666 : 0x39a7ff;
      const line = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
      line.userData = { baseColor: color };
      frame.line = line;
      this.frustumGroup.add(line);
    }

    this.scene.add(this.frustumGroup);
  }

  updateNearestFrame(force = false) {
    if (!this.ready || !this.frames.length) return;

    const view = this.camera.position.clone().sub(this.center).normalize();
    let best = this.frames[0];
    let bestDot = -2;
    for (const frame of this.frames) {
      if (!frame.fromCenter) continue;
      const dot = view.dot(frame.fromCenter);
      if (dot > bestDot) {
        bestDot = dot;
        best = frame;
      }
    }

    if (!force && best.id === this.currentFrameId) return;
    this.currentFrameId = best.id;
    this.placeDepthWarp(best);
    this.prefetchNeighborTextures(best);
    this.highlightFrame(best);
    this.setHud(`depth-warp - frame ${best.id} - ${best.split}\nDrag to rotate. Hover vertically for RGB/NIR split.`);
  }

  prefetchNeighborTextures(activeFrame) {
    if (!activeFrame?.fromCenter || !this.frames.length) return;

    const neighbors = this.frames
      .filter((frame) => frame.id !== activeFrame.id && frame.fromCenter)
      .map((frame) => ({
        frame,
        score: activeFrame.fromCenter.dot(frame.fromCenter),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, this.prefetchRadius)
      .map((item) => item.frame);

    for (const frame of [activeFrame, ...neighbors]) {
      if (this.prefetchedFrameIds.has(frame.id)) continue;
      this.prefetchedFrameIds.add(frame.id);
      this.requestTexture(frame, "rgb");
      this.requestTexture(frame, "nir");
    }
  }

  placeDepthWarp(frame) {
    if (!this.pointData) {
      this.placeImagePlanes(frame);
      return;
    }

    if (this.depthFrameId !== frame.id || !this.rgbDepthMesh || !this.nirDepthMesh) {
      const geometry = this.buildDepthGeometry(frame, this.depthWidth);
      if (!this.rgbDepthMesh) {
        this.rgbDepthMesh = new THREE.Mesh(geometry, createSplitMaterial("rgb"));
        this.rgbDepthMesh.renderOrder = 1;
        this.rgbDepthMesh.frustumCulled = false;
        this.scene.add(this.rgbDepthMesh);

        this.nirDepthMesh = new THREE.Mesh(geometry, createSplitMaterial("nir"));
        this.nirDepthMesh.renderOrder = 2;
        this.nirDepthMesh.frustumCulled = false;
        this.scene.add(this.nirDepthMesh);
      } else {
        this.depthGeometry?.dispose();
        this.rgbDepthMesh.geometry = geometry;
        this.nirDepthMesh.geometry = geometry;
      }
      this.depthGeometry = geometry;
      this.depthFrameId = frame.id;
    }

    this.rgbDepthMesh.material.map = this.getTexture(frame, "rgb");
    this.nirDepthMesh.material.map = this.getTexture(frame, "nir");
    this.rgbDepthMesh.material.needsUpdate = true;
    this.nirDepthMesh.material.needsUpdate = true;
    this.rgbDepthMesh.visible = true;
    this.nirDepthMesh.visible = true;

    if (this.rgbPlane) this.rgbPlane.visible = false;
    if (this.nirPlane) this.nirPlane.visible = false;
    if (this.points) this.points.visible = false;
    if (this.frustumGroup) this.frustumGroup.visible = this.showFrustums;
    this.updateLayerState();
  }

  rasterizeDepth(frame, width) {
    const height = Math.max(8, Math.round(width * this.aspectFromData));
    const inf = 1e20;
    const depth = new Float32Array(width * height).fill(inf);
    const tanX = Math.tan(this.fovX / 2);
    const tanY = tanX * this.aspectFromData;
    const positions = this.pointData.positions;
    const scales = this.pointData.scales;
    const count = this.pointData.count;

    for (let i = 0; i < count; i++) {
      const ax = positions[i * 3] - frame.posV.x;
      const ay = positions[i * 3 + 1] - frame.posV.y;
      const az = positions[i * 3 + 2] - frame.posV.z;
      const z = ax * frame.dirV.x + ay * frame.dirV.y + az * frame.dirV.z;
      if (z <= 0.001) continue;

      const x = ax * frame.rightV.x + ay * frame.rightV.y + az * frame.rightV.z;
      const y = ax * frame.upV.x + ay * frame.upV.y + az * frame.upV.z;
      const u = 0.5 + 0.5 * x / (z * tanX);
      const v = 0.5 - 0.5 * y / (z * tanY);
      if (u < 0 || u > 1 || v < 0 || v > 1) continue;

      const cx = Math.floor(u * (width - 1));
      const cy = Math.floor(v * (height - 1));
      const scale = scales ? scales[i] : this.radius * 0.004;
      const rad = clamp(Math.ceil(scale * (width / (2 * z * tanX)) * 2.0), 1, 8);
      const r2 = rad * rad;

      for (let yy = -rad; yy <= rad; yy++) {
        const qy = cy + yy;
        if (qy < 0 || qy >= height) continue;
        for (let xx = -rad; xx <= rad; xx++) {
          if (xx * xx + yy * yy > r2) continue;
          const qx = cx + xx;
          if (qx < 0 || qx >= width) continue;
          const index = qy * width + qx;
          if (z < depth[index]) depth[index] = z;
        }
      }
    }

    return { depth, width, height, inf };
  }

  dilateDepth(depth, width, height, inf, passes) {
    let current = depth;
    for (let pass = 0; pass < passes; pass++) {
      const next = new Float32Array(current);
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          const index = y * width + x;
          if (current[index] < inf) continue;
          let best = inf;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              best = Math.min(best, current[(y + dy) * width + (x + dx)]);
            }
          }
          if (best < inf) next[index] = best;
        }
      }
      current = next;
    }
    return current;
  }

  buildDepthGeometry(frame, width) {
    const raster = this.rasterizeDepth(frame, width);
    const { height, inf } = raster;
    const depth = this.dilateDepth(raster.depth, width, height, inf, 3);
    const tanX = Math.tan(this.fovX / 2);
    const tanY = tanX * this.aspectFromData;
    const z0 = Math.max(0.1, frame.toCenter);
    const zUsed = new Float32Array(width * height);
    const positions = [];
    const uvs = [];

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = y * width + x;
        const rawZ = depth[index] < inf ? depth[index] : z0;
        const z = z0 + (rawZ - z0) * this.depthStrength;
        zUsed[index] = z;

        const u = x / (width - 1);
        const v = y / (height - 1);
        const sx = (u - 0.5) * 2 * tanX;
        const sy = (0.5 - v) * 2 * tanY;

        positions.push(
          frame.posV.x + frame.dirV.x * z + frame.rightV.x * sx * z + frame.upV.x * sy * z,
          frame.posV.y + frame.dirV.y * z + frame.rightV.y * sx * z + frame.upV.y * sy * z,
          frame.posV.z + frame.dirV.z * z + frame.rightV.z * sx * z + frame.upV.z * sy * z
        );
        uvs.push(u, 1 - v);
      }
    }

    const indices = [];
    const maxJump = this.radius * (0.08 + 0.25 * (1 - this.depthStrength));
    const ok = (a, b, c) => Math.max(zUsed[a], zUsed[b], zUsed[c]) - Math.min(zUsed[a], zUsed[b], zUsed[c]) < maxJump;
    for (let y = 0; y < height - 1; y++) {
      for (let x = 0; x < width - 1; x++) {
        const i00 = y * width + x;
        const i10 = i00 + 1;
        const i01 = i00 + width;
        const i11 = i01 + 1;
        if (ok(i00, i10, i01)) indices.push(i00, i10, i01);
        if (ok(i10, i11, i01)) indices.push(i10, i11, i01);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    return geometry;
  }

  placeImagePlanes(frame) {
    const distance = Math.max(0.1, frame.toCenter);
    const planeW = 2 * distance * Math.tan(this.fovX / 2);
    const planeH = planeW * this.aspectFromData;

    if (!this.rgbPlane) {
      const geometry = new THREE.PlaneGeometry(1, 1);
      this.rgbPlane = new THREE.Mesh(geometry, createSplitMaterial("rgb"));
      this.rgbPlane.renderOrder = 1;
      this.scene.add(this.rgbPlane);

      this.nirPlane = new THREE.Mesh(geometry, createSplitMaterial("nir"));
      this.nirPlane.renderOrder = 2;
      this.scene.add(this.nirPlane);
    }

    const basis = new THREE.Matrix4().makeBasis(frame.rightV, frame.upV, frame.dirV.clone().negate());
    for (const plane of [this.rgbPlane, this.nirPlane]) {
      plane.quaternion.setFromRotationMatrix(basis);
      plane.position.copy(frame.posV).addScaledVector(frame.dirV, distance);
      plane.scale.set(planeW, planeH, 1);
      plane.visible = true;
    }

    this.rgbPlane.material.map = this.getTexture(frame, "rgb");
    this.nirPlane.material.map = this.getTexture(frame, "nir");
    this.rgbPlane.material.needsUpdate = true;
    this.nirPlane.material.needsUpdate = true;
    if (this.rgbDepthMesh) this.rgbDepthMesh.visible = false;
    if (this.nirDepthMesh) this.nirDepthMesh.visible = false;
    this.updateLayerState();
  }

  getTexture(frame, modality) {
    const record = this.requestTexture(frame, modality, true);
    if (record?.status === "loaded" && record.texture) return record.texture;

    return this.findLoadedNeighborTexture(frame, modality) || this.lastTextureByModality[modality] || null;
  }

  requestTexture(frame, modality, priority = false) {
    if (!frame) return null;
    const key = `${modality}:${frame.stem}`;
    let record = this.textureRecords.get(key);
    if (record) {
      if (priority && record.status === "queued") {
        this.textureQueue = this.textureQueue.filter((item) => item !== record);
        this.textureQueue.unshift(record);
      }
      return record;
    }

    const folder = modality === "nir" ? "nir_rgba" : "rgba";
    const url = `${this.basePath}viewer_cache/${folder}/${frame.stem}.${IMG_EXT}`;
    record = {
      key,
      url,
      frameId: frame.id,
      modality,
      status: "queued",
      texture: null,
      token: this.textureToken,
    };
    this.textureRecords.set(key, record);

    if (priority) this.textureQueue.unshift(record);
    else this.textureQueue.push(record);
    this.pumpTextureQueue();
    return record;
  }

  pumpTextureQueue() {
    while (this.loadingTextureCount < this.maxConcurrentTextureLoads && this.textureQueue.length) {
      const record = this.textureQueue.shift();
      if (!record || record.status !== "queued") continue;

      const token = this.textureToken;
      record.status = "loading";
      record.token = token;
      this.loadingTextureCount += 1;

      this.textureLoader.load(
        record.url,
        (texture) => {
          this.loadingTextureCount = Math.max(0, this.loadingTextureCount - 1);
          if (record.token !== this.textureToken || token !== this.textureToken) {
            texture.dispose();
            this.pumpTextureQueue();
            return;
          }

          texture.colorSpace = THREE.SRGBColorSpace;
          record.texture = texture;
          record.status = "loaded";
          this.textures.set(record.key, texture);
          this.lastTextureByModality[record.modality] = texture;

          if (record.frameId === this.currentFrameId) this.updateNearestFrame(true);
          this.pumpTextureQueue();
        },
        undefined,
        () => {
          this.loadingTextureCount = Math.max(0, this.loadingTextureCount - 1);
          record.status = "failed";
          this.showError(`Could not load ${record.modality.toUpperCase()} preview texture.`);
          this.pumpTextureQueue();
        }
      );
    }
  }

  findLoadedNeighborTexture(frame, modality) {
    let bestTexture = null;
    let bestScore = -2;
    for (const candidate of this.frames) {
      if (!candidate.fromCenter || candidate.id === frame.id) continue;
      const record = this.textureRecords.get(`${modality}:${candidate.stem}`);
      if (record?.status !== "loaded" || !record.texture) continue;
      const score = frame.fromCenter?.dot(candidate.fromCenter) ?? -2;
      if (score > bestScore) {
        bestScore = score;
        bestTexture = record.texture;
      }
    }
    return bestTexture;
  }

  updateLayerState() {
    const size = new THREE.Vector2();
    this.renderer.getDrawingBufferSize(size);
    const splitEnabled = this.splitTop !== null;
    const splitFromBottom = splitEnabled ? size.y * (1 - this.splitTop) : 0;
    const nirMode = splitEnabled ? 1 : this.modality === "nir" ? 2 : 0;
    const showRgb = splitEnabled || this.modality === "rgb";

    for (const mesh of [this.rgbDepthMesh, this.rgbPlane]) {
      if (mesh) mesh.visible = showRgb;
    }
    for (const mesh of [this.nirDepthMesh, this.nirPlane]) {
      if (!mesh) continue;
      mesh.visible = nirMode > 0;
      const uniforms = mesh.material.userData.splitUniforms;
      if (uniforms) {
        uniforms.uSplitMode.value = nirMode;
        uniforms.uSplitY.value = splitFromBottom;
      }
    }
  }

  highlightFrame(active) {
    for (const frame of this.frames) {
      if (!frame.line) continue;
      frame.line.material.color.setHex(frame.id === active.id ? 0xffffff : frame.line.userData.baseColor);
      frame.line.material.opacity = frame.id === active.id ? 1 : 0.35;
    }
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this.updateLayerState();
  }

  animate() {
    if (this.paused) return;
    this.raf = requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.updateNearestFrame();
    this.renderer.render(this.scene, this.camera);
  }

  setHud(text) {
    this.hud.textContent = text;
  }

  showError(message) {
    if (!message) {
      this.error.classList.add("hidden");
      this.error.textContent = "";
      return;
    }
    this.error.textContent = message;
    this.error.classList.remove("hidden");
    this.error.classList.add("flex");
  }
}

function createSplitMaterial(kind) {
  const material = new THREE.MeshBasicMaterial({
    transparent: true,
    alphaTest: 0.005,
    depthTest: true,
    depthWrite: kind === "rgb",
    side: THREE.DoubleSide,
    polygonOffset: kind === "nir",
    polygonOffsetFactor: kind === "nir" ? -1 : 0,
    polygonOffsetUnits: kind === "nir" ? -1 : 0,
  });

  if (kind === "nir") {
    const uniforms = {
      uSplitMode: { value: 0 },
      uSplitY: { value: 0 },
    };
    material.userData.splitUniforms = uniforms;
    material.onBeforeCompile = (shader) => {
      shader.uniforms.uSplitMode = uniforms.uSplitMode;
      shader.uniforms.uSplitY = uniforms.uSplitY;
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <clipping_planes_pars_fragment>",
        "#include <clipping_planes_pars_fragment>\nuniform float uSplitMode;\nuniform float uSplitY;"
      );
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <clipping_planes_fragment>",
        "#include <clipping_planes_fragment>\nif (uSplitMode < 0.5) discard;\nif (uSplitMode < 1.5 && gl_FragCoord.y > uSplitY) discard;"
      );
    };
  }

  return material;
}

function columnVector(matrix, column) {
  return new THREE.Vector3(matrix[0][column], matrix[1][column], matrix[2][column]);
}

function parsePly(buffer) {
  const bytes = new Uint8Array(buffer);
  const header = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 30000)));
  const endToken = "end_header\n";
  const endIndex = header.indexOf(endToken);
  if (endIndex < 0) throw new Error("PLY header end not found");

  const dataOffset = endIndex + endToken.length;
  const lines = header.slice(0, endIndex).split("\n");
  let count = 0;
  const props = [];
  for (const line of lines) {
    if (line.startsWith("element vertex")) count = Number.parseInt(line.split(/\s+/).pop(), 10);
    else if (line.startsWith("property")) props.push(line.split(/\s+/).pop());
  }

  const propIndex = {};
  props.forEach((name, index) => {
    propIndex[name] = index;
  });

  const stride = props.length * 4;
  const view = new DataView(buffer);
  const read = (base, name, fallback = 0) => {
    const index = propIndex[name];
    return index === undefined ? fallback : view.getFloat32(base + index * 4, true);
  };

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const scales = new Float32Array(count);
  const center = new THREE.Vector3();
  let weightSum = 0;

  for (let i = 0; i < count; i++) {
    const base = dataOffset + i * stride;
    const x = read(base, "x");
    const y = read(base, "y");
    const z = read(base, "z");
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    colors[i * 3] = clamp(0.5 + SH_C0 * read(base, "f_dc_0", 0), 0, 1);
    colors[i * 3 + 1] = clamp(0.5 + SH_C0 * read(base, "f_dc_1", 0), 0, 1);
    colors[i * 3 + 2] = clamp(0.5 + SH_C0 * read(base, "f_dc_2", 0), 0, 1);

    const alpha = sigmoid(read(base, "opacity", 1));
    scales[i] = Math.max(Math.exp(read(base, "scale_0", -5)), Math.exp(read(base, "scale_1", -5)));
    center.x += x * alpha;
    center.y += y * alpha;
    center.z += z * alpha;
    weightSum += alpha;
  }

  if (weightSum > 0) center.multiplyScalar(1 / weightSum);

  const distances = [];
  const step = Math.max(1, Math.floor(count / 4000));
  for (let i = 0; i < count; i += step) {
    const dx = positions[i * 3] - center.x;
    const dy = positions[i * 3 + 1] - center.y;
    const dz = positions[i * 3 + 2] - center.z;
    distances.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  distances.sort((a, b) => a - b);
  const radius = distances.length ? distances[Math.floor(distances.length * 0.9)] : 1;

  return { count, positions, colors, scales, centroid: center, radius };
}

function disposeObject(object) {
  object.traverse?.((child) => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => material.dispose?.());
    } else {
      child.material?.dispose?.();
    }
  });
}
