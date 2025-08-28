# app2.py
import os, re, cv2, time, json, argparse, threading, asyncio, platform, urllib.request
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import uvicorn
import numpy as np

from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from shapely.geometry import box as shapely_box
from shapely.validation import make_valid

# ---------------- FastAPI (create ONCE, then add CORS) ----------------
app = FastAPI()

BEEP_FILE = os.path.join(os.path.dirname(__file__), "beep.wav")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("Multi-camera backend (YOLO + zones + OpenCV face allowlist)")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLOv8 *.pt (COCO)")
    ap.add_argument("--cams", default="cameraA=0,cameraB=1",
                    help='Comma list of logical ids: "cameraA=0,cameraB=1". Only the KEYS matter; sources are chosen at runtime by the UI.')
    ap.add_argument("--zones", default="zones.json", help="Default zones file (fallback)")
    ap.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--draw-zones", action="store_true",
                    help="Open polygon drawer for a single cam (use --cam-id).")
    ap.add_argument("--cam-id", default=None, help="When --draw-zones, which camera id to use.")
    ap.add_argument("--hit", choices=["center","overlap","feet"], default="overlap",
                    help="Intrusion rule: overlap (any part), center (bbox centroid), or feet (bottom-center).")
    ap.add_argument("--overlap-thresh", type=float, default=0.05,
                    help="Min overlap ratio bbox∩zone / bbox area for --hit overlap.")

    # ---- Face allowlist (OpenCV YuNet + SFace) ----
    ap.add_argument("--allowlist-dir", default="./allowed_faces",
                    help="Folder of allowed persons’ face images. Supports folders-per-person or single files.")
    ap.add_argument("--allow-thresh", type=float, default=0.65,    # <— stricter default
                    help="Cosine similarity threshold (0–1) for allowlist face match (SFace).")
    ap.add_argument("--allow-cache-sec", type=float, default=2.0,
                    help="Keep a short permit cache (seconds) after an allowed face is seen.")
    ap.add_argument("--models-dir", default="./models",
                    help="Directory containing YuNet and SFace ONNX files.")
    ap.add_argument("--autodownload-models", action="store_true",
                    help="If ONNX files are missing, download from OpenCV Zoo automatically.")

    # ---- NEW strictness/quality controls ----
    ap.add_argument("--strict", action="store_true",
                    help="Require best score to beat second-best by a margin for acceptance.")
    ap.add_argument("--margin", type=float, default=0.18,
                    help="Required (best - second_best) margin when --strict is set.")
    ap.add_argument("--min-face", type=int, default=80,
                    help="Reject faces smaller than this width or height in pixels (reduces false accepts).")

    return ap.parse_args()


ARGS = parse_args()

# ---------------- Parse cameras ----------------
def parse_cams(spec: str) -> Dict[str, str]:
    out = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f'Bad --cams token "{token}". Use id=source.')
        cid, _src = token.split("=", 1)
        cid = cid.strip()
        out[cid] = True
    if not out:
        raise ValueError("No cameras parsed from --cams.")
    return out

CAM_IDS: List[str] = list(parse_cams(ARGS.cams).keys())

# ---------------- Geometry helpers ----------------
def clamp_pts(pts, w, h):
    return [[max(0, min(w-1, int(x))), max(0, min(h-1, int(y)))] for x, y in pts]

def fix_poly(poly):
    try:
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        poly = poly.buffer(0)
        if poly.is_empty:
            raise ValueError("empty after fix")
    except Exception:
        poly = poly.convex_hull
    return poly

# ---------------- Globals ----------------
app.state.loop = None  # set at startup
latest_jpeg: Dict[str, Optional[bytes]] = {}  # filled when workers start
events_log: List[dict] = []
subscribers: List[asyncio.Queue] = []

# Worker registries
workers_by_id: Dict[str, "CameraWorker"] = {}
active_sources_by_id: Dict[str, int] = {}  # cam_id -> index

# --- COLORS ---
ZONE_COLOR   = (255, 0, 0)   # BLUE (zone)
SAFE_COLOR   = (0, 255, 0)   # GREEN (person safe)
ALERT_COLOR  = (0, 0, 255)   # RED   (person in zone)
ALLOW_COLOR  = (255, 165, 0) # ORANGE (allowed in zone)

# --------- Allowlist DB ----------
ALLOWLIST_DB: List[Tuple[str, np.ndarray]] = []  # [(name, embeddingN)]
ALLOWLIST_LOCK = threading.Lock()

# Recent-permit cache (per-cam): [(ts, (cx,cy))]
RECENT_ALLOW: Dict[str, List[Tuple[float, Tuple[int,int]]]] = {}
RECENT_LOCK = threading.Lock()

# --------- Model paths / autodownload ----------
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

def ensure_models(models_dir: str) -> Tuple[str, str]:
    os.makedirs(models_dir, exist_ok=True)
    yunet = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    sface = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")

    missing = []
    if not os.path.isfile(yunet): missing.append(("YuNet", yunet, YUNET_URL))
    if not os.path.isfile(sface): missing.append(("SFace", sface, SFACE_URL))

    if missing and getattr(ARGS, "autodownload_models", False):
        for name, path, url in missing:
            try:
                print(f"[models] downloading {name} from {url} -> {path}")
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"[models] failed to download {name}: {e}")

    for name, path, _ in [("YuNet", yunet, YUNET_URL), ("SFace", sface, SFACE_URL)]:
        if not os.path.isfile(path):
            raise RuntimeError(
                f"{name} ONNX not found at {path}. Place it there or run with --autodownload-models."
            )
    return yunet, sface

# --------- OpenCV face stack ----------
class FaceStack:
    """
    Thin wrapper around OpenCV YuNet (detector) + SFace (recognizer).
    """
    def __init__(self, yunet_path: str, sface_path: str):
        self.detector = cv2.FaceDetectorYN_create(
            model=yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF_create(sface_path, "")

    def set_size(self, w: int, h: int):
        self.detector.setInputSize((w, h))

    def detect(self, bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        try:
            faces = self.detector.detect(bgr)
        except cv2.error:
            faces = None
        if faces is None:
            return []
        dets = faces[1] if isinstance(faces, tuple) and len(faces) == 2 else faces
        out = []
        if dets is None:
            return out
        for det in dets:
            x, y, w, h = det[:4]
            score = det[-1] if det.shape[0] >= 15 else 1.0
            out.append((int(x), int(y), int(x+w), int(y+h), float(score)))
        return out

    def embed(self, bgr: np.ndarray, box: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = box
        face_box = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.int32)  # x,y,w,h
        try:
            aligned = self.recognizer.alignCrop(bgr, face_box)
            feat = self.recognizer.feature(aligned)
        except cv2.error:
            return None
        if feat is None:
            return None
        feat = feat.astype("float32").ravel()
        n = np.linalg.norm(feat) + 1e-9
        return feat / n

# --------- Allowed faces utilities ----------
def cosine_sim(a, b):
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def _recent_permit(cam_id: str, cx: int, cy: int) -> bool:
    with RECENT_LOCK:
        arr = RECENT_ALLOW.get(cam_id, [])
    now = time.time()
    arr = [(ts, pt) for (ts, pt) in arr if now - ts <= ARGS.allow_cache_sec]
    with RECENT_LOCK:
        RECENT_ALLOW[cam_id] = arr
    for _, (px, py) in arr:
        if (px - cx) ** 2 + (py - cy) ** 2 <= (50 ** 2):
            return True
    return False

def _add_recent_permit(cam_id: str, cx: int, cy: int):
    with RECENT_LOCK:
        arr = RECENT_ALLOW.get(cam_id, [])
        arr.append((time.time(), (cx, cy)))
        RECENT_ALLOW[cam_id] = arr

def load_allowlist_embeddings(dirpath: str, face_stack: FaceStack) -> List[Tuple[str, np.ndarray]]:
    if not dirpath or not os.path.isdir(dirpath):
        print(f"[allow] directory not found: {dirpath}")
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    subdirs = [os.path.join(dirpath, d) for d in os.listdir(dirpath)
               if os.path.isdir(os.path.join(dirpath, d))]
    using_subdirs = len(subdirs) > 0

    out: List[Tuple[str, np.ndarray]] = []

    if using_subdirs:
        print(f"[allow] using per-person folders under {dirpath}")
        for person_dir in sorted(subdirs):
            person = os.path.basename(person_dir).strip()
            img_paths = []
            for root, _, fns in os.walk(person_dir):
                for fn in fns:
                    if os.path.splitext(fn)[1].lower() in exts:
                        img_paths.append(os.path.join(root, fn))
            if not img_paths:
                print(f"[allow] {person}: no images")
                continue

            embs = []
            for fp in img_paths:
                img = cv2.imread(fp)
                if img is None:
                    continue
                h, w = img.shape[:2]
                face_stack.set_size(w, h)
                dets = face_stack.detect(img)
                if not dets:
                    continue
                dets.sort(key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
                box = dets[0][:4]
                emb = face_stack.embed(img, box)
                if emb is not None:
                    embs.append(emb)

            if not embs:
                print(f"[allow] {person}: no usable faces")
                continue

            mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
            out.append((person, mean_emb))
            print(f"[allow] added person '{person}' with {len(embs)} image(s)")
    else:
        files: List[str] = []
        for root, _, fns in os.walk(dirpath):
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in exts:
                    files.append(os.path.join(root, fn))
        for fp in sorted(files):
            img = cv2.imread(fp)
            if img is None:
                continue
            h, w = img.shape[:2]
            face_stack.set_size(w, h)
            dets = face_stack.detect(img)
            if not dets:
                print(f"[allow] no face detected in {fp}")
                continue
            dets.sort(key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
            box = dets[0][:4]
            emb = face_stack.embed(img, box)
            if emb is None:
                continue
            name = os.path.splitext(os.path.basename(fp))[0]
            out.append((name, emb))
            print(f"[allow] added '{name}' from {os.path.basename(fp)}")
    return out

def match_allowlist(emb: np.ndarray, thresh: float) -> Tuple[bool, Optional[str], float]:
    """
    Strict matcher:
      - score must be >= threshold
      - if --strict is ON, best score must exceed the second-best by --margin
    """
    with ALLOWLIST_LOCK:
        db = list(ALLOWLIST_DB)
    if not db:
        return (False, None, 0.0)

    scores = []
    for name, ref in db:
        scores.append((name, cosine_sim(emb, ref)))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = scores[0]
    second_best = scores[1][1] if len(scores) > 1 else -1.0

    accepted = best_score >= thresh
    if accepted and getattr(ARGS, "strict", False):
        if best_score - second_best < getattr(ARGS, "margin", 0.18):
            accepted = False

    return (accepted, best_name if accepted else None, best_score)


# ---------------- Devices (probe with OpenCV only) ----------------
def _probe_opencv_indices(max_index: int = 10) -> List[dict]:
    """
    Probe indices 0..max_index-1 with several backends and a short warm-up.
    """
    out = []
    for i in range(max_index):
        opened = False
        for backend in (0, cv2.CAP_DSHOW, cv2.CAP_MSMF):   # CAP_ANY first
            cap = cv2.VideoCapture(i, backend) if backend != 0 else cv2.VideoCapture(i)
            if not cap.isOpened():
                cap.release()
                continue
            ok = False
            t0 = time.time()
            while time.time() - t0 < 0.3:
                r, _ = cap.read()
                if r:
                    ok = True
                    break
                time.sleep(0.02)
            cap.release()
            if ok:
                opened = True
                break
        if opened:
            out.append({"index": i, "name": f"Camera #{i} (index {i})"})
    return out

def list_local_cams() -> List[dict]:
    # Avoid COM/pygrabber instability; rely on OpenCV probing everywhere
    return _probe_opencv_indices(max_index=10)

# ---------------- Health ----------------
@app.get("/health")
def health():
    return {"ok": True, "slots": CAM_IDS, "active": list(active_sources_by_id.keys())}

# ---------------- API for React ----------------
@app.get("/cam_ids")
def cam_ids():
    return {"cam_ids": CAM_IDS}

@app.get("/devices")
def devices():
    return {"devices": list_local_cams()}

class ActivateMapBody(BaseModel):
    map: Dict[str, str]  # {"cameraA": "Camera #0 (index 0)"} or {"cameraA": "0"}

def _parse_index_from_label(label: str) -> Optional[int]:
    label = str(label).strip()
    if label.isdigit():
        return int(label)
    m = re.search(r"\(index\s+(\d+)\)", label)
    if m:
        return int(m.group(1))
    try:
        return int(label)
    except Exception:
        return None

def stop_worker(cam_id: str):
    w = workers_by_id.pop(cam_id, None)
    if w:
        w.stop()
        try: w.join(timeout=2.0)
        except: pass
    latest_jpeg[cam_id] = None
    active_sources_by_id.pop(cam_id, None)

def start_worker(cam_id: str, src_index: int):
    w = CameraWorker(cam_id, src_index, ARGS.model)
    w.start()
    workers_by_id[cam_id] = w
    active_sources_by_id[cam_id] = src_index
    if cam_id not in latest_jpeg:
        latest_jpeg[cam_id] = None

@app.post("/activate_map")
def activate_map(body: ActivateMapBody):
    new_map: Dict[str, int] = {}
    for cam_id, label in body.map.items():
        if cam_id not in CAM_IDS:
            raise HTTPException(400, f"Unknown cam id: {cam_id}")
        idx = _parse_index_from_label(label)
        if idx is None:
            raise HTTPException(400, f"Could not parse index from '{label}'. Use a number or the 'Camera #X (index X)' label.")
        new_map[cam_id] = idx

    changed = []
    for cam_id in list(active_sources_by_id.keys()):
        if cam_id not in new_map:
            stop_worker(cam_id)
            changed.append(cam_id)

    for cam_id, src in new_map.items():
        have = active_sources_by_id.get(cam_id)
        if have != src:
            if have is not None:
                stop_worker(cam_id)
            start_worker(cam_id, src)
            changed.append(cam_id)

    return {"ok": True, "changed": changed, "active": list(active_sources_by_id.keys())}

@app.get("/cams")
def list_cams():
    return {"cams": list(active_sources_by_id.keys())}

@app.get("/video")
def video_feed(cam: str = Query(..., description="Camera id e.g. cameraA")):
    if cam not in active_sources_by_id:
        raise HTTPException(404, f"Camera '{cam}' is not active")
    async def gen():
        boundary = b"--frame\r\n"
        blank_wait = 0
        while True:
            buf = latest_jpeg.get(cam)
            if buf is None:
                blank_wait += 1
                if blank_wait % 250 == 0:
                    print(f"[video] waiting for frames from {cam}...")
                await asyncio.sleep(0.02); continue
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
            await asyncio.sleep(0.02)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/beep")
def beep():
    if os.path.exists(BEEP_FILE):
        return FileResponse(BEEP_FILE, media_type="audio/wav", filename="beep.wav")
    return {"error": "Beep file not found"}

@app.get("/events")
async def sse(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    subscribers.append(q)
    async def gen():
        try:
            while True:
                if await request.is_disconnected(): break
                data = await q.get()
                yield {"event": "message", "data": json.dumps(data)}
        finally:
            if q in subscribers: subscribers.remove(q)
    return EventSourceResponse(gen())

@app.get("/stats")
def stats():
    last = events_log[-1]["human_time"] if events_log else "-"
    return {"count": len(events_log), "last": last}

async def push_event(d: Dict):
    d = dict(d)
    d["human_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    events_log.append(d)
    for q in list(subscribers):
        try: await q.put(d)
        except: pass
    return JSONResponse({"ok": True})

def threadsafe_event(d: Dict):
    loop = app.state.loop
    if loop:
        asyncio.run_coroutine_threadsafe(push_event(d), loop)

# ---------------- Zones ----------------
def load_zones_for_cam(cam_id: str, frame_size: Tuple[int,int]) -> List[Dict]:
    w, h = frame_size
    primary = f"zones_{cam_id}.json"
    for candidate in (primary, ARGS.zones):
        if os.path.isfile(candidate):
            try:
                raw = json.load(open(candidate, "r"))
                out = []
                for i, z in enumerate(raw):
                    pts = clamp_pts(z["points"], w, h)
                    poly = fix_poly(Polygon(pts))
                    out.append({
                        "id": z.get("zone_id", f"Z{i+1}"),
                        "label": z.get("label", "RESTRICTED"),
                        "poly": poly,
                        "points": pts,
                    })
                if out:
                    print(f"[zones] loaded {len(out)} from {candidate} for {cam_id}")
                    return out
            except Exception as e:
                print(f"[zones] read error {candidate}: {e}")
    padw, padh = int(w*0.2), int(h*0.2)
    pts = [[padw,padh],[w-padw,padh],[w-padw,h-padh],[padw,h-padh]]
    return [{"id": "Z1", "label": "RESTRICTED", "poly": fix_poly(Polygon(pts)), "points": pts}]

# ---------------- Drawer (single cam) ----------------
def draw_zones_interactive(cam_id: str, cam_src):
    cap = cv2.VideoCapture(cam_src, cv2.CAP_DSHOW); ok, frame = cap.read()
    if not (cap.isOpened() and ok):
        print(f"[draw] Camera {cam_id} open failed"); return
    polys=[]; current=[]
    win = f"DRAW ZONES ({cam_id}): L-click add point | n=finish | u=undo | r=remove | s=save | q=quit"
    cv2.namedWindow(win)
    def on_mouse(e,x,y,flags,param):
        nonlocal current
        if e==cv2.EVENT_LBUTTONDOWN: current.append((x,y))
    cv2.setMouseCallback(win,on_mouse)
    while True:
        vis = frame.copy()
        for pl in polys:
            for i in range(len(pl)): cv2.line(vis, pl[i], pl[(i+1)%len(pl)], (255,0,0), 2)
        for i in range(len(current)):
            cv2.line(vis, current[i], current[(i+1)%len(current)], (0,255,0), 2)
            cv2.circle(vis, current[i], 3, (0,255,0), -1)
        cv2.putText(vis, "n=finish, u=undo, r=remove, s=save, q=quit", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,50), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k==ord('n') and len(current)>=3: polys.append(current.copy()); current=[]
        elif k==ord('u') and current: current.pop()
        elif k==ord('r') and polys: polys.pop()
        elif k==ord('s'):
            out=[]; idx=1
            if len(current)>=3: polys.append(current.copy()); current=[]
            for pl in polys:
                out.append({"zone_id":f"Z{idx}","label":"RESTRICTED","points":[list(p) for p in pl]}); idx+=1
            path = f"zones_{cam_id}.json"
            json.dump(out, open(path,"w"), indent=2)
            print(f"[draw] saved {len(out)} zone(s) to {path}")
            break
        elif k==ord('q') or k==27: break
    cap.release(); cv2.destroyAllWindows()

# ---------------- Camera Worker ----------------
class CameraWorker(threading.Thread):
    def __init__(self, cam_id: str, cam_index: int, model_path: str):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.cam_index = cam_index
        self.model_path = model_path
        self.stop_flag = False

        yunet, sface = ensure_models(ARGS.models_dir)
        self.face_stack = FaceStack(yunet, sface)

    def stop(self):
        self.stop_flag = True

    def _open_cap(self, index: int):
        """Try several backends. Return opened cap or None."""
        for backend in (0, cv2.CAP_DSHOW, cv2.CAP_MSMF):  # CAP_ANY first
            cap = cv2.VideoCapture(index, backend) if backend != 0 else cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue
            # warm-up for up to 1s
            ok = False
            t0 = time.time()
            while time.time() - t0 < 1.0:
                r, _ = cap.read()
                if r:
                    ok = True
                    break
                time.sleep(0.02)
            if ok:
                return cap
            cap.release()
        return None

    def run(self):
        global latest_jpeg

        # Probe once to get frame size
        probe = self._open_cap(self.cam_index)
        if probe is None:
            print(f"[{self.cam_id}] camera open/read failed (index {self.cam_index})"); return
        ok, fr = probe.read()
        if not ok or fr is None:
            print(f"[{self.cam_id}] camera read failed on probe"); probe.release(); return
        h, w = fr.shape[:2]; probe.release()

        zones = load_zones_for_cam(self.cam_id, (w, h))

        model_file = self.model_path if os.path.isfile(self.model_path) else "yolov8n.pt"
        if model_file != self.model_path:
            print(f"[{self.cam_id}] {self.model_path} not found; using {model_file}")
        model = YOLO(model_file)

        cap = self._open_cap(self.cam_index)
        if cap is None:
            print(f"[{self.cam_id}] cannot open cam again"); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        last_emit = 0.0
        fail_reads = 0

        while not self.stop_flag:
            ok, frame = cap.read()
            if not ok or frame is None:
                fail_reads += 1
                time.sleep(0.02)
                if fail_reads > 50:  # ~1s of failures → reopen
                    cap.release()
                    cap = self._open_cap(self.cam_index)
                    fail_reads = 0
                    if cap is None:
                        time.sleep(0.5)
                continue
            fail_reads = 0

            # person detection
            res = model.predict(frame, imgsz=ARGS.imgsz, conf=ARGS.conf, classes=[0], verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy() if len(res) else []

            vis = frame.copy()
            for z in zones:
                pts = z["points"]
                for i in range(len(pts)):
                    p1 = tuple(map(int, pts[i])); p2 = tuple(map(int, pts[(i+1)%len(pts)]))
                    cv2.line(vis, p1, p2, ZONE_COLOR, 2)

            triggered = False
            triggered_zone = None

            # set YuNet size for this frame
            H, W = frame.shape[:2]
            self.face_stack.set_size(W, H)

            for b in boxes:
                x1,y1,x2,y2 = map(int, b[:4])
                in_zone = False

                px = int((x1+x2)//2)
                py = int((y1+y2)//2)

                if ARGS.hit == "center":
                    p = Point((px,py))
                    for z in zones:
                        if z["poly"].contains(p):
                            in_zone = True; triggered_zone = z; break
                elif ARGS.hit == "feet":
                    p = Point((px,int(y2)))
                    for z in zones:
                        if z["poly"].contains(p):
                            in_zone = True; triggered_zone = z; break
                else:
                    bb = shapely_box(x1,y1,x2,y2)
                    for z in zones:
                        try:
                            inter_area = bb.intersection(z["poly"]).area
                        except Exception:
                            try:
                                inter_area = bb.buffer(0).intersection(z["poly"].buffer(0)).area
                            except Exception:
                                continue
                        ratio = inter_area / max(bb.area, 1.0)
                        if ratio >= ARGS.overlap_thresh:
                            in_zone = True; triggered_zone = z; break

                allowed_name, allowed_score = None, 0.0
                if in_zone and (len(ALLOWLIST_DB) > 0):
                    # run face detector inside the person box (expand slightly)
                    x1c = max(0, x1-10); y1c = max(0, y1-20)
                    x2c = min(W-1, x2+10); y2c = min(H-1, y2+10)
                    crop = frame[y1c:y2c, x1c:x2c]
                    if crop.size > 0:
                        self.face_stack.set_size(x2c - x1c, y2c - y1c)
                        dets = self.face_stack.detect(crop)
                        dets_full = []
                        for (fx1, fy1, fx2, fy2, sc) in dets:
                            dets_full.append((fx1+x1c, fy1+y1c, fx2+x1c, fy2+y1c, sc))
                        if dets_full:
                            dets_full.sort(key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
                            fbox = dets_full[0][:4]
                            fx1, fy1, fx2, fy2 = fbox
                            # Enforce a minimum face size to avoid noisy, tiny detections
                            if (fx2 - fx1) < ARGS.min_face or (fy2 - fy1) < ARGS.min_face:
                                emb = None
                            else:
                                emb = self.face_stack.embed(frame, fbox)

                            if emb is not None:
                                is_allowed, nm, sc = match_allowlist(emb, ARGS.allow_thresh)
                                if is_allowed:
                                    allowed_name, allowed_score = nm, sc
                                    _add_recent_permit(self.cam_id, px, py)
                        else:
                            if _recent_permit(self.cam_id, px, py):
                                allowed_name, allowed_score = "(recent)", 1.0

                if in_zone:
                    if allowed_name:
                        color = ALLOW_COLOR
                        band_y1 = max(0, y1 - 24)
                        cv2.rectangle(vis, (x1, band_y1), (x2, y1), ALLOW_COLOR, -1)
                        label = f"ALLOWED: {allowed_name}" + (f" ({allowed_score:.2f})" if allowed_name != "(recent)" else "")
                        cv2.putText(vis, label, (x1 + 6, max(12, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                    else:
                        color = ALERT_COLOR
                        triggered = True
                else:
                    color = SAFE_COLOR

                cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
                if ARGS.hit in ("feet","center"):
                    cv2.circle(vis,(px, py if ARGS.hit=="center" else y2),4,color,-1)

            if triggered:
                cv2.rectangle(vis, (0,0), (vis.shape[1], 48), ALERT_COLOR, -1)
                cv2.putText(vis, f"ALERT: {self.cam_id} PERSON IN RESTRICTED ZONE", (12,32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                now = time.time()
                if now - last_emit > 0.25:
                    evt = {"cam_id": self.cam_id,
                           "zone_id": triggered_zone["id"] if triggered_zone else "Z?",
                           "label": triggered_zone["label"] if triggered_zone else "RESTRICTED",
                           "timestamp": int(now),
                           "bbox": [-1,-1,-1,-1]}
                    threadsafe_event(evt)
                    last_emit = now

            ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                latest_jpeg[self.cam_id] = jpg.tobytes()

        cap.release()

# ---------------- Allowlist endpoints ----------------
@app.get("/allowlist")
def allowlist_info():
    with ALLOWLIST_LOCK:
        names = [n for n, _ in ALLOWLIST_DB]
    return {"count": len(names), "names": names, "threshold": ARGS.allow_thresh}

@app.post("/reload_allowlist")
def reload_allowlist():
    try:
        yunet, sface = ensure_models(ARGS.models_dir)
        face_stack = FaceStack(yunet, sface)
        db = load_allowlist_embeddings(ARGS.allowlist_dir, face_stack)
        with ALLOWLIST_LOCK:
            ALLOWLIST_DB.clear()
            ALLOWLIST_DB.extend(db)
        return {"ok": True, "count": len(ALLOWLIST_DB)}
    except Exception as e:
        raise HTTPException(500, f"reload failed: {e}")

# ---------------- Lifespan ----------------
@app.on_event("startup")
async def on_startup():
    app.state.loop = asyncio.get_event_loop()
    try:
        yunet, sface = ensure_models(ARGS.models_dir)
        face_stack = FaceStack(yunet, sface)
        if ARGS.allowlist_dir and os.path.isdir(ARGS.allowlist_dir):
            db = load_allowlist_embeddings(ARGS.allowlist_dir, face_stack)
            with ALLOWLIST_LOCK:
                ALLOWLIST_DB.clear()
                ALLOWLIST_DB.extend(db)
            print(f"[allow] loaded {len(ALLOWLIST_DB)} allowed face(s) from {ARGS.allowlist_dir}")
        else:
            print(f"[allow] directory not found or not set: {ARGS.allowlist_dir}")
    except Exception as e:
        print(f"[startup] model init/allowlist failed: {e}")
        print("[startup] continuing without allowlist.")

# ---------------- Main ----------------
if __name__ == "__main__":
    if ARGS.draw_zones:
        if not ARGS.cam_id or ARGS.cam_id not in CAM_IDS:
            print("When using --draw-zones, also provide --cam-id that exists in --cams.")
            print('Example: --draw-zones --cam-id cameraA --cams "cameraA=0"')
            raise SystemExit(1)
        draw_zones_interactive(ARGS.cam_id, 0)
        raise SystemExit(0)

    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")
