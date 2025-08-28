import os, cv2, time, json, argparse, threading, asyncio, platform, glob
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import uvicorn

from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from shapely.geometry import box as shapely_box
from shapely.validation import make_valid

# ---- face allowlist imports (ADDED) ----
import numpy as np
from insightface.app import FaceAnalysis

# ---- optional (Windows) webcam name enumeration ----
try:
    from pygrabber.dshow_graph import FilterGraph  # pip install pygrabber
except Exception:
    FilterGraph = None

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
    ap = argparse.ArgumentParser("Multi-camera backend (YOLO + zones) for React UI (with face allowlist)")
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

    # ---- Face allowlist (ADDED) ----
    ap.add_argument("--allowlist-dir", default=None,
                    help="Folder of allowed persons’ face images. Filenames become display names, e.g., 'Alice.jpg'.")
    ap.add_argument("--allow-thresh", type=float, default=0.35,
                    help="Cosine similarity threshold (0–1) for allowlist face match.")
    ap.add_argument("--allow-cache-sec", type=float, default=2.0,
                    help="Keep a short permit cache (seconds) after an allowed face is seen.")
    return ap.parse_args()

ARGS = parse_args()

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

# Logical camera IDs (slots) from --cams keys
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
active_sources_by_id: Dict[str, int] = {}  # cam_id -> DS index (extend to RTSP if needed)

# --- COLORS ---
ZONE_COLOR   = (255, 0, 0)   # BLUE (zone)
SAFE_COLOR   = (0, 255, 0)   # GREEN (person safe)
ALERT_COLOR  = (0, 0, 255)   # RED   (person in zone)
ALLOW_COLOR  = (255, 165, 0) # ORANGE (allowed in zone)

# --------- Allowlist (face embeddings) ----------
ALLOWLIST_DB: List[Tuple[str, np.ndarray]] = []  # [(name, embedding512)]
ALLOWLIST_LOCK = threading.Lock()

# Recent-permit cache (per-cam): [(ts, (cx,cy))]
RECENT_ALLOW: Dict[str, List[Tuple[float, Tuple[int,int]]]] = {}
RECENT_LOCK = threading.Lock()

# ---------------- Devices (Windows) ----------------
def _probe_opencv_indices(max_index: int = 10) -> List[dict]:
    """
    Fallback: try indices 0..max_index-1 with DSHOW/MSMF and return those that open.
    Produces names like 'Camera #0 (index 0)' if we cannot get a friendly name.
    """
    out = []
    for i in range(max_index):
        opened = False
        # Try DSHOW first (works well on many Windows setups), then MSMF, then default
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
            cap = cv2.VideoCapture(i, backend) if backend != 0 else cv2.VideoCapture(i)
            ok, _ = cap.read()
            if cap.isOpened() and ok:
                opened = True
                cap.release()
                break
            cap.release()
        if opened:
            out.append({"index": i, "name": f"Camera #{i} (index {i})"})
    return out

def list_local_cams() -> List[dict]:
    """
    Returns [{"index": int, "name": "..."}].
    - On Windows, try pygrabber for friendly names.
    - If pygrabber not available (or returns none), fall back to OpenCV probing.
    - On other OS, use OpenCV probing directly.
    """
    # Prefer pygrabber friendly names on Windows
    if platform.system() == "Windows" and FilterGraph is not None:
        try:
            g = FilterGraph()
            names = g.get_input_devices()  # list[str]
            devices = [{"index": i, "name": name} for i, name in enumerate(names)]
            if devices:
                return devices
        except Exception as e:
            print("[devices] pygrabber enumerate error:", e)

    # Fallback: probe indices with OpenCV (works on all OSes)
    devices = _probe_opencv_indices(max_index=10)
    return devices


# ---------------- Health ----------------
@app.get("/health")
def health():
    return {"ok": True, "slots": CAM_IDS, "active": list(active_sources_by_id.keys())}

# ---------------- API for React ----------------
@app.get("/cam_ids")
def cam_ids():
    """Logical slots (from --cams keys)."""
    return {"cam_ids": CAM_IDS}

@app.get("/devices")
def devices():
    """Physical camera devices (Windows)."""
    return {"devices": list_local_cams()}

class ActivateMapBody(BaseModel):
    # Example: {"cameraA": "Logitech HD Pro Webcam C920", "cameraB": "Integrated Camera"}
    map: Dict[str, str]

def stop_worker(cam_id: str):
    w = workers_by_id.pop(cam_id, None)
    if w:
        w.stop_flag = True
        try: w.join(timeout=2.0)
        except: pass
    latest_jpeg[cam_id] = None
    active_sources_by_id.pop(cam_id, None)

def start_worker(cam_id: str, src):
    w = CameraWorker(cam_id, src, ARGS.model)
    w.start()
    workers_by_id[cam_id] = w
    active_sources_by_id[cam_id] = src
    if cam_id not in latest_jpeg:
        latest_jpeg[cam_id] = None

@app.post("/activate_map")
def activate_map(body: ActivateMapBody):
    """
    Bind logical ids to device names; start/stop workers as needed.
    """
    devs = list_local_cams()
    name_to_index = {d["name"]: d["index"] for d in devs}

    # Build cam_id -> source mapping from names
    new_map: Dict[str, int] = {}
    for cam_id, dev_name in body.map.items():
        if cam_id not in CAM_IDS:
            raise HTTPException(400, f"Unknown cam id: {cam_id}")
        if dev_name not in name_to_index:
            raise HTTPException(400, f"Device not found: {dev_name}")
        new_map[cam_id] = int(name_to_index[dev_name])

    changed = []
    # Stop cams not in the new map
    for cam_id in list(active_sources_by_id.keys()):
        if cam_id not in new_map:
            stop_worker(cam_id)
            changed.append(cam_id)

    # Start/restart cams in the new map
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
    """Currently active cams (workers running)."""
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

# ---------------- Face allowlist helpers (UPDATED) ----------------
def _build_face_app():
    """
    Robust initializer that supports very old and newer insightface versions.
    - Tries positional 'name' (old API) and keyword (newer).
    - Does NOT pass 'providers' (since your build doesn't support it).
    - Forces a writable INSIGHTFACE_HOME so models can download.
    - Verifies a detection model actually exists; otherwise tries next pack.
    """
    packs = ["antelopev2", "antelope", "buffalo_l", "buffalo_m"]
    root = _insightface_home()
    last_err = None

    for pack in packs:
        try:
            # Try old-style positional name first (your error shows this is required)
            try:
                fa = FaceAnalysis(pack, root=root)
            except TypeError:
                # Fallback: some versions expect keywords
                fa = FaceAnalysis(name=pack, root=root)

            # Old API: prepare(ctx_id=..., det_size=..., det_thresh=?)
            # Do NOT pass providers (not supported in your build)
            try:
                fa.prepare(ctx_id=0, det_size=(256, 256))
            except TypeError:
                # Some very old builds use different arg order/names—retry minimal
                fa.prepare(ctx_id=0)

            # ---- verify detection actually loaded ----
            has_det = False
            if hasattr(fa, "models") and isinstance(getattr(fa, "models"), dict):
                has_det = bool(fa.models.get("detection"))
            if not has_det and hasattr(fa, "det_model"):
                has_det = fa.det_model is not None

            if has_det:
                print(f"[face] initialized using pack='{pack}', cache='{root}'")
                return fa
            else:
                print(f"[face] pack '{pack}' initialized but no detection model; trying next...")
        except Exception as e:
            last_err = e
            print(f"[face] init failed for pack '{pack}': {repr(e)}")

    print("[face] all packs failed; face allowlist disabled.")
    if last_err:
        print("[face] last error:", repr(last_err))
    return None



def _load_allowlist_embeddings(dirpath: str) -> List[Tuple[str, np.ndarray]]:
    fa = _build_face_app()
    if fa is None:
        return []

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files: List[str] = []
    for root, _, fns in os.walk(dirpath):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))

    out: List[Tuple[str, np.ndarray]] = []
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        faces = fa.get(img)
        if not faces:
            print(f"[allow] no face detected in {fp} (use a clear, front-facing photo)")
            continue
        else:
            print(f"[allow] {fp}: detected {len(faces)} face(s)")
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        emb = getattr(f, "normed_embedding", None) or getattr(f, "embedding", None)
        if emb is None:
            continue
        emb = emb.astype("float32")
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        name = os.path.splitext(os.path.basename(fp))[0]
        out.append((name, emb))
        print(f"[allow] added {name} from {os.path.basename(fp)}")
    return out

def _match_allowlist(emb: np.ndarray, thresh: float) -> Tuple[bool, Optional[str], float]:
    with ALLOWLIST_LOCK:
        db = list(ALLOWLIST_DB)
    if not db:
        return (False, None, 0.0)
    best_name, best_score = None, -1.0
    for name, ref in db:
        score = float(np.dot(emb, ref))  # cosine; embeddings normalized
        if score > best_score:
            best_score, best_name = score, name
    return (best_score >= thresh, best_name if best_score >= thresh else best_name, best_score)

def _extract_face_embedding(face_app: FaceAnalysis, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    """
    Top->bottom multi-region sweep over the full person bbox.
    Returns normalized embedding or None.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    H = y2 - y1
    regions = [
        (y1, y2),                                  # FULL
        (y1, y1 + int(0.70 * H)),                  # TOP 70%
        (y1 + int(0.20 * H), y1 + int(0.80 * H)),  # MIDDLE 60%
        (y1 + int(0.40 * H), y2),                  # BOTTOM 60%
    ]

    best_face = None
    best_area = -1
    for (sy, ey) in regions:
        crop = frame[sy:ey, x1:x2]
        if crop.size == 0:
            continue
        ch, cw = crop.shape[:2]
        scale = 640.0 / max(ch, cw) if max(ch, cw) > 640 else 1.0
        crop_in = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else crop
        faces = face_app.get(crop_in)
        if not faces:
            continue
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        area = (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
        if area > best_area:
            best_area = area
            best_face = f

    if best_face is None:
        return None
    emb = getattr(best_face, "normed_embedding", None) or getattr(best_face, "embedding", None)
    if emb is None:
        return None
    emb = emb.astype("float32")
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb

def _recent_permit(cam_id: str, cx: int, cy: int) -> bool:
    """Return True if a recent allowed bbox center is near (cx,cy)."""
    with RECENT_LOCK:
        arr = RECENT_ALLOW.get(cam_id, [])
    now = time.time()
    # purge old
    arr = [(ts, pt) for (ts, pt) in arr if now - ts <= ARGS.allow_cache_sec]
    with RECENT_LOCK:
        RECENT_ALLOW[cam_id] = arr
    # check proximity
    for _, (px, py) in arr:
        if (px - cx) ** 2 + (py - cy) ** 2 <= (50 ** 2):  # ~50px radius
            return True
    return False

def _add_recent_permit(cam_id: str, cx: int, cy: int):
    with RECENT_LOCK:
        arr = RECENT_ALLOW.get(cam_id, [])
        arr.append((time.time(), (cx, cy)))
        RECENT_ALLOW[cam_id] = arr

# ---------------- Management endpoints for allowlist (optional) ----------------
@app.get("/allowlist")
def allowlist_info():
    with ALLOWLIST_LOCK:
        names = [n for n, _ in ALLOWLIST_DB]
    return {"count": len(names), "names": names, "threshold": ARGS.allow_thresh}

@app.post("/reload_allowlist")
def reload_allowlist():
    if not ARGS.allowlist_dir or not os.path.isdir(ARGS.allowlist_dir):
        raise HTTPException(400, "allowlist-dir not set or not a directory")
    db = _load_allowlist_embeddings(ARGS.allowlist_dir)
    with ALLOWLIST_LOCK:
        ALLOWLIST_DB.clear()
        ALLOWLIST_DB.extend(db)
    return {"ok": True, "count": len(ALLOWLIST_DB)}

# ---------------- Camera Worker ----------------
class CameraWorker(threading.Thread):
    def __init__(self, cam_id: str, cam_src, model_path: str):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.cam_src = cam_src
        self.model_path = model_path
        self.stop_flag = False

    def run(self):
        global latest_jpeg

        def open_cap(src):
            for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, 0):
                cap = cv2.VideoCapture(src, backend) if backend != 0 else cv2.VideoCapture(src)
                ok, _ = cap.read()
                if cap.isOpened() and ok:
                    return cap
                cap.release()
            return None

        probe = open_cap(self.cam_src)
        if probe is None:
            print(f"[{self.cam_id}] camera open/read failed"); return
        ok, fr = probe.read()
        if not ok:
            print(f"[{self.cam_id}] camera read failed on probe"); probe.release(); return
        h, w = fr.shape[:2]; probe.release()

        zones = load_zones_for_cam(self.cam_id, (w, h))

        model_file = self.model_path if os.path.isfile(self.model_path) else "yolov8n.pt"
        if model_file != self.model_path:
            print(f"[{self.cam_id}] {self.model_path} not found; using {model_file}")
        model = YOLO(model_file)

        cap = open_cap(self.cam_src)
        if cap is None:
            print(f"[{self.cam_id}] cannot open cam again"); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # ---- Face module (init once) ----
        face_app = None
        if ARGS.allowlist_dir:
            face_app = _build_face_app()
            if face_app is None:
                print(f"[{self.cam_id}] face module not available; allowlist disabled")

        last_emit = 0.0
        while not self.stop_flag:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02); continue

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

            for b in boxes:
                x1,y1,x2,y2 = map(int, b[:4])
                in_zone = False

                # representative points
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

                # ---- Face allowlist check ----
                allowed_name, allowed_score = None, 0.0
                if in_zone:
                    # quick cache to avoid flicker if face is briefly missed
                    if _recent_permit(self.cam_id, px, py):
                        allowed_name, allowed_score = "(recent)", 1.0
                    elif face_app:
                        emb = _extract_face_embedding(face_app, frame, x1, y1, x2, y2)
                        if emb is not None:
                            is_allowed, nm, sc = _match_allowlist(emb, ARGS.allow_thresh)
                            if is_allowed:
                                allowed_name, allowed_score = nm, sc
                                _add_recent_permit(self.cam_id, px, py)

                # decide color/state
                if in_zone:
                    if allowed_name:  # allowed person in zone → NO alert
                        color = ALLOW_COLOR
                        # small band label for clarity
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

            # emit banner + event only when triggered and NOT allowed
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

# ---- InsightFace cache helper (ADD THIS) ----
def _insightface_home() -> str:
    """
    Returns a writable directory to cache/download InsightFace models.
    Uses INSIGHTFACE_HOME if set; otherwise creates ./.insightface next to this file.
    """
    home = os.environ.get("INSIGHTFACE_HOME")
    if not home:
        home = os.path.join(os.path.dirname(__file__), ".insightface")
    os.makedirs(home, exist_ok=True)
    return home
    
# ---------------- Lifespan ----------------
@app.on_event("startup")
async def on_startup():
    app.state.loop = asyncio.get_event_loop()
    if ARGS.allowlist_dir and os.path.isdir(ARGS.allowlist_dir):
        try:
            db = _load_allowlist_embeddings(ARGS.allowlist_dir)
            with ALLOWLIST_LOCK:
                ALLOWLIST_DB.clear()
                ALLOWLIST_DB.extend(db)
            print(f"[allow] loaded {len(ALLOWLIST_DB)} allowed face(s) from {ARGS.allowlist_dir}")
        except Exception as e:
            print(f"[allow] failed to load allowlist: {repr(e)}")
            print("[allow] continuing without face allowlist.")
    elif ARGS.allowlist_dir:
        print(f"[allow] directory not found: {ARGS.allowlist_dir}")


# ---------------- Main ----------------
if __name__ == "__main__":
    # Zone drawer for one cam index (utility) — here we still use a raw index
    if ARGS.draw_zones:
        if not ARGS.cam_id or ARGS.cam_id not in CAM_IDS:
            print("When using --draw-zones, also provide --cam-id that exists in --cams.")
            print('Example: --draw-zones --cam-id cameraA --cams "cameraA=0"')
            raise SystemExit(1)
        draw_zones_interactive(ARGS.cam_id, 0)
        raise SystemExit(0)

    # Start ONLY the web server; workers are started via POST /activate_map
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="info")
