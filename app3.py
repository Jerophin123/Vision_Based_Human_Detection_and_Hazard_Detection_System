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
                    help='Comma list of logical ids: "cameraA=0,cameraB=1".')
    ap.add_argument("--zones", default="zones.json", help="Default zones file (fallback)")
    ap.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--draw-zones", action="store_true",
                    help="Open polygon drawer for a single cam (use --cam-id).")
    ap.add_argument("--cam-id", default=None)
    ap.add_argument("--hit", choices=["center","overlap","feet"], default="overlap")
    ap.add_argument("--overlap-thresh", type=float, default=0.05)

    # ---- Face allowlist (OpenCV YuNet + SFace) ----
    ap.add_argument("--allowlist-dir", default="./allowed_faces",
                    help="Folder of allowed persons’ face images.")
    ap.add_argument("--allow-thresh", type=float, default=0.60,
                    help="Cosine similarity threshold (0–1).")
    ap.add_argument("--allow-cache-sec", type=float, default=2.0)
    ap.add_argument("--models-dir", default="./models")
    ap.add_argument("--autodownload-models", action="store_true")

    # ---- NEW strictness/quality controls ----
    ap.add_argument("--strict", action="store_true",
                    help="Require best score to beat second-best by a margin.")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="Required (best - second_best) margin when --strict is set.")
    ap.add_argument("--min-face", type=int, default=80,
                    help="Reject faces smaller than this in pixels.")

    return ap.parse_args()

ARGS = parse_args()

# ---------------- Parse cameras ----------------
def parse_cams(spec: str) -> Dict[str, str]:
    out = {}
    for token in spec.split(","):
        token = token.strip()
        if not token: continue
        if "=" not in token: raise ValueError(f'Bad --cams token "{token}"')
        cid, _src = token.split("=", 1)
        out[cid.strip()] = True
    if not out: raise ValueError("No cameras parsed from --cams.")
    return out

CAM_IDS: List[str] = list(parse_cams(ARGS.cams).keys())

# ---------------- Helpers ----------------
def clamp_pts(pts, w, h):
    return [[max(0, min(w-1, int(x))), max(0, min(h-1, int(y)))] for x, y in pts]

def fix_poly(poly):
    try:
        if not poly.is_valid: poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon": poly = max(poly.geoms, key=lambda g: g.area)
        poly = poly.buffer(0)
        if poly.is_empty: raise ValueError("empty after fix")
    except Exception:
        poly = poly.convex_hull
    return poly

# ---------------- Globals ----------------
app.state.loop = None
latest_jpeg: Dict[str, Optional[bytes]] = {}
events_log: List[dict] = []
subscribers: List[asyncio.Queue] = []

workers_by_id: Dict[str, "CameraWorker"] = {}
active_sources_by_id: Dict[str, int] = {}

ZONE_COLOR   = (255, 0, 0)
SAFE_COLOR   = (0, 255, 0)
ALERT_COLOR  = (0, 0, 255)
ALLOW_COLOR  = (255, 165, 0)

ALLOWLIST_DB: List[Tuple[str, np.ndarray]] = []  # multiple embeddings per person
ALLOWLIST_LOCK = threading.Lock()

RECENT_ALLOW: Dict[str, List[Tuple[float, Tuple[int,int]]]] = {}
RECENT_LOCK = threading.Lock()

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

def ensure_models(models_dir: str) -> Tuple[str, str]:
    os.makedirs(models_dir, exist_ok=True)
    yunet = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    sface = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
    for name, path, url in [("YuNet", yunet, YUNET_URL), ("SFace", sface, SFACE_URL)]:
        if not os.path.isfile(path):
            if ARGS.autodownload_models:
                print(f"[models] downloading {name}...")
                urllib.request.urlretrieve(url, path)
            else:
                raise RuntimeError(f"{name} ONNX not found at {path}")
    return yunet, sface

# ---------------- Face Stack ----------------
class FaceStack:
    def __init__(self, yunet_path: str, sface_path: str):
        self.detector = cv2.FaceDetectorYN_create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF_create(sface_path, "")
    def set_size(self, w: int, h: int): self.detector.setInputSize((w, h))
    def detect(self, bgr: np.ndarray):
        try: faces = self.detector.detect(bgr)
        except cv2.error: faces = None
        if faces is None: return []
        dets = faces[1] if isinstance(faces, tuple) else faces
        out = []
        if dets is None: return out
        for det in dets:
            x, y, w, h = det[:4]; score = det[-1] if det.shape[0] >= 15 else 1.0
            out.append((int(x), int(y), int(x+w), int(y+h), float(score)))
        return out
    def embed(self, bgr: np.ndarray, box: Tuple[int,int,int,int]):
        x1,y1,x2,y2 = box
        face_box = np.array([x1,y1,x2-x1,y2-y1], dtype=np.int32)
        try: aligned = self.recognizer.alignCrop(bgr, face_box); feat = self.recognizer.feature(aligned)
        except cv2.error: return None
        if feat is None: return None
        feat = feat.astype("float32").ravel()
        return feat / (np.linalg.norm(feat)+1e-9)

# ---------------- Allowlist utils ----------------
def cosine_sim(a,b):
    a=np.asarray(a,"float32").ravel(); b=np.asarray(b,"float32").ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

def load_allowlist_embeddings(dirpath: str, fs: FaceStack):
    if not os.path.isdir(dirpath): return []
    exts = (".jpg",".jpeg",".png",".bmp",".webp")
    out=[]
    for root,_,fns in os.walk(dirpath):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() not in exts: continue
            img=cv2.imread(os.path.join(root,fn)); 
            if img is None: continue
            h,w=img.shape[:2]; fs.set_size(w,h)
            dets=fs.detect(img)
            if not dets: continue
            dets.sort(key=lambda d:(d[2]-d[0])*(d[3]-d[1]),reverse=True)
            box=dets[0][:4]; emb=fs.embed(img,box)
            if emb is not None:
                name=os.path.basename(root) if root!=dirpath else os.path.splitext(fn)[0]
                out.append((name,emb))
    return out

def match_allowlist(emb: np.ndarray, thresh: float):
    with ALLOWLIST_LOCK: pairs=list(ALLOWLIST_DB)
    if not pairs: return (False,None,0.0,0.0)
    per_name={}
    for name,ref in pairs:
        s=cosine_sim(emb,ref)
        if name not in per_name or s>per_name[name]: per_name[name]=s
    ranked=sorted(per_name.items(),key=lambda kv:kv[1],reverse=True)
    best_name,best_score=ranked[0]; second=ranked[1][1] if len(ranked)>1 else -1.0
    gap=best_score-second
    accept=best_score>=thresh
    if ARGS.strict: accept=accept and (gap>=ARGS.margin)
    return (accept,best_name if accept else None,float(best_score),float(gap))

# ---------------- Camera Worker ----------------
class CameraWorker(threading.Thread):
    def __init__(self, cam_id, cam_index, model_path):
        super().__init__(daemon=True); self.cam_id=cam_id; self.cam_index=cam_index
        yunet,sface=ensure_models(ARGS.models_dir); self.face_stack=FaceStack(yunet,sface)
        self.model=YOLO(model_path if os.path.isfile(model_path) else "yolov8n.pt")
        self.stop_flag=False
    def stop(self): self.stop_flag=True
    def _open_cap(self,idx):
        for backend in (0, cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap=cv2.VideoCapture(idx,backend) if backend!=0 else cv2.VideoCapture(idx)
            if not cap.isOpened(): cap.release(); continue
            ok,_=cap.read(); 
            if ok: return cap
            cap.release()
        return None
    def run(self):
        global latest_jpeg
        cap=self._open_cap(self.cam_index)
        if cap is None: return
        ok,fr=cap.read(); 
        if not ok: return
        h,w=fr.shape[:2]; zones=load_zones_for_cam(self.cam_id,(w,h))
        while not self.stop_flag:
            ok,frame=cap.read(); 
            if not ok: continue
            res=self.model.predict(frame,imgsz=ARGS.imgsz,conf=ARGS.conf,classes=[0],verbose=False)
            boxes=res[0].boxes.xyxy.cpu().numpy() if len(res) else []
            vis=frame.copy()
            for z in zones:
                pts=z["points"]
                for i in range(len(pts)): cv2.line(vis,tuple(pts[i]),tuple(pts[(i+1)%len(pts)]),ZONE_COLOR,2)
            # detect faces once
            H,W=frame.shape[:2]; self.face_stack.set_size(W,H)
            all_faces=[f for f in self.face_stack.detect(frame) if (f[2]-f[0])>=ARGS.min_face and (f[3]-f[1])>=ARGS.min_face]
            def box_iou(a,b):
                ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
                inter_x1=max(ax1,bx1); inter_y1=max(ay1,by1)
                inter_x2=min(ax2,bx2); inter_y2=min(ay2,by2)
                iw=max(0,inter_x2-inter_x1); ih=max(0,inter_y2-inter_y1)
                inter=iw*ih; area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
                return inter/max(area_a+area_b-inter,1)
            for b in boxes:
                x1,y1,x2,y2=map(int,b[:4]); px=(x1+x2)//2; py=(y1+y2)//2
                in_zone=False
                if ARGS.hit=="center":
                    if any(z["poly"].contains(Point((px,py))) for z in zones): in_zone=True
                elif ARGS.hit=="feet":
                    if any(z["poly"].contains(Point((px,y2))) for z in zones): in_zone=True
                else:
                    bb=shapely_box(x1,y1,x2,y2)
                    for z in zones:
                        if bb.intersection(z["poly"]).area/max(bb.area,1)>ARGS.overlap_thresh: in_zone=True; break
                allowed_name=None;allowed_score=0;allowed_gap=0
                if in_zone and ALLOWLIST_DB:
                    best_f=None;best_iou=0
                    for (fx1,fy1,fx2,fy2,_) in all_faces:
                        iou=box_iou((x1,y1,x2,y2),(fx1,fy1,fx2,fy2))
                        if iou>best_iou: best_iou=iou; best_f=(fx1,fy1,fx2,fy2)
                    if best_f:
                        emb=self.face_stack.embed(frame,best_f)
                        if emb is not None:
                            ok_match,nm,sc,gap=match_allowlist(emb,ARGS.allow_thresh)
                            if ok_match: allowed_name,allowed_score,allowed_gap=nm,sc,gap
                color=SAFE_COLOR
                if in_zone:
                    if allowed_name:
                        color=ALLOW_COLOR
                        cv2.putText(vis,f"ALLOWED:{allowed_name} {allowed_score:.2f}/{allowed_gap:.2f}",(x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                    else:
                        color=ALERT_COLOR
                        cv2.putText(vis,"ALERT",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                cv2.rectangle(vis,(x1,y1),(x2,y2),color,2)
            ok,jpg=cv2.imencode(".jpg",vis,[int(cv2.IMWRITE_JPEG_QUALITY),80])
            if ok: latest_jpeg[self.cam_id]=jpg.tobytes()
        cap.release()

# ---------------- Zones loader ----------------
def load_zones_for_cam(cam_id, frame_size):
    w,h=frame_size
    primary=f"zones_{cam_id}.json"
    for candidate in (primary,ARGS.zones):
        if os.path.isfile(candidate):
            try:
                raw=json.load(open(candidate))
                out=[]
                for i,z in enumerate(raw):
                    pts=clamp_pts(z["points"],w,h)
                    poly=fix_poly(Polygon(pts))
                    out.append({"id":z.get("zone_id",f"Z{i+1}"),"label":z.get("label","RESTRICTED"),"poly":poly,"points":pts})
                return out
            except: pass
    return [{"id":"Z1","label":"RESTRICTED","poly":fix_poly(Polygon([[0,0],[w,0],[w,h],[0,h]])),"points":[[0,0],[w,0],[w,h],[0,h]]}]

# ---------------- API endpoints ----------------
@app.get("/allowlist")
def allowlist_info():
    with ALLOWLIST_LOCK: names=[n for n,_ in ALLOWLIST_DB]
    return {"count":len(names),"names":names,"threshold":ARGS.allow_thresh}

@app.post("/reload_allowlist")
def reload_allowlist():
    yunet,sface=ensure_models(ARGS.models_dir)
    fs=FaceStack(yunet,sface)
    db=load_allowlist_embeddings(ARGS.allowlist_dir,fs)
    with ALLOWLIST_LOCK: ALLOWLIST_DB.clear(); ALLOWLIST_DB.extend(db)
    return {"ok":True,"count":len(ALLOWLIST_DB)}

@app.on_event("startup")
async def on_startup():
    app.state.loop=asyncio.get_event_loop()
    try:
        yunet,sface=ensure_models(ARGS.models_dir)
        fs=FaceStack(yunet,sface)
        if os.path.isdir(ARGS.allowlist_dir):
            db=load_allowlist_embeddings(ARGS.allowlist_dir,fs)
            with ALLOWLIST_LOCK: ALLOWLIST_DB.clear(); ALLOWLIST_DB.extend(db)
            print(f"[allow] loaded {len(ALLOWLIST_DB)} templates")
    except Exception as e: print(f"[startup] failed {e}")

# ---------------- Main ----------------
if __name__=="__main__":
    uvicorn.run(app,host=ARGS.host,port=ARGS.port,log_level="info")
