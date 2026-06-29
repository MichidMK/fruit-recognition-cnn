import os, sys, glob, csv, datetime, collections
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RESULTS  = "results"
SS_DIR   = os.path.join(RESULTS, "screenshots")
LOG_PATH = os.path.join(RESULTS, "prediction_log.csv")
IMG_SIZE = 224

CLASS_NAMES = [
    "apple","banana","carrot","cucumber","eggplant",
    "ginger","lemon","orange","peach","pear"
]

C_GREEN  = ( 80, 200,  80)
C_ORANGE = ( 40, 165, 255)
C_RED    = ( 60,  60, 220)
C_WHITE  = (240, 240, 240)
C_GREY   = (120, 120, 120)
C_DARK   = ( 18,  18,  18)
C_YELLOW = ( 40, 210, 210)

CONF_THRESHOLD = 0.62
SMOOTH_WINDOW  = 10
PREDICT_EVERY  = 3
FACE_EVERY     = 6

#  YOLO loader
def try_load_yolo():
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")   # downloads automatically on first run
        print("[INFO] YOLOv8 nano loaded — using real object detection.")
        return yolo
    except ImportError:
        print("[WARN] ultralytics not installed. Using MOG2 fallback.")
        print("       Run: pip install ultralytics --break-system-packages")
        return None
    except Exception as e:
        print(f"[WARN] YOLO failed to load ({e}). Using MOG2 fallback.")
        return None

#  Classifier model 
class FruitClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path or self._find_best()
        self._load()

    def _find_best(self):
        # Priority: TFLite > finetuned keras > efficientnet > mobilenet > baseline
        tflite = os.path.join(RESULTS, "mobilenet_quantized.tflite")
        if os.path.exists(tflite):
            return tflite
        for pat in [
            "mobilenet_finetuned_*.keras",
            "efficientnet_*.keras",
            "mobilenet_2*.keras",
            "baseline_cnn_*.keras"
        ]:
            m = sorted(glob.glob(os.path.join(RESULTS, pat)), reverse=True)
            if m: return m[0]
        print("[ERROR] No model found in results/"); sys.exit(1)

    def _load(self):
        import tensorflow as tf
        path = self.model_path
        print(f"\n[INFO] Loading: {os.path.basename(path)}")
        if path.endswith(".tflite"):
            self.interp = tf.lite.Interpreter(model_path=path)
            self.interp.allocate_tensors()
            self.inp_det = self.interp.get_input_details()
            self.out_det = self.interp.get_output_details()
            self.mode    = "tflite"
        else:
            self.keras_model = tf.keras.models.load_model(path)
            # detect preprocessing
            name = os.path.basename(path).lower()
            if "efficientnet" in name:
                from tensorflow.keras.applications.efficientnet import preprocess_input
                self.arch = "EfficientNetB0"
            else:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                self.arch = "MobileNetV2"
            self.preprocess = preprocess_input
            self.mode = "keras"
        arch_name = getattr(self, 'arch', self.mode.upper())
        print(f"[INFO] Model ready — {arch_name} ({self.mode})")

    def predict(self, bgr_roi):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pi
        rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
        rsz = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        if self.mode == "tflite":
            inp = pi(np.expand_dims(rsz, 0))
            self.interp.set_tensor(self.inp_det[0]['index'], inp)
            self.interp.invoke()
            return self.interp.get_tensor(self.out_det[0]['index'])[0]
        else:
            inp = self.preprocess(np.expand_dims(rsz, 0))
            return self.keras_model.predict(inp, verbose=0)[0]

    @property
    def name(self):
        n = os.path.basename(self.model_path)
        if "tflite" in n:      return "TFLite"
        if "finetuned" in n:   return "MNv2-FT"
        if "efficientnet" in n: return "EffNetB0"
        if "mobilenet" in n:   return "MNv2"
        return "CNN"

#  MOG2 fallback localiser
class MOG2Localiser:
    def __init__(self):
        self.mog  = cv2.createBackgroundSubtractorMOG2(
                        history=200, varThreshold=60, detectShadows=False)
        self.box  = None
        self.kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))

    def update(self, frame, faces=None):
        fh, fw = frame.shape[:2]
        mask   = self.mog.apply(frame)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kern, iterations=3)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kern, iterations=2)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid  = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not (750 < area < 0.35*fh*fw):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / (bw + 1e-5)
            if aspect > 2.0 or aspect < 0.45:
                continue
            if faces:
                skip = False
                for (hx, hy, hw, hh) in faces:
                    if bx < hx+hw and bx+bw > hx and by < hy+hh and by+bh > hy:
                        skip = True
                        break
                if skip:
                    continue
            valid.append(c)
        if not valid:
            self.box = None; return None
        if self.box is not None:
            ox, oy, ow, oh = self.box
            prev_cx = ox + ow // 2
            prev_cy = oy + oh // 2
            def dist_score(c):
                bx,by,bw,bh = cv2.boundingRect(c)
                cx = bx + bw // 2
                cy = by + bh // 2
                return (cx - prev_cx)**2 + (cy - prev_cy)**2
            best = min(valid, key=dist_score)
        else:
            best = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        if self.box is not None:
            ox, oy, ow, oh = self.box
            dist = ((x+w//2 - ox-ow//2)**2 + (y+h//2 - oy-oh//2)**2) ** 0.5
            if dist > 300:
                return self.box
        pad = 20
        x=max(0,x-pad); y=max(0,y-pad)
        w=min(fw,x+w+pad*2)-x; h=min(fh,y+h+pad*2)-y
        nb = (x,y,w,h)
        if self.box is None:
            self.box = nb
        else:
            ox,oy,ow,oh = self.box
            nx,ny,nw,nh = nb
            a = 0.25
            self.box=(int(ox+a*(nx-ox)),int(oy+a*(ny-oy)),
                      int(ow+a*(nw-ow)),int(oh+a*(nh-oh)))
        return self.box

    def reset(self):
        self.mog = cv2.createBackgroundSubtractorMOG2(
                       history=200, varThreshold=60, detectShadows=False)
        self.box = None

#  YOLO localiser 
class YOLOLocaliser:
    def __init__(self, yolo):
        self.yolo = yolo
        self.box  = None

    def update(self, frame):
        results = self.yolo(frame, verbose=False, conf=0.25)[0]
        fh, fw  = frame.shape[:2]
        best    = None
        best_conf = 0
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = box
        if best is None:
            self.box = None; return None
        x1,y1,x2,y2 = map(int, best.xyxy[0])
        pad = 12
        x1=max(0,x1-pad); y1=max(0,y1-pad)
        x2=min(fw,x2+pad); y2=min(fh,y2+pad)
        nb = (x1, y1, x2-x1, y2-y1)
        if self.box is None:
            self.box = nb
        else:
            ox,oy,ow,oh = self.box
            nx,ny,nw,nh = nb
            a=0.35
            self.box=(int(ox+a*(nx-ox)),int(oy+a*(ny-oy)),
                      int(ow+a*(nw-ow)),int(oh+a*(nh-oh)))
        return self.box

    def reset(self):
        self.box = None

#  Utilities
def smooth(history):
    if not history: return "---", 0.0
    avg = np.mean(history, axis=0)
    i   = int(np.argmax(avg))
    return CLASS_NAMES[i], float(avg[i])

def init_log():
    os.makedirs(RESULTS, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH,"w",newline="") as f:
            csv.writer(f).writerow(["timestamp","label","confidence","faces","model"])

def write_log(label, conf, faces, model_name):
    with open(LOG_PATH,"a",newline="") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label, f"{conf:.3f}", faces, model_name
        ])

def save_ss(frame, label):
    os.makedirs(SS_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SS_DIR, f"{label.replace(' ','_')}_{ts}.png")
    cv2.imwrite(path, frame)
    print(f"  [Screenshot] {path}")

#  Drawing
def _alpha_rect(f,x1,y1,x2,y2,col,a=0.55):
    ov=f.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),col,-1)
    cv2.addWeighted(ov,a,f,1-a,0,f)

def draw_fruit_box(frame, box, label, conf):
    if box is None: return
    x,y,w,h=box; x2,y2=x+w,y+h
    cv2.rectangle(frame,(x-2,y-2),(x2+2,y2+2),(30,30,30),2)
    cv2.rectangle(frame,(x,y),(x2,y2),C_GREEN,2)
    CL,CT=18,3
    for cx,cy,dx,dy in [(x,y,1,1),(x2,y,-1,1),(x,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(cx,cy),(cx+dx*CL,cy),C_GREEN,CT)
        cv2.line(frame,(cx,cy),(cx,cy+dy*CL),C_GREEN,CT)
    tag=f"{label}  {conf*100:.0f}%"
    (tw,th),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
    tx=x; ty=y-8 if y>22 else y2+20
    _alpha_rect(frame,tx-3,ty-th-4,tx+tw+5,ty+4,C_DARK,0.75)
    cv2.putText(frame,tag,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.52,C_GREEN,1,cv2.LINE_AA)

def draw_face_boxes(frame, faces):
    for (fx,fy,fw,fh) in faces:
        cv2.rectangle(frame,(fx-2,fy-2),(fx+fw+2,fy+fh+2),(20,20,20),2)
        cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),C_YELLOW,2)
        CL,CT=14,2
        for cx,cy,dx,dy in [(fx,fy,1,1),(fx+fw,fy,-1,1),(fx,fy+fh,1,-1),(fx+fw,fy+fh,-1,-1)]:
            cv2.line(frame,(cx,cy),(cx+dx*CL,cy),C_YELLOW,CT)
            cv2.line(frame,(cx,cy),(cx,cy+dy*CL),C_YELLOW,CT)
        (tw,th),_=cv2.getTextSize("Human",cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
        _alpha_rect(frame,fx-1,fy-th-8,fx+tw+5,fy-1,C_DARK,0.75)
        cv2.putText(frame,"Human",(fx+2,fy-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,C_YELLOW,1,cv2.LINE_AA)

def draw_top_bar(frame, label, conf, state, faces, model_name, detector_name):
    h,w=frame.shape[:2]
    _alpha_rect(frame,0,0,w,52,C_DARK,0.72)
    lc = C_GREEN if state=="detected" else C_GREY
    cv2.putText(frame,f"Fruit: {label.upper()}",
                (14,33),cv2.FONT_HERSHEY_DUPLEX,0.72,lc,1,cv2.LINE_AA)
    conf_txt=f"Confidence: {conf*100:.1f}%"
    (cw,_),_=cv2.getTextSize(conf_txt,cv2.FONT_HERSHEY_DUPLEX,0.62,1)
    cv2.putText(frame,conf_txt,(w//2-cw//2,33),
                cv2.FONT_HERSHEY_DUPLEX,0.62,C_WHITE,1,cv2.LINE_AA)
    # Right info
    info1 = f"Faces: {len(faces)}  |  {model_name}"
    info2 = f"Detector: {detector_name}"
    (iw,_),_=cv2.getTextSize(info1,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
    cv2.putText(frame,info1,(w-iw-12,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.42,C_ORANGE,1,cv2.LINE_AA)
    (iw2,_),_=cv2.getTextSize(info2,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
    cv2.putText(frame,info2,(w-iw2-12,38),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,C_GREY,1,cv2.LINE_AA)
    cv2.line(frame,(0,52),(w,52),C_GREY,1)

def draw_bottom_bar(frame, frozen):
    h,w=frame.shape[:2]
    _alpha_rect(frame,0,h-30,w,h,C_DARK,0.72)
    cv2.line(frame,(0,h-30),(w,h-30),C_GREY,1)
    hint="S=screenshot   Space=freeze   F=fullscreen   M=switch model   Q=quit"
    if frozen: hint="[FROZEN]  "+hint
    (hw,_),_=cv2.getTextSize(hint,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
    cv2.putText(frame,hint,((w-hw)//2,h-9),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,C_GREY,1,cv2.LINE_AA)

def draw_scan_hint(frame):
    h,w=frame.shape[:2]
    msg="Point camera at a fruit"
    (mw,mh),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.60,1)
    mx=(w-mw)//2; my=h//2
    _alpha_rect(frame,mx-10,my-mh-6,mx+mw+10,my+8,C_DARK,0.55)
    cv2.putText(frame,msg,(mx,my),cv2.FONT_HERSHEY_SIMPLEX,0.60,C_GREY,1,cv2.LINE_AA)

#  Main 
def main():
    # Load all available classifiers
    all_models = []
    for pat in ["mobilenet_quantized.tflite",
                "mobilenet_finetuned_*.keras",
                "efficientnet_*.keras",
                "mobilenet_2*.keras"]:
        matches = sorted(glob.glob(os.path.join(RESULTS, pat)), reverse=True)
        for m in matches:
            if m not in all_models:
                all_models.append(m)
                break   # only newest of each type

    if not all_models:
        print("[ERROR] No models found in results/"); sys.exit(1)

    model_idx  = 0
    classifier = FruitClassifier(all_models[model_idx])

    # Load YOLO or fallback
    yolo = try_load_yolo()
    if yolo:
        localiser     = YOLOLocaliser(yolo)
        detector_name = "YOLOv8n"
    else:
        localiser     = MOG2Localiser()
        detector_name = "MOG2"

    # Face detector
    xml      = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_det = cv2.CascadeClassifier(xml)
    if face_det.empty(): face_det = None

    init_log()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    WIN = "Fruit Recognition  |  v5"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 580)
    fullscreen = False

    print(f"[INFO] Detector: {detector_name}")
    print(f"[INFO] Models available: {len(all_models)}")
    print("[INFO] Q=quit  S=screenshot  F=fullscreen  Space=freeze  M=switch model\n")

    pred_history = collections.deque(maxlen=SMOOTH_WINDOW)
    frozen       = False
    frozen_frame = None
    frozen_box   = None
    frozen_faces = []
    frame_count  = 0
    cur_label    = "---"
    cur_conf     = 0.0
    faces        = []

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret: print("[ERROR] Webcam lost."); break
            frame_count += 1
        else:
            frame = frozen_frame.copy()

        # Face detection
        if not frozen and face_det is not None and frame_count % FACE_EVERY == 0:
            small = cv2.resize(frame, (640, 360))   # half of 1280×720
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            raw   = face_det.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))
            faces = [(x*2, y*2, w*2, h*2) for (x,y,w,h) in raw] if len(raw) > 0 else []
        elif frozen:
            faces = frozen_faces

        # Object localisation
        det_box = localiser.update(frame, faces) if not frozen else frozen_box

        # Fruit classification
        if not frozen and frame_count % PREDICT_EVERY == 0:
            roi = (frame[det_box[1]:det_box[1]+det_box[3],
                         det_box[0]:det_box[0]+det_box[2]]
                   if det_box else frame)
            if roi.size > 0:
                preds = classifier.predict(roi)
                pred_history.append(preds)

        # Smooth
        raw_label, raw_conf = smooth(pred_history)
        if raw_conf >= CONF_THRESHOLD and det_box is not None:
            cur_label = raw_label; cur_conf = raw_conf
            state     = "frozen" if frozen else "detected"
            show_box  = True
        else:
            cur_label = "No fruit"; cur_conf = raw_conf
            state     = "frozen" if frozen else "uncertain"
            show_box  = False

        # Draw
        display = frame.copy()
        if show_box:
            draw_fruit_box(display, det_box, cur_label, cur_conf)
        else:
            draw_scan_hint(display)
        if faces:
            draw_face_boxes(display, faces)
        draw_top_bar(display, cur_label, cur_conf, state,
                     faces, classifier.name, detector_name)
        draw_bottom_bar(display, frozen)
        if frozen:
            h,w=display.shape[:2]
            cv2.putText(display,"FROZEN",(w//2-38,90),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(100,215,255),1,cv2.LINE_AA)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'),ord('Q'),27):
            print("[INFO] Quit."); break
        elif key in (ord('s'),ord('S')):
            save_ss(display, cur_label)
        elif key == ord(' '):
            frozen = not frozen
            if frozen:
                frozen_frame = frame.copy()
                frozen_box   = det_box
                frozen_faces = list(faces)
                print(f"  [Frozen] {cur_label}  {cur_conf*100:.1f}%  faces={len(faces)}")
                write_log(cur_label, cur_conf, len(faces), classifier.name)
            else:
                frozen_box=None; frozen_faces=[]
                print("  [Live]")
        elif key in (ord('f'),ord('F')):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, prop)
            if not fullscreen: cv2.resizeWindow(WIN, 960, 580)
        elif key in (ord('m'),ord('M')):
            # Cycle to next available model
            model_idx  = (model_idx + 1) % len(all_models)
            classifier = FruitClassifier(all_models[model_idx])
            pred_history.clear()
            print(f"  [Model] switched to: {classifier.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Log saved -> {LOG_PATH}")

if __name__ == "__main__":
    main()