import os, cv2, datetime, time
import numpy as np

CLASSES = [
    "apple","banana","carrot","cucumber","eggplant",
    "ginger","lemon","orange","peach","pear"
]
SAVE_DIR       = os.path.join("dataset", "WebcamTrain")
TARGET         = 90
CAPTURE_EVERY  = 1.5   # seconds between auto-captures
BLUR_THRESHOLD = 80    # below this Laplacian variance = blurry, skip

INSTRUCTIONS = [
    "Hold fruit CENTRE, facing front",
    "Tilt fruit slightly LEFT",
    "Tilt fruit slightly RIGHT",
    "Move fruit CLOSER to camera",
    "Move fruit FURTHER away",
    "Rotate fruit 45 degrees",
    "Rotate fruit 90 degrees",
    "Hold fruit in LEFT side of box",
    "Hold fruit in RIGHT side of box",
    "Change lighting — tilt or block lamp",
    "Rotate fruit upside down",
    "Hold fruit at an ANGLE",
    "Move to UPPER part of box",
    "Move to LOWER part of box",
    "Hold with BOTH hands — partial occlusion",
]

def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    WIN = "Webcam Data Collector  v2"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 580)

    class_idx    = 0
    counts       = {c: 0 for c in CLASSES}
    paused       = True   # start PAUSED so user can get ready
    last_capture = time.time()
    instr_idx    = 0
    skipped_blur = 0

    # Count existing images
    for c in CLASSES:
        d = os.path.join(SAVE_DIR, c)
        if os.path.exists(d):
            counts[c] = len([f for f in os.listdir(d) if f.endswith(".jpg")])

    print("\n[INFO] Starts PAUSED — press P to begin capturing.")
    print("[INFO] P=pause/resume   N=next class   Q=quit\n")

    while class_idx < len(CLASSES):
        cls = CLASSES[class_idx]

        # Skip classes already done
        if counts[cls] >= TARGET:
            print(f"  [{cls}] already has {counts[cls]} images. Skipping.")
            class_idx += 1
            paused = True   # pause so user can grab next fruit
            continue

        save_path = os.path.join(SAVE_DIR, cls)
        os.makedirs(save_path, exist_ok=True)

        ret, frame = cap.read()
        if not ret: break

        fh, fw   = frame.shape[:2]
        display  = frame.copy()
        done     = counts[cls]
        now      = time.time()
        blurry   = is_blurry(frame)

        # ── Auto-capture (only when not paused) ──────────────────────────────
        captured_this_frame = False
        if not paused and done < TARGET and (now - last_capture) >= CAPTURE_EVERY:
            if blurry:
                skipped_blur += 1
            else:
                ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = os.path.join(save_path, f"webcam_{ts}.jpg")
                cv2.imwrite(fname, frame)
                counts[cls] += 1
                done = counts[cls]
                instr_idx = (instr_idx + 1) % len(INSTRUCTIONS)
                captured_this_frame = True
                print(f"  [{cls}] {done}/{TARGET}")
            last_capture = now

        # ── UI ────────────────────────────────────────────────────────────────
        ov = display.copy()
        cv2.rectangle(ov, (0,0),(fw,85),(15,15,15),-1)
        cv2.addWeighted(ov, 0.72, display, 0.28, 0, display)

        # Class + count
        name_col = (80,200,80) if not captured_this_frame else (40,210,255)
        cv2.putText(display, f"Class: {cls.upper()}  ({done}/{TARGET})",
                    (14,30), cv2.FONT_HERSHEY_DUPLEX, 0.85, name_col, 1, cv2.LINE_AA)

        # Instruction (hidden when paused)
        if not paused:
            instr = INSTRUCTIONS[instr_idx % len(INSTRUCTIONS)]
            cv2.putText(display, f">> {instr}",
                        (14,54), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (40,200,255), 1, cv2.LINE_AA)

        # Controls hint
        cv2.putText(display, "[P] pause/resume   [N] next   [Q] quit",
                    (fw-310, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (130,130,130), 1, cv2.LINE_AA)

        # Progress bar
        bar_w = int((fw-40) * done / TARGET)
        cv2.rectangle(display,(20,72),(fw-20,80),(50,50,50),-1)
        cv2.rectangle(display,(20,72),(20+bar_w,80),(80,200,80),-1)

        # Guide box
        bx = int(fw*0.20); by = int(fh*0.12)
        bw = int(fw*0.60); bh = int(fh*0.75)
        box_col = (40,210,255) if captured_this_frame else (150,150,150)
        cv2.rectangle(display,(bx,by),(bx+bw,by+bh), box_col, 2)

        # ── PAUSED overlay ────────────────────────────────────────────────────
        if paused:
            ov2 = display.copy()
            cv2.rectangle(ov2,(0,0),(fw,fh),(15,15,15),-1)
            cv2.addWeighted(ov2, 0.45, display, 0.55, 0, display)

            # Big PAUSED text
            msg = "PAUSED"
            (mw,mh),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3)
            cv2.putText(display, msg,
                        ((fw-mw)//2, fh//2 - 20),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (40,210,255), 3, cv2.LINE_AA)

            # Sub-message
            sub = f"Get ready with: {cls.upper()}"
            (sw,_),_ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 1)
            cv2.putText(display, sub,
                        ((fw-sw)//2, fh//2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200,200,200), 1, cv2.LINE_AA)

            resume = "Press [P] to start capturing"
            (rw,_),_ = cv2.getTextSize(resume, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(display, resume,
                        ((fw-rw)//2, fh//2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130,130,130), 1, cv2.LINE_AA)

        else:
            # Countdown to next capture
            time_left = max(0, CAPTURE_EVERY - (now - last_capture))
            timer_txt = f"Next in {time_left:.1f}s"
            cv2.putText(display, timer_txt,
                        (bx+8, by+bh-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (130,130,130), 1, cv2.LINE_AA)

            # Blur warning
            if blurry:
                msg = "BLURRY — hold still"
                (mw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.putText(display, msg,
                            ((fw-mw)//2, fh//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40,40,220), 2, cv2.LINE_AA)

        # Done message
        if done >= TARGET:
            msg = "DONE — press N for next class"
            (mw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
            cv2.putText(display, msg,
                        ((fw-mw)//2, fh//2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (80,200,80), 2, cv2.LINE_AA)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'),ord('Q')):
            break
        elif key in (ord('p'),ord('P')):
            paused = not paused
            if paused:
                print(f"  [PAUSED]")
            else:
                last_capture = time.time()  # reset timer on resume
                print(f"  [RESUMED] capturing {cls}...")
        elif key in (ord('n'),ord('N')):
            print(f"  [{cls}] Saved {counts[cls]} images. Moving to next.")
            class_idx += 1
            paused    = True   # auto-pause so user can grab next fruit
            instr_idx = 0

    cap.release()
    cv2.destroyAllWindows()

    print("\n── Collection Summary ──────────────────")
    for c in CLASSES:
        print(f"  {c:<12}  {counts[c]:>3} images")
    print(f"\nImages saved to: {SAVE_DIR}")
    print(f"Blurry frames skipped: {skipped_blur}")
    print("Run finetune_webcam.py next.\n")

if __name__ == "__main__":
    main()