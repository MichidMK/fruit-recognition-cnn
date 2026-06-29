import os
import cv2
import datetime

SAVE_DIR = os.path.join("dataset", "Train", "background")
TARGET = 300  # images to collect


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
    print(f"\n[INFO] Existing background images: {count}")
    print("[INFO] Point camera at EMPTY background (table, hand, room, etc.)")
    print("[INFO] Controls: SPACE=capture  Q=quit\n")

    while count < TARGET:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        fh, fw = display.shape[:2]

        # Overlay
        ov = display.copy()
        cv2.rectangle(ov, (0, 0), (fw, 70), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.7, display, 0.3, 0, display)

        pct = int(count / TARGET * 100)
        bar_w = int((fw - 40) * count / TARGET)

        cv2.putText(display, f"BACKGROUND CLASS  ({count}/{TARGET})",
                    (14, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (200, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(display, "[SPACE] capture   [Q] quit",
                    (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1, cv2.LINE_AA)

        # Progress bar
        cv2.rectangle(display, (20, 64), (fw - 20, 72), (50, 50, 50), -1)
        cv2.rectangle(display, (20, 64), (20 + bar_w, 72), (200, 80, 80), -1)

        # Warning text
        cv2.putText(display, "NO FRUIT IN FRAME",
                    (fw // 2 - 100, fh // 2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (80, 80, 200), 2, cv2.LINE_AA)

        cv2.imshow("Background Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif key == ord(' '):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(SAVE_DIR, f"bg_{ts}.jpg")
            cv2.imwrite(fname, frame)
            count += 1
            print(f"  Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Total background images: {count}")
    print(f"[INFO] Saved to: {SAVE_DIR}")
    print("[INFO] Retrain your model to include the 'background' class.\n")


if __name__ == "__main__":
    main()
