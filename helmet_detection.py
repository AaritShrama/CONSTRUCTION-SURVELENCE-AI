import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def get_head_box(x1, y1, x2, y2, frac=0.35):
    """Return a box covering roughly the top `frac` of a person's bbox (the head region)."""
    h = y2 - y1
    head_y2 = y1 + int(h * frac)
    return x1, y1, x2, head_y2


def run_trained_mode(video_path, model_path, conf=0.25):

    model = YOLO(model_path)
    names = model.names

    PERSON_ID = None
    HARDHAT_ID = None
    VEST_ID = None

    # Automatically find class IDs
    for cid, cname in names.items():

        if cname == "Person":
            PERSON_ID = cid

        elif cname == "Hardhat":
            HARDHAT_ID = cid

        elif cname == "Safety Vest":
            VEST_ID = cid

    print("Person ID :", PERSON_ID)
    print("Hardhat ID:", HARDHAT_ID)
    print("Vest ID   :", VEST_ID)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            verbose=False
        )[0]

        persons = []
        hardhats = []
        vests = []

        # -------------------------
        # Collect detections
        # -------------------------
        for box in results.boxes:

            cls = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == PERSON_ID:

                track_id = -1

                if box.id is not None:
                    track_id = int(box.id.item())

                persons.append((x1, y1, x2, y2, track_id))

            elif cls == HARDHAT_ID:

                hardhats.append((x1, y1, x2, y2))

            elif cls == VEST_ID:

                vests.append((x1, y1, x2, y2))

        # -------------------------
        # Associate PPE to persons
        # -------------------------
        for px1, py1, px2, py2, track_id in persons:

            ########################
            # Helmet
            ########################

            head_bottom = py1 + int((py2 - py1) * 0.25)

            helmet_found = False

            for hx1, hy1, hx2, hy2 in hardhats:

                helmet_center_x = (hx1 + hx2) // 2
                helmet_center_y = (hy1 + hy2) // 2

                if (
                    px1 <= helmet_center_x <= px2
                    and py1 <= helmet_center_y <= head_bottom
                ):
                    helmet_found = True
                    break

            ########################
            # Vest
            ########################

            torso_top = py1 + int((py2 - py1) * 0.25)
            torso_bottom = py1 + int((py2 - py1) * 0.85)

            vest_found = False

            for vx1, vy1, vx2, vy2 in vests:

                vest_center_x = (vx1 + vx2) // 2
                vest_center_y = (vy1 + vy2) // 2

                if (
                    px1 <= vest_center_x <= px2
                    and torso_top <= vest_center_y <= torso_bottom
                ):
                    vest_found = True
                    break

            ########################
            # Draw
            ########################

            helmet_text = "Helmet" if helmet_found else "NO Helmet"
            vest_text = "Vest" if vest_found else "NO Vest"

            if helmet_found and vest_found:
                color = (0,255,0)

            elif helmet_found or vest_found:
                color = (0,255,255)

            else:
                color = (0,0,255)

            label = f"ID {track_id} | {helmet_text} | {vest_text}"

            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

            cv2.putText(
                frame,
                label,
                (px1, max(py1-10,0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

        cv2.imshow("PPE Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_fallback_mode(video_path, conf=0.4):
    """Rough heuristic mode. Uses stock yolov8n.pt (auto-downloads on first run)."""
    model = YOLO("yolov8n.pt")  # auto-downloads ~6MB the first time you run this
    PERSON_CLASS_ID = 0  # 'person' in COCO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, classes=[PERSON_CLASS_ID], verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            hx1, hy1, hx2, hy2 = get_head_box(x1, y1, x2, y2)

            head_crop = frame[hy1:hy2, hx1:hx2]
            if head_crop.size == 0:
                continue

            # Very rough heuristic: helmets are usually smooth, saturated, solid-colored
            # (hardhat plastic), while bare hair/skin has more texture variance.
            gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
            avg_sat = hsv[:, :, 1].mean()

            has_helmet = edge_density < 0.08 and avg_sat > 40

            color = (0, 255, 0) if has_helmet else (0, 0, 255)
            label = "Helmet?" if has_helmet else "No Helmet?"

            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(frame, label, (hx1, max(hy1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Helmet Detection (fallback heuristic - not reliable)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet detection on video using YOLOv8 + OpenCV")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--mode", choices=["trained", "fallback"], default="fallback",
                         help="'trained' = use your custom hardhat model, "
                              "'fallback' = quick heuristic demo, no custom model needed")
    parser.add_argument("--model", default=None,
                         help="Path to trained .pt weights (required for --mode trained)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    if args.mode == "trained":
        if not args.model:
            raise ValueError("--model is required when --mode trained (path to your best.pt)")
        run_trained_mode(args.video, args.model, args.conf)
    else:
        print("Running in FALLBACK heuristic mode. This is a rough demo, not production-accurate.")
        print("For real accuracy, train a model (see instructions at top of this file) "
              "and run with --mode trained --model path/to/best.pt")
        run_fallback_mode(args.video, args.conf)