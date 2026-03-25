import os
import csv
import cv2
from ultralytics import YOLO
from collections import deque
from datetime import datetime

# ==========================
# 1. CẤU HÌNH
# ==========================

MODEL_PATH = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject4\train_helmet\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject4\khong doi mu hiem\video\NguyenTatThanh.mp4"
SNAPSHOT_DIR = "violations_snapshots"
OUTPUT_DIR = "output_videos"
LOG_CSV = "violations_log.csv"

NO_HELMET_CLASS_ID = 0  # 0 = no_helmet
HELMET_CLASS_ID = 1  # 1 = helmet

# Tăng ngưỡng phát hiện vi phạm
MIN_NO_HELMET_FRAMES = 30
REQUIRED_CONSECUTIVE_FRAMES = 15
WINDOW_SIZE = 20

# Ngưỡng confidence
DETECTION_CONFIDENCE = 0.5


# ==========================
# 2. HÀM PHỤ TRỢ
# ==========================

def get_majority_vote(history):
    """Lấy kết quả đa số từ lịch sử"""
    if not history:
        return None

    counts = {}
    for item in history:
        if item is not None:
            counts[item] = counts.get(item, 0) + 1

    if not counts:
        return None

    return max(counts.items(), key=lambda x: x[1])[0]


def draw_info_panel(frame, frame_idx, fps, total_violations):
    """Vẽ panel thông tin trên video"""
    h, w = frame.shape[:2]

    # Tạo panel đen ở trên cùng
    panel_height = 100
    panel = frame[0:panel_height, :].copy()
    overlay = panel.copy()
    overlay[:] = (30, 30, 30)
    cv2.addWeighted(overlay, 0.7, panel, 0.3, 0, panel)
    frame[0:panel_height, :] = panel

    # Thông tin bên trái
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Time: {frame_idx / fps:.1f}s", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Thông tin giữa
    cv2.putText(frame, "HELMET DETECTION SYSTEM", (w // 2 - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Thông tin bên phải
    cv2.putText(frame, f"Violations: {total_violations}", (w - 280, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Chỉ báo REC
    cv2.circle(frame, (w - 280, 70), 8, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (w - 260, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame


# ==========================
# 3. HÀM CHÍNH
# ==========================

def main():
    # Tạo các thư mục cần thiết
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model YOLO
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Lấy thông tin video
    cap_info = cv2.VideoCapture(VIDEO_PATH)
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_info.release()

    if fps <= 0:
        fps = 30

    print(f"[INFO] Video info:")
    print(f"       FPS: {fps:.2f}")
    print(f"       Resolution: {frame_width}x{frame_height}")
    print(f"       Total frames: {total_frames}")

    # Thiết lập VideoWriter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(OUTPUT_DIR, f"helmet_detection_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"[INFO] Output video: {output_video_path}")

    # Lưu trạng thái theo track_id
    tracker_state = {}

    # Log vi phạm
    violations_log = []
    total_violations = 0

    frame_idx = 0

    print("[INFO] Starting tracking + detection...")
    print("       Press 'q' to quit, 'p' to pause")

    results_generator = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        conf=DETECTION_CONFIDENCE,
        iou=0.5,
        imgsz=640,
        stream=True,
        persist=True
    )

    for result in results_generator:
        frame = result.orig_img.copy()  # frame BGR gốc

        if frame is None:
            break

        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()

            for track_id, cls_id, conf, box in zip(ids, clss, confs, xyxy):
                x1, y1, x2, y2 = map(int, box)

                # Lấy / tạo state cho track_id
                state = tracker_state.setdefault(track_id, {
                    "history": deque(maxlen=WINDOW_SIZE),
                    "consecutive_no_helmet": 0,
                    "violated": False,
                    "last_valid_class": None
                })

                # CHỈ xử lý nếu confidence đủ cao
                if conf < 0.4:
                    continue

                # Thêm kết quả vào lịch sử
                state["history"].append(cls_id)

                # Xác định class dựa trên đa số trong lịch sử
                majority_class = get_majority_vote(state["history"])
                current_class = majority_class if majority_class is not None else cls_id

                if current_class == NO_HELMET_CLASS_ID:
                    state["consecutive_no_helmet"] += 1
                    state["last_valid_class"] = NO_HELMET_CLASS_ID
                elif current_class == HELMET_CLASS_ID:
                    if state["last_valid_class"] == HELMET_CLASS_ID:
                        state["consecutive_no_helmet"] = 0
                    state["last_valid_class"] = HELMET_CLASS_ID

                # Chọn màu & label
                if current_class == NO_HELMET_CLASS_ID:
                    color = (0, 0, 255)  # đỏ
                    label = f"NO_HELMET #{track_id}"
                elif current_class == HELMET_CLASS_ID:
                    color = (0, 255, 0)  # xanh lá
                    label = f"HELMET #{track_id}"
                else:
                    color = (255, 255, 0)  # vàng
                    label = f"CLS{cls_id} #{track_id}"

                # Vẽ bbox với viền dày hơn nếu vi phạm
                thickness = 4 if state["violated"] else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Vẽ label với background
                label_text = f"{label} ({conf:.2f})"
                (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Hiển thị streak
                streak_text = f"Streak: {state['consecutive_no_helmet']}"
                cv2.putText(frame, streak_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Phát hiện vi phạm
                if (not state["violated"] and
                        state["consecutive_no_helmet"] >= MIN_NO_HELMET_FRAMES and
                        conf >= 0.5):
                    state["violated"] = True
                    total_violations += 1

                    timestamp_sec = frame_idx / fps
                    timestamp_str = f"{timestamp_sec:.2f}s"

                    # Lưu snapshot
                    snap_name = f"violation_id{track_id}_frame{frame_idx}.jpg"
                    snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
                    cv2.imwrite(snap_path, frame)

                    print(f"[VIOLATION] ID {track_id} at frame {frame_idx}, time {timestamp_str}")
                    print(f"            Confidence: {conf:.2f}, Streak: {state['consecutive_no_helmet']}")
                    print(f"            Snapshot: {snap_path}")

                    # Vẽ chữ VIOLATION lớn
                    violation_text = "VIOLATION!"
                    (viol_w, viol_h), _ = cv2.getTextSize(violation_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    viol_x = max(0, x1)
                    viol_y = max(30, y1 - 40)

                    # Background cho chữ VIOLATION
                    cv2.rectangle(frame, (viol_x, viol_y - viol_h - 5),
                                  (viol_x + viol_w + 10, viol_y + 5), (0, 0, 255), -1)
                    cv2.putText(frame, violation_text, (viol_x + 5, viol_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                    # Ghi vào log
                    violations_log.append({
                        "track_id": track_id,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_str,
                        "confidence": f"{conf:.2f}",
                        "streak": state["consecutive_no_helmet"],
                        "snapshot": snap_path
                    })

        # Vẽ panel thông tin
        frame = draw_info_panel(frame, frame_idx, fps, total_violations)

        # Ghi frame vào video output
        video_writer.write(frame)

        # Hiển thị progress
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(
                f"[PROGRESS] Processing: {frame_idx}/{total_frames} frames ({progress:.1f}%) - Violations: {total_violations}")

        # Hiển thị real-time
        cv2.imshow("Helmet Violation Detection", frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Stopped by user.")
            break
        elif key == ord('p'):
            print("[INFO] Paused. Press any key to continue...")
            cv2.waitKey(0)

    # Giải phóng tài nguyên
    video_writer.release()
    cv2.destroyAllWindows()

    # Ghi log CSV
    if violations_log:
        csv_path = f"violations_log_{timestamp}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "track_id", "frame_idx", "timestamp", "confidence",
                "streak", "snapshot"
            ])
            writer.writeheader()
            writer.writerows(violations_log)
        print(f"[INFO] Saved violations log to {csv_path}")
    else:
        print("[INFO] No violations detected.")

    # Tổng kết
    print("\n" + "=" * 80)
    print(" TỔNG KẾT:")
    print("=" * 80)
    print(f"    Tổng frame xử lý: {frame_idx}")
    print(f"    Tổng vi phạm: {total_violations}")
    print(f"    FPS video: {fps:.2f}")
    print(f"\n  DỮ LIỆU ĐÃ LƯU:")
    print(f"    Snapshots: ./{SNAPSHOT_DIR}/")
    if violations_log:
        print(f"    CSV Log: ./{csv_path}")
    print(f"    📹 VIDEO OUTPUT: {output_video_path}")
    print("=" * 80)
    print("\n PHÍM TẮT:")
    print("   Q - Thoát")
    print("   P - Tạm dừng")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()