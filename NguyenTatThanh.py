import os
import csv
import cv2
from ultralytics import YOLO
from collections import deque

# ==========================
# 1. CẤU HÌNH
# ==========================

MODEL_PATH = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject4\train_helmet\runs\detect\train\weights\best.pt"
VIDEO_PATH = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject4\khong doi mu hiem\video\NguyenTatThanh.mp4"
SNAPSHOT_DIR = "violations_snapshots"
LOG_CSV = "violations_log.csv"

NO_HELMET_CLASS_ID = 0  # 0 = no_helmet
HELMET_CLASS_ID = 1  # 1 = helmet

# CẢI THIỆN: Tăng ngưỡng phát hiện vi phạm
MIN_NO_HELMET_FRAMES = 30  # Tăng từ 10 lên 30 frame
REQUIRED_CONSECUTIVE_FRAMES = 15  # Số frame liên tiếp không đội mũ
WINDOW_SIZE = 20  # Kích thước cửa sổ để xem xét

# Ngưỡng confidence cao hơn
DETECTION_CONFIDENCE = 0.5  # Tăng từ 0.3 lên 0.5


# ==========================
# 2. HÀM PHỤ TRỢ
# ==========================

def get_majority_vote(history):
    """Lấy kết quả đa số từ lịch sử"""
    if not history:
        return None

    # Đếm số lần xuất hiện của mỗi class
    counts = {}
    for item in history:
        if item is not None:
            counts[item] = counts.get(item, 0) + 1

    if not counts:
        return None

    # Trả về class có số lần xuất hiện nhiều nhất
    return max(counts.items(), key=lambda x: x[1])[0]


# ==========================
# 3. HÀM CHÍNH
# ==========================

def main():
    # Tạo thư mục lưu ảnh vi phạm
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # Load model YOLO
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Lấy FPS video để tính thời gian
    cap_info = cv2.VideoCapture(VIDEO_PATH)
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    cap_info.release()
    if fps <= 0:
        fps = 30

    print(f"[INFO] Video FPS ~ {fps:.2f}")

    # Lưu trạng thái theo track_id
    # Cải thiện: lưu lịch sử các frame gần nhất
    tracker_state = {}

    # Log vi phạm
    violations_log = []

    frame_idx = 0

    print("[INFO] Starting tracking + detection...")
    results_generator = model.track(
        source=VIDEO_PATH,
        tracker="bytetrack.yaml",
        conf=DETECTION_CONFIDENCE,  # Dùng confidence cao hơn
        iou=0.5,
        imgsz=640,
        stream=True,
        persist=True
    )

    for result in results_generator:
        frame = result.orig_img  # frame BGR gốc

        if frame is None:
            break

        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()  # Lấy confidence scores
            xyxy = boxes.xyxy.cpu().tolist()

            for track_id, cls_id, conf, box in zip(ids, clss, confs, xyxy):
                x1, y1, x2, y2 = map(int, box)

                # Lấy / tạo state cho track_id
                state = tracker_state.setdefault(track_id, {
                    "history": deque(maxlen=WINDOW_SIZE),  # Lưu lịch sử các frame
                    "consecutive_no_helmet": 0,
                    "violated": False,
                    "last_valid_class": None
                })

                # CHỈ xử lý nếu confidence đủ cao
                if conf < 0.4:  # Bỏ qua detection có confidence thấp
                    continue

                # Thêm kết quả vào lịch sử
                state["history"].append(cls_id)

                # Xác định class dựa trên đa số trong lịch sử
                majority_class = get_majority_vote(state["history"])

                # Cập nhật streak dựa trên class đa số
                current_class = majority_class if majority_class is not None else cls_id

                if current_class == NO_HELMET_CLASS_ID:
                    state["consecutive_no_helmet"] += 1
                    state["last_valid_class"] = NO_HELMET_CLASS_ID
                elif current_class == HELMET_CLASS_ID:
                    # Reset streak CHỈ KHI có đủ bằng chứng
                    if state["last_valid_class"] == HELMET_CLASS_ID:
                        state["consecutive_no_helmet"] = 0
                    state["last_valid_class"] = HELMET_CLASS_ID

                # Chọn màu & label vẽ bbox dựa trên class hiện tại
                if current_class == NO_HELMET_CLASS_ID:
                    color = (0, 0, 255)  # đỏ
                    label = f"NO_HELMET #{track_id} (c:{conf:.2f})"
                elif current_class == HELMET_CLASS_ID:
                    color = (0, 255, 0)  # xanh lá
                    label = f"HELMET #{track_id} (c:{conf:.2f})"
                else:
                    color = (255, 255, 0)  # vàng cho class lạ
                    label = f"CLS{cls_id} #{track_id} (c:{conf:.2f})"

                # Vẽ bbox + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

                # Hiển thị thông tin về streak
                cv2.putText(
                    frame, f"Streak: {state['consecutive_no_helmet']}",
                    (x1, max(0, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )

                # Nếu chưa từng bị đánh dấu vi phạm và streak đủ dài
                # Yêu cầu khắt khe hơn: cần cả streak dài VÀ confidence cao
                if (not state["violated"] and
                        state["consecutive_no_helmet"] >= MIN_NO_HELMET_FRAMES and
                        conf >= 0.5):  # Thêm điều kiện confidence cao

                    state["violated"] = True

                    # Tính timestamp (giây)
                    timestamp_sec = frame_idx / fps
                    timestamp_str = f"{timestamp_sec:.2f}s"

                    # Lưu snapshot (ảnh toàn cảnh)
                    snap_name = f"violation_id{track_id}_frame{frame_idx}.jpg"
                    snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
                    cv2.imwrite(snap_path, frame)

                    print(f"[VIOLATION] ID {track_id} at frame {frame_idx}, time {timestamp_str}")
                    print(f"          Confidence: {conf:.2f}, Streak: {state['consecutive_no_helmet']}")
                    print(f"          Snapshot: {snap_path}")

                    # Vẽ chữ VIOLATION to trên đầu bbox
                    cv2.putText(
                        frame, "VIOLATION!",
                        (x1, max(0, y1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

                    # Ghi vào log
                    violations_log.append({
                        "track_id": track_id,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_str,
                        "confidence": f"{conf:.2f}",
                        "streak": state["consecutive_no_helmet"],
                        "snapshot": snap_path
                    })

        # Hiển thị real-time (ấn 'q' để thoát)
        cv2.imshow("Helmet Violation Detection", frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Stop by user.")
            break

    # Giải phóng tài nguyên
    cv2.destroyAllWindows()

    # Ghi log CSV
    if violations_log:
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "track_id", "frame_idx", "timestamp", "confidence",
                "streak", "snapshot"
            ])
            writer.writeheader()
            writer.writerows(violations_log)
        print(f"[INFO] Saved violations log to {LOG_CSV}")
    else:
        print("[INFO] No violations detected.")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()