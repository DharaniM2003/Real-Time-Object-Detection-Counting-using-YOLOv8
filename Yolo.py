# ==========================================
# 1️⃣ Install
# ==========================================
!pip install -q ultralytics opencv-python

# ==========================================
# 2️⃣ Imports
# ==========================================
import cv2
import random
import urllib.request
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

# ==========================================
# 3️⃣ Download Video from Direct URL
# ==========================================

video_url = "https://dm0qx8t0i9gc9.cloudfront.net/watermarks/video/Vd3bj2jPe/67fe605f0f7ea96ac93a7a3e-a6ayuf6tr4__cb9a65539de9d8b14163e9fcc7779305__P360.mp4"
output_video = "video.mp4"

req = urllib.request.Request(video_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    with open(output_video, 'wb') as out_file:
        out_file.write(response.read())

print("✅ Video downloaded")

# ==========================================
# 4️⃣ Load YOLOv8 Model
# ==========================================

model = YOLO("yolov8s.pt")

def get_color(cls_id):
    random.seed(cls_id)
    return tuple(random.randint(0, 255) for _ in range(3))

# ==========================================
# 5️⃣ Detection + Bounding Box + Counting
# ==========================================

cap = cv2.VideoCapture(output_video)
object_counts = {}
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        names = result.names

        for box in result.boxes:
            if float(box.conf[0]) > 0.4:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = names[cls]
                conf = float(box.conf[0])

                # Count objects
                object_counts[label] = object_counts.get(label, 0) + 1

                # Draw bounding box
                color = get_color(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

    # Show first 5 frames
    if frame_count < 5:
        cv2_imshow(frame)

    frame_count += 1

cap.release()

print("\n📊 Final Object Counts:")
print(object_counts)
