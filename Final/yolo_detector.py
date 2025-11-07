import torch
import requests
import time
import cv2
import warnings
import os
import sys

warnings.filterwarnings("ignore")

# Add path for gpio_control
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sensors import gpio_control

# Load YOLOv5 model (optimized small)
print("🔁 Loading YOLOv5s model (optimized)...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False)
model.conf = 0.6  # confidence threshold
print("✅ YOLOv5 model loaded successfully!\n")

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Camera not detected!")
    exit()

print("📸 Camera initialized — Smart Security Bot is active.\n")

AGENTIC_AI_URL = "http://127.0.0.1:8888/query"
prev_action = "None"

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame read error, skipping...")
            continue

        # Run YOLO detection
        results = model(frame)
        detections = results.xyxy[0]

        person_detected = False
        confidence = 0.0

        for *box, conf, cls in detections:
            label = model.names[int(cls)]
            if label == "person" and conf > 0.6:
                person_detected = True
                confidence = float(conf) * 100
                break

        if person_detected:
            print(f"🧍 Person detected ({confidence:.1f}%) → Consulting Agentic AI...")
            try:
                response = requests.post(
                    AGENTIC_AI_URL,
                    json={
                        "detection": "person",
                        "confidence": confidence,
                        "prev_action": prev_action
                    },
                    timeout=15
                )
                if response.status_code == 200:
                    ai_reply = response.json()["response"].lower()
                    print(f"🤖 Agentic AI Decision: {ai_reply}")

                    if "buzzer" in ai_reply or "(c" in ai_reply or "sound" in ai_reply:
                        print("🔊 Triggering buzzer alert...")
                        gpio_control.buzzer_alert(0.5)
                        prev_action = "Buzzer Alert"
                    elif "led" in ai_reply or "(b" in ai_reply:
                        print("💡 Triggering LED alert...")
                        gpio_control.led_alert(1)
                        prev_action = "LED Alert"
                    else:
                        prev_action = "None"

                else:
                    print(f"⚠️ Agentic AI returned HTTP {response.status_code}")

            except requests.Timeout:
                print("⚠️ AI request timeout, skipping...")
            except Exception as e:
                print(f"⚠️ Agentic AI request failed: {e}")

        else:
            print("🙈 No person detected.")
            prev_action = "None"

        time.sleep(0.4)

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user, cleaning up...")
    cap.release()
    gpio_control.cleanup()
    print("✅ Exiting Smart Security Bot.")
