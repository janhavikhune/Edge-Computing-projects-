import cv2
import numpy as np

# ---------------- PLAYER DETECTION SETUP ----------------
player_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
if player_cascade.empty():
    print("Error loading Haar Cascade file for full body!")
    exit()

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)  # Change 0 to 1 if your camera is /dev/video1
if not cap.isOpened():
    print("Error: Could not open USB camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected or no frame captured")
        break

    # ---------------- PLAYER DETECTION ----------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    players = player_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 120))

    # Draw bounding boxes and count players
    player_count = len(players)
    cv2.putText(frame, f"Players: {player_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for (x, y, w, h) in players:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Player", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---------------- SHOW RESULTS ----------------
    cv2.imshow("Player Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
