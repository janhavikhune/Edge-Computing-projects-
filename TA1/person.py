import cv2
import numpy as np

# Haar cascade for full body players
player_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
if player_cascade.empty():
    print("Error loading Haar Cascade file for full body!")
    exit()

cap = cv2.VideoCapture(0)  # or replace with video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- PLAYER DETECTION ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    players = player_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 120))

    # Draw boxes around players
    for (x, y, w, h) in players:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # green box
        cv2.putText(frame, "Player", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Count number of players detected
    player_count = len(players)
    cv2.putText(frame, f"Players: {player_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ---------- BALL DETECTION (Contours) ----------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Example: detecting a white football (tune values for your ball color!)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 3000:  # filter noise and very big areas
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # blue box
            cx = x + w//2
            cy = y + h//2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # red centroid
            cv2.putText(frame, "Ball", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    # ---------- SHOW RESULTS ----------
    cv2.imshow("Mask (Ball)", mask)
    cv2.imshow("Sports Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
