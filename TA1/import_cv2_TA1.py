import cv2
import numpy as np

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open USB camera")
    exit()

# List to store ball centroids for trajectory
trajectory = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected or no frame captured")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Invert (negative image)
    neg = cv2.bitwise_not(gray)

    # Thresholding to separate object
    _, thresh = cv2.threshold(neg, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Filter out small noise
        if cv2.contourArea(cnt) > 500:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw centroid
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Add centroid to trajectory
                trajectory.append((cx, cy))
                # Keep last 50 points for visibility
                if len(trajectory) > 50:
                    trajectory.pop(0)

    # Draw trajectory
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), 2)

    # Optional: draw bounding boxes
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show results
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
