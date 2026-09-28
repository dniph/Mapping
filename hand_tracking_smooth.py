import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import math
import numpy as np
import threading

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Descargando modelo...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )
    print("Modelo descargado!")

def draw_star(frame, center, size, color):
    cx, cy = center
    points = []
    for i in range(5):
        angle_out = (i * 4 * math.pi / 5) - math.pi / 2
        points.append((
            int(cx + size * math.cos(angle_out)),
            int(cy + size * math.sin(angle_out))
        ))
        angle_in = angle_out + 2 * math.pi / 5
        points.append((
            int(cx + (size * 0.4) * math.cos(angle_in)),
            int(cy + (size * 0.4) * math.sin(angle_in))
        ))
    pts = np.array(points, np.int32)
    cv2.fillPoly(frame, [pts], color)

# Estado compartido entre hilos
latest_landmarks = []
lock = threading.Lock()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def detection_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        with lock:
            latest_landmarks.clear()
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    h, w, _ = frame.shape
                    fingertips = [4, 8, 12, 16, 20]
                    for tip in fingertips:
                        finger = hand[tip]
                        x = int(finger.x * w)
                        y = int(finger.y * h)
                        latest_landmarks.append((x, y))
                    


# Arranca el hilo de deteccion
t = threading.Thread(target=detection_loop, daemon=True)
t.start()

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
trail = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Fade suave
    canvas = cv2.addWeighted(canvas, 0.88, np.zeros_like(canvas), 0.12, 0)

    with lock:
        current_landmarks = list(latest_landmarks)

    if current_landmarks:
        for pos in current_landmarks:
            trail.append(pos)
        if len(trail) > 60:
            trail = trail[-60:]

        for i, pos in enumerate(trail):
            t_val = i / len(trail)
            size = int(2 + t_val * 16)
            color = (
                int(203 * t_val),
                int(192 * t_val),
                int(255 * t_val)
            )
            draw_star(canvas, pos, size, color)
    else:
        trail.clear()

    output = cv2.addWeighted(frame, 0.5, canvas, 1.0, 0)
    cv2.imshow("Hand Tracking", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()