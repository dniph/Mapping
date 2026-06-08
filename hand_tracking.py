import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import math
import numpy as np

# Descarga el modelo si no existe
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

# Configuracion
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h, w, _ = frame.shape
            for landmark in hand:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (147, 20, 255), -1)

            # Estrella rosa en el indice
            index = hand[8]
            x = int(index.x * w)
            y = int(index.y * h)
            draw_star(frame, (x, y), 15, (203, 192, 255))
            print(f"Índice: x={x}, y={y}")

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()