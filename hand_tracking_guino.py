import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import math
import numpy as np
import threading
import random
from datetime import datetime
import time

model_path = "hand_landmarker.task"
face_model_path = "face_landmarker.task"

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

latest_landmarks = []
wink_stars = []
lock = threading.Lock()

base_options_hand = python.BaseOptions(model_asset_path=model_path)
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options_hand,
    num_hands=2,
    min_hand_detection_confidence=0.7
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

base_options_face = python.BaseOptions(model_asset_path=face_model_path)
face_options = vision.FaceLandmarkerOptions(
    base_options=base_options_face,
    num_faces=1
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

def eye_ratio(landmarks, points, w, h):
    top = landmarks[points[0]]
    bottom = landmarks[points[1]]
    left = landmarks[points[2]]
    right = landmarks[points[3]]
    vertical = math.dist((top.x * w, top.y * h), (bottom.x * w, bottom.y * h))
    horizontal = math.dist((left.x * w, left.y * h), (right.x * w, right.y * h))
    return vertical / horizontal if horizontal > 0 else 0

def detection_loop():
    wink_cooldown = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = hand_detector.detect(mp_image)
        face_result = face_detector.detect(mp_image)

        h, w = frame.shape[:2]

        with lock:
            latest_landmarks.clear()

            if hand_result.hand_landmarks:
                for hand in hand_result.hand_landmarks:
                    for tip in [4, 8, 12, 16, 20]:
                        finger = hand[tip]
                        x = int(finger.x * w)
                        y = int(finger.y * h)
                        latest_landmarks.append((x, y))

            if face_result.face_landmarks and wink_cooldown == 0:
                face = face_result.face_landmarks[0]
                left_ear = eye_ratio(face, [159, 145, 33, 133], w, h)
                right_ear = eye_ratio(face, [386, 374, 362, 263], w, h)

                if left_ear < 0.15 and right_ear > 0.2:
                    eye_x = int(face[33].x * w)
                    eye_y = int(face[33].y * h)
                    for i in range(8):
                        angle = random.uniform(0.3, 0.9)
                        speed = random.uniform(2, 6)
                        size = random.randint(6, 16)
                        wink_stars.append({
                            "x": float(eye_x),
                            "y": float(eye_y),
                            "vx": -math.cos(angle) * speed,
                            "vy": -math.sin(angle) * speed,
                            "size": size,
                            "life": 1.0
                        })
                    wink_cooldown = 20

            if wink_cooldown > 0:
                wink_cooldown -= 1

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
trail = []
recording = False
video_writer = None

# FPS real del loop
fps_timer = time.time()
fps_actual = 15
frame_count_fps = 0

t = threading.Thread(target=detection_loop, daemon=True)
t.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Medir fps real
    frame_count_fps += 1
    if time.time() - fps_timer >= 1.0:
        fps_actual = frame_count_fps
        frame_count_fps = 0
        fps_timer = time.time()

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
            color = (int(203 * t_val), int(192 * t_val), int(255 * t_val))
            draw_star(canvas, pos, size, color)
    else:
        trail.clear()

    for star in wink_stars:
        star["x"] += star["vx"]
        star["y"] += star["vy"]
        star["vy"] += 0.1
        star["life"] -= 0.04
        if star["life"] > 0:
            alpha = star["life"]
            color = (0, int(255 * alpha), int(255 * alpha))
            draw_star(canvas, (int(star["x"]), int(star["y"])), star["size"], color)

    wink_stars[:] = [s for s in wink_stars if s["life"] > 0]

    output = cv2.addWeighted(frame, 0.5, canvas, 1.0, 0)

    # FPS en pantalla
    cv2.putText(output, f"FPS: {fps_actual}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Tracking", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        if not recording:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_writer = cv2.VideoWriter(filename, fourcc, fps_actual, (640, 480))
            recording = True
            print(f"⏺ Grabando a {fps_actual}fps... {filename}")
        else:
            recording = False
            video_writer.release()
            video_writer = None
            print("⏹ Guardado!")

    if recording and video_writer:
        video_writer.write(output)

    if key == ord('p'):
        photo_filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(photo_filename, output)
        print(f"📸 Foto guardada: {photo_filename}")

    if key == ord('q'):
        break

if video_writer:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()