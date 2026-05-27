import random
import sys

import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image


video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])


COIN_RADIUS = 24
MIN_MARKER_RADIUS = 36
FALL_SPEED = 7
START_LIVES = 3

COLORS = [
    ("RED", (0, 0, 255)),
    ("BLUE", (255, 0, 0)),
    ("GREEN", (0, 180, 0)),
    ("YELLOW", (0, 220, 255)),
]
BUTTON_LABELS = ["A", "B", "X", "Y"]


# converts OpenCV image to PIL image and then to pyglet texture
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg


def draw_text(frame, text, position):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)


def draw_centered_text(frame, text, center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(center[0] - text_size[0] / 2)
    y = int(center[1] + text_size[1] / 2)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness + 2)
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness)


def draw_frame(window, frame):
    window.clear()
    img = cv2glet(frame, "BGR")
    img.blit(0, 0, 0, width=window.width, height=window.height)


def get_marker_center(corners):
    points = corners.reshape(4, 2)
    return points.mean(axis=0)


def get_marker_radius(corners):
    points = corners.reshape(4, 2).astype(np.float32)
    _, marker_radius = cv2.minEnclosingCircle(points)
    return max(MIN_MARKER_RADIUS, int(marker_radius * 1.7))


def assign_marker_colors(ids, marker_colors):
    for marker_id in sorted(ids):
        if marker_id in marker_colors:
            continue

        used_colors = set(marker_colors.values())
        for color_index in range(len(COLORS)):
            if color_index not in used_colors:
                marker_colors[marker_id] = color_index
                break
        else:
            marker_colors[marker_id] = marker_id % len(COLORS)


def find_markers(frame, detector, marker_colors):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    markers = []

    if ids is None:
        return markers

    marker_ids = [int(marker_id[0]) for marker_id in ids]
    assign_marker_colors(marker_ids, marker_colors)
    aruco.drawDetectedMarkers(frame, corners)

    for i, marker_id in enumerate(marker_ids):
        center = get_marker_center(corners[i])
        radius = get_marker_radius(corners[i])
        markers.append(
            {
                "id": marker_id,
                "center": (int(center[0]), int(center[1])),
                "radius": radius,
                "color": marker_colors[marker_id],
            }
        )

    return markers


def draw_marker(frame, marker):
    color_name, color = COLORS[marker["color"]]
    label = BUTTON_LABELS[marker["color"]]

    cv2.circle(frame, marker["center"], marker["radius"], color, -1)
    cv2.circle(frame, marker["center"], marker["radius"], (255, 255, 255), 3)
    draw_centered_text(frame, label, marker["center"])
    draw_text(
        frame,
        color_name + " #" + str(marker["id"]),
        (marker["center"][0] + 12, max(24, marker["center"][1] - 12)),
    )


def draw_coin(frame, coin):
    center = (coin["x"], coin["y"])
    color_name, color = COLORS[coin["color"]]

    cv2.circle(frame, center, COIN_RADIUS, color, -1)
    cv2.circle(frame, center, COIN_RADIUS, (255, 255, 255), 2)
    draw_text(frame, color_name[0], (center[0] - 8, center[1] + 8))


def new_coin(width, markers):
    margin = COIN_RADIUS + 10
    marker_colors = [marker["color"] for marker in markers]

    return {
        "x": random.randint(margin, width - margin),
        "y": -COIN_RADIUS,
        "color": random.choice(marker_colors),
    }


def coin_hits_marker(coin, marker):
    coin_position = np.array([coin["x"], coin["y"]])
    marker_position = np.array(marker["center"])
    distance = np.linalg.norm(coin_position - marker_position)
    return distance < COIN_RADIUS + marker["radius"]


def find_hit_marker(coin, markers):
    for marker in markers:
        if coin_hits_marker(coin, marker):
            return marker
    return None


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(video_id)

ret, frame = cap.read()
if not ret or frame is None:
    print("Could not read from webcam.")
    cap.release()
    sys.exit(1)

WINDOW_HEIGHT, WINDOW_WIDTH = frame.shape[:2]
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, "AR Marker Coin Game")

marker_colors = {}
coin = None
score = 0
lives = START_LIVES
last_result = "Move the matching marker circle into the coin"


@window.event
def on_draw():
    global coin
    global score, lives, last_result

    ret, frame = cap.read()
    if not ret or frame is None:
        return

    markers = find_markers(frame, detector, marker_colors)

    if coin is None and lives > 0 and len(markers) > 0:
        coin = new_coin(WINDOW_WIDTH, markers)

    if coin is not None and lives > 0:
        coin["y"] += FALL_SPEED
        hit_marker = find_hit_marker(coin, markers)

        if hit_marker is not None:
            if hit_marker["color"] == coin["color"]:
                score += 1
                last_result = "Correct color"
            else:
                lives -= 1
                last_result = "Wrong color"
            if lives > 0:
                coin = new_coin(WINDOW_WIDTH, markers)
            else:
                coin = None
        elif coin["y"] > WINDOW_HEIGHT + COIN_RADIUS:
            lives -= 1
            last_result = "Missed coin"
            if lives > 0 and len(markers) > 0:
                coin = new_coin(WINDOW_WIDTH, markers)
            else:
                coin = None

    if coin is not None and lives > 0:
        draw_coin(frame, coin)

    for marker in markers:
        draw_marker(frame, marker)

    if len(markers) == 0 and lives > 0:
        draw_text(frame, "Show the ArUco markers", (20, 200))

    draw_text(frame, "Score: " + str(score), (20, 40))
    draw_text(frame, "Lives: " + str(lives), (20, 80))
    draw_text(frame, last_result, (20, 120))
    draw_text(frame, "Detected markers: " + str(len(markers)), (20, 160))

    if lives <= 0:
        draw_text(frame, "Game over", (20, 200))

    draw_frame(window, frame)


@window.event
def on_close():
    cap.release()
    pyglet.app.exit()


def update(dt):
    window.invalid = True


pyglet.clock.schedule_interval(update, 1 / 30)
pyglet.app.run()
cap.release()
