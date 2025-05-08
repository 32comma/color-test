# pip install opencv-python scikit-learn numpy

import cv2
import numpy as np
from sklearn.cluster import KMeans


def get_dominant_color(image, k=1):
    # 이미지를 1차원 배열로 변환
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # KMeans로 색상 클러스터링
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(data)

    # 가장 큰 클러스터의 중심을 반환
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color.astype(int)


def bgr_to_color_name(bgr):
    # 단순하게 RGB 값을 문자로 매핑하는 예시
    b, g, r = bgr
    if r > 200 and g < 100 and b < 100:
        return "Red"
    elif g > 200 and r < 100 and b < 100:
        return "Green"
    elif b > 200 and r < 100 and g < 100:
        return "Blue"
    elif r > 200 and g > 200 and b < 100:
        return "Yellow"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    else:
        return f"R:{r} G:{g} B:{b}"


# 동영상 파일 열기
cap = cv2.VideoCapture('test.mp4')

frame_interval = 30  # 30프레임마다 추출 (1초에 한 번 정도)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        resized = cv2.resize(frame, (100, 100))  # 계산량 줄이기
        color = get_dominant_color(resized)
        color_name = bgr_to_color_name(color)
        print(f"Frame {frame_count}: Dominant Color is {color_name}")

    frame_count += 1

cap.release()
