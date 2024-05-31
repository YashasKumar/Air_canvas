import mediapipe as mp
import numpy as np
import cv2
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

points = [deque(maxlen=1024)]
index = 0

paint_window = np.ones((480, 640, 3)) * 255

drawing_mode = False

def reset_canvas():
    global points, paint_window, index
    points = [deque(maxlen=1024)]
    paint_window = np.ones((480, 640, 3)) * 255
    index = 0

def draw_text(window, text, position):
    cv2.rectangle(window, (0, 0), (200, 30), (255, 255, 255), -1)
    cv2.putText(window, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_on_canvas():
    for j in range(len(points)):
        for k in range(1, len(points[j])):
            if points[j][k - 1] is None or points[j][k] is None:
                continue
            cv2.line(paint_window, points[j][k - 1], points[j][k], (0, 0, 0), 2)

    
with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9) as hands:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                                              mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2))
                    if drawing_mode:
                        index_tip = hand_landmarks.landmark[8]
                        thumb_tip = hand_landmarks.landmark[4]
                        center = (int(index_tip.x * 640), int(index_tip.y * 480))
                        thumb = (int(thumb_tip.x * 640), int(thumb_tip.y * 480))

                        if (thumb[1] - center[1] < 30):
                            points.append(deque(maxlen=1024))
                            index += 1
                        points[index].appendleft(center)

            draw_on_canvas()

            cv2.imshow("Air Canvas", paint_window)
            cv2.imshow("Hand Tracking", image)

            if cv2.waitKey(1) == ord('q'):
                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = f'Image_{timestamp}.png'
                cv2.imwrite(filename, paint_window)
                break
            elif key == ord('c'):
                reset_canvas()
            elif key == ord('d'):
                drawing_mode = True
                draw_text(paint_window, 'Drawing mode', (10, 20))
            elif key == ord('r'):
                drawing_mode = False
                draw_text(paint_window, 'Viewing mode', (10, 20))

cap.release()
cv2.destroyAllWindows()
