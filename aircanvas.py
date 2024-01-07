import mediapipe as mp
import numpy as np
import cv2
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

coors = []
paint_window = np.zeros((480, 640, 3)) + 255
drawing_mode = False #Keeps track of if we are in the drawing mode

def reset():
    global coors, paint_window #Accesses the global variables and changes the values of it everywhere
    coors.clear()
    paint_window = np.zeros((480, 640, 3)) + 255

with mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.flip(frame, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        landmarks = results.multi_hand_landmarks
        if landmarks:
            for num, hand in enumerate(landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
                                          )
            if drawing_mode: # Takes in values of the coordinates and saves it in coors only if in drawing mode
                for hand in landmarks:
                    coors.append(np.array([hand.landmark[8].x, hand.landmark[8].y])) # Appending the values into coors only of the index tip
                for i in range(1, len(coors)):
                        cv2.line(paint_window, tuple(np.multiply(coors[i - 1], (640, 480)).astype(int)),
                                tuple(np.multiply(coors[i], (640, 480)).astype(int)), (0,0,0), 2)# This command draws lines between 2 consecutive points in coors to maintain continuity

        cv2.imshow("Air canvas", paint_window)
        cv2.imshow("Hand tracking", image)
        key = cv2.waitKey(1)
        if key == ord('q'):#Stores the final image on the paint window 
            cv2.rectangle(paint_window, (0, 0), (150, 30), (255, 255, 255), -1)  # Clear previous text
            timestamp = time.strftime("%Y%m%d%H%M%S")
            filename = f'Image_{timestamp}.png'
            cv2.imwrite(filename, paint_window)
            break
# Assuming paint_window is a white canvas initially
# You can replace (255, 255, 255) with the appropriate background color if it's different
        if key == ord('c'):
            reset()
            if drawing_mode:
                cv2.putText(paint_window, 'Drawing mode', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)
            else:
                cv2.putText(paint_window, 'Viewing mode', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)
        elif key == ord('d'):
            cv2.rectangle(paint_window, (0, 0), (150, 30), (255, 255, 255), -1)  # Clear previous text
            cv2.putText(paint_window, 'Drawing mode', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)
            drawing_mode = True
        elif key == ord('r'):
            cv2.rectangle(paint_window, (0, 0), (150, 30), (255, 255, 255), -1)  # Clear previous text
            cv2.putText(paint_window, 'Viewing mode', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)
            drawing_mode = False
        else:
            cv2.rectangle(paint_window, (0, 0), (150, 30), (255, 255, 255), -1)  # Clear previous text
            if drawing_mode:
                cv2.putText(paint_window, 'Drawing mode', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)
            else:
                cv2.putText(paint_window, 'Viewing mode', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_4)

cap.release()
cv2.destroyAllWindows()
