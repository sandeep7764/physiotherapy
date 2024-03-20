__author__ = 'Sandeepa H A'
import cv2
import time
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog


def calc_angle(ankle, knee, hip):
    # Dropping the z-coordinate as angle between two lines requires 2D coordinates
    ankle = np.array([ankle.x, ankle.y])
    knee = np.array([knee.x, knee.y])
    hip = np.array([hip.x, hip.y])

    # Building the straight lines in vector form
    ankle_knee = np.subtract(ankle, knee)  # Straight line joining the points ankle and knee
    knee_hip = np.subtract(knee, hip)  # Straight line joining the points knee and hip

    # Calculating the angles between the two straight lines
    angle = np.arccos(np.dot(ankle_knee, knee_hip) / np.multiply(np.linalg.norm(ankle_knee), np.linalg.norm(knee_hip)))
    angle = 180 - 180 * angle / 3.14

    return np.round(angle, 2)


def knee_rehab(videoCapture_param=0, webcam=False, filepath=None, rep_time=8):
    flag = None  # Current position of leg. Either 'straight' or 'bent'
    count = -1
    start = 0
    timer = 0
    angle = 0

    mp_drawing = mp.solutions.drawing_utils  # Connecting Keypoints Visuals
    mp_pose = mp.solutions.pose  # Keypoint detection model

    if webcam:
        cap = cv2.VideoCapture(videoCapture_param)
    else:
        cap = cv2.VideoCapture(filepath)

    pose = mp_pose.Pose(min_detection_confidence=0.1,
                        min_tracking_confidence=0.95)  # min_tracking_confidence is set high to increase robustness

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB as mediapipe requires RGB images
        image.flags.writeable = False

        results = pose.process(image)  # Make predictions

        cv2.rectangle(frame, (0, 0), (640, 60), (245, 117, 16),
                      -1)  # Banner at top of the display window to write texts on

        try:
            # Extract the required landmarks aka keypoints
            landmarks = results.pose_landmarks.landmark
            hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            knee_left = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            knee_right = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle_left = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            ankle_right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            right_shoulder_z = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z
            left_shoulder_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z

            if left_shoulder_z < right_shoulder_z:  # Body part closer to the camera to be used for rehab
                cv2.putText(frame, '|| Detecting ||', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1)
                angle = calc_angle(ankle_left, knee_left, hip_left)

            else:
                cv2.putText(frame, '|| Detecting ||', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1)
                angle = calc_angle(ankle_right, knee_right, hip_right)

            # Display angle of knee bend in the OpenCV window
            cv2.putText(frame, str(angle),
                        tuple(np.multiply([knee_left.x, knee_left.y], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 255), 1)
        except:
            pass

        # Straight leg condition
        if angle > 160:
            if timer >= rep_time:  # If patient had bent its knee for more than rep_time sec
                count = count + 1  # Increase the rep counter
                timer = 0  # Reset the timer
                flag = 'straight'  # Leg is straight
            elif timer < rep_time and timer > 1 and flag == 'bent':  # Leg straighened before rep_time seconds
                # timer > 1 is indtroduced to remove model's inaccurate predictions from disturbing our counter and warning flags
                cv2.rectangle(frame, (0, 440), (640, 480), (245, 117, 16), -1)
                cv2.putText(frame, 'Please Keep Your Knee Bent',
                            (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                timer = 0  # Timer reset without Rep count increament
                start = time.time()

                # Bent leg condition
        elif angle <= 160:
            if flag == 'straight':
                start = time.time()
                flag = 'bent'
            timer = time.time() - start

        # Illustrations on OpenCV Window

        cv2.putText(frame, 'Reps = ' + str(count), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Timer = " + str(round(timer)) + ' sec', (450, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220, 1))
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def choose_input():
    def on_webcam():
        rep_time = int(rep_time_entry.get())
        root.destroy()
        knee_rehab(webcam=True, rep_time=rep_time)

    def on_video_file():
        filepath = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4")])
        if filepath:
            rep_time = int(rep_time_entry.get())
            root.destroy()
            knee_rehab(filepath=filepath, rep_time=rep_time)

    root = tk.Tk()
    root.title("Major Project")
    root.geometry("500x300")

    rep_time_label = tk.Label(root, text="Select Repetition Time (seconds):")
    rep_time_label.pack(pady=5)

    rep_time_entry = tk.Entry(root)
    rep_time_entry.pack(pady=5)
    rep_time_entry.insert(0, "8")  # Default repetition time

    webcam_button = tk.Button(root, text="Use Webcam", command=on_webcam)
    webcam_button.pack(pady=10)

    file_button = tk.Button(root, text="Use Video File", command=on_video_file)
    file_button.pack(pady=10)

    root.mainloop()


if __name__ == '__main__':
    choose_input()
