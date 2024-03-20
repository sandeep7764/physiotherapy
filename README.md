# physiotherapy
This code provided is a computer vision-based application designed to assist individuals in performing knee rehabilitation exercise correctly. Below is an overview of its functionality:

**Keypoint Detection**: The application uses the MediaPipe library for keypoint detection in real-time or from a video source. Specifically, it detects keypoints related to various body parts, including the hips, knees, and ankles.

**Angle Calculation**: Once the keypoints are detected, the application calculates the angle of knee flex. It computes the angle between the line segments connecting the ankle, knee, and hip joints. This angle serves as a crucial metric for assessing the correctness of knee rehabilitation exercises.

**Visual Feedback**: The application provides real-time visual feedback to the user. It displays the video feed with overlaid keypoint landmarks and the calculated angle of knee flexion. This visual feedback helps users monitor their exercise performance and make necessary adjustments.

**Rehabilitation Monitoring**: The application continuously monitors the user's knee flexion angle during the exercise session. It distinguishes between correct and incorrect knee movements based on predefined angle thresholds.

**Rep Counting**: The application tracks the number of repetitions completed by the user. It increments the rep count each time the user performs a correct knee movement. Rep counting helps users track their progress and adherence to the exercise regimen.

**Missed Rep Detection**: The application includes functionality to detect missed repetitions. If the user fails to maintain the correct knee flex angle within a specified time frame, the application registers a missed repetition. It provides feedback to the user to encourage adherence to proper form.

**User Interface**: The application features a graphical user interface (GUI) built using the Tkinter library. The GUI allows users to start and stop the exercise session, upload video files for analysis, and view real-time feedback on their exercise performance.

**Data Logging**: Upon completion of the exercise session, the application logs relevant data, including the user's name, exercise session timestamp, number of successful repetitions, and number of missed repetitions, into a CSV file. This data logging enables users to track their exercise history and share progress with healthcare providers if necessary.
