# import csv
# import cv2
# import mediapipe as mp
# import numpy as np
# import os
#
# # Initialize MediaPipe pose
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_draw = mp.solutions.drawing_utils
#
# # Define injury thresholds
# #football
# basketball_injury_angle_ranges = {
#     "Ankle Sprain": {
#         "left_shoulder": (0, 3), "right_shoulder": (87, 93),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (7, 13), "right_knee": (77, 83),
#         "left_hip": (27, 33), "right_hip": (47, 53)
#     },
#     "ACL Tear": {
#         "left_shoulder": (77, 83), "right_shoulder": (87, 93),
#         "left_elbow": (0, 3), "right_elbow": (7, 13),
#         "left_knee": (157, 163), "right_knee": (117, 123),
#         "left_hip": (47, 53), "right_hip": (22, 28)
#     },
#     "MCL Tear": {
#         "left_shoulder": (47, 53), "right_shoulder": (57, 63),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (27, 33), "right_knee": (157, 163),
#         "left_hip": (17, 23), "right_hip": (37, 43)
#     },
#     "Achilles Tendon Rupture": {
#         "left_shoulder": (0, 3), "right_shoulder": (7, 13),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (57, 63), "right_knee": (47, 53),
#         "left_hip": (27, 33), "right_hip": (17, 23)
#     },
#     "Rotator Cuff Tear": {
#         "left_shoulder": (3, 6), "right_shoulder": (117, 123),
#         "left_elbow": (7, 13), "right_elbow": (27, 33),
#         "left_knee": (17, 23), "right_knee": (22, 28),
#         "left_hip": (7, 13), "right_hip": (37, 43)
#     },
#     "Labrum Tear (Shoulder)": {
#         "left_shoulder": (177, 183), "right_shoulder": (27, 33),
#         "left_elbow": (17, 23), "right_elbow": (47, 53),
#         "left_knee": (0, 3), "right_knee": (0, 3),
#         "left_hip": (87, 93), "right_hip": (12, 18)
#     },
#     "Hamstring Strain": {
#         "left_shoulder": (7, 13), "right_shoulder": (57, 63),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (7, 13), "right_knee": (27, 33),
#         "left_hip": (87, 93), "right_hip": (47, 53)
#     },
#     "Quadriceps Contusion": {
#         "left_shoulder": (7, 13), "right_shoulder": (87, 93),
#         "left_elbow": (17, 23), "right_elbow": (7, 13),
#         "left_knee": (77, 83), "right_knee": (87, 93),
#         "left_hip": (37, 43), "right_hip": (57, 63)
#     },
#     "Patellar Tendinitis": {
#         "left_shoulder": (57, 63), "right_shoulder": (67, 73),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (77, 83), "right_knee": (87, 93),
#         "left_hip": (47, 53), "right_hip": (57, 63)
#     },
#     "Wrist Fracture": {
#         "left_shoulder": (97, 103), "right_shoulder": (127, 133),
#         "left_elbow": (0, 3), "right_elbow": (147, 153),
#         "left_knee": (17, 23), "right_knee": (27, 33),
#         "left_hip": (7, 13), "right_hip": (37, 43)
#     },
#     "Finger Dislocation": {
#         "left_shoulder": (87, 93), "right_shoulder": (47, 53),
#         "left_elbow": (0, 3), "right_elbow": (97, 103),
#         "left_knee": (27, 33), "right_knee": (47, 53),
#         "left_hip": (12, 18), "right_hip": (17, 23)
#     },
#     "Labral Tear (Hip)": {
#         "left_shoulder": (7, 13), "right_shoulder": (0, 3),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (57, 63), "right_knee": (87, 93),
#         "left_hip": (97, 103), "right_hip": (27, 33)
#     },
#     "Concussion": {
#         "left_shoulder": (0, 3), "right_shoulder": (0, 3),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (0, 3), "right_knee": (0, 3),
#         "left_hip": (0, 3), "right_hip": (0, 3)  # Neutral angles
#     },
#     "Hip Pointer": {
#         "left_shoulder": (7, 13), "right_shoulder": (27, 33),
#         "left_elbow": (0, 3), "right_elbow": (7, 13),
#         "left_knee": (67, 73), "right_knee": (77, 83),
#         "left_hip": (127, 133), "right_hip": (97, 103)
#     },
#     "Calf Strain": {
#         "left_shoulder": (0, 3), "right_shoulder": (17, 23),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (27, 33), "right_knee": (47, 53),
#         "left_hip": (17, 23), "right_hip": (37, 43)
#     },
#     "Plantar Fasciitis": {
#         "left_shoulder": (7, 13), "right_shoulder": (0, 3),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (17, 23), "right_knee": (27, 33),
#         "left_hip": (47, 53), "right_hip": (57, 63)
#     },
#     "Back Spasm": {
#         "left_shoulder": (27, 33), "right_shoulder": (27, 33),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (47, 53), "right_knee": (57, 63),
#         "left_hip": (67, 73), "right_hip": (77, 83)
#     },
#     "Shoulder Impingement": {
#         "left_shoulder": (107, 113), "right_shoulder": (27, 33),
#         "left_elbow": (57, 63), "right_elbow": (67, 73),
#         "left_knee": (0, 3), "right_knee": (0, 3),
#         "left_hip": (47, 53), "right_hip": (57, 63)
#     },
#     "Stress Fracture": {
#         "left_shoulder": (7, 13), "right_shoulder": (7, 13),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (87, 93), "right_knee": (47, 53),
#         "left_hip": (27, 33), "right_hip": (77, 83)
#     },
#     "Knee Bursitis": {
#         "left_shoulder": (27, 33), "right_shoulder": (17, 23),
#         "left_elbow": (7, 13), "right_elbow": (0, 3),
#         "left_knee": (97, 103), "right_knee": (87, 93),
#         "left_hip": (47, 53), "right_hip": (57, 63)
#     }
# }
#
#
#
# # Function to calculate angles
# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ab = a - b
#     bc = c - b
#     # Avoid division by zero
#     ab_norm = np.linalg.norm(ab)
#     bc_norm = np.linalg.norm(bc)
#     if ab_norm == 0 or bc_norm == 0:
#         return 0
#     angle = np.arccos(np.clip(np.dot(ab, bc) / (ab_norm * bc_norm), -1.0, 1.0))
#     return np.degrees(angle)
#
#
# # Function to check if angle is within range
# def is_within_range(angle, range_tuple):
#     return range_tuple[0] <= angle <= range_tuple[1]
#
# # Input/output paths
# input_video_path = 'videoplayback_basketball_2.mp4'
# output_csv_path = 'detected_injuries_basketball_2.csv'
# output_video_path = 'output_video_basketball_2.mp4'
#
# print("Processing video for rendering joint angles and risk probabilities...")
#
# cap = cv2.VideoCapture(input_video_path)
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()
#
# frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
# # Initialize video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
#
# # Open CSV file for output
# with open(output_csv_path, mode='w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerow(['Timestamp (mm:ss)', 'Joint', 'Image File'])
#
#     print("Processing video. This may take some time...")
#
#     frame_count = 0
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#
#         frame_count += 1
#         timestamp_seconds = frame_count / fps
#         timestamp = f"{int(timestamp_seconds // 60):02}:{int(timestamp_seconds % 60):02}"
#
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = pose.process(rgb_frame)
#
#         if result.pose_landmarks:
#             mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#             landmarks = result.pose_landmarks.landmark
#             h, w, _ = frame.shape
#
#             # Get bounding box coordinates
#             x_coords = [landmark.x for landmark in landmarks if landmark.visibility > 0.5]
#             y_coords = [landmark.y for landmark in landmarks if landmark.visibility > 0.5]
#
#             if x_coords and y_coords:
#                 # Convert normalized coordinates to pixel values
#                 x_min = int(min(x_coords) * w)
#                 x_max = int(max(x_coords) * w)
#                 y_min = int(min(y_coords) * h)
#                 y_max = int(max(y_coords) * h)
#
#                 # Draw bounding box
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#
#             # Get joint positions
#             points = {
#                 'left_shoulder': [landmarks[11].x * w, landmarks[11].y * h],
#                 'right_shoulder': [landmarks[12].x * w, landmarks[12].y * h],
#                 'left_elbow': [landmarks[13].x * w, landmarks[13].y * h],
#                 'right_elbow': [landmarks[14].x * w, landmarks[14].y * h],
#                 'left_wrist': [landmarks[15].x * w, landmarks[15].y * h],
#                 'right_wrist': [landmarks[16].x * w, landmarks[16].y * h],
#                 'left_hip': [landmarks[23].x * w, landmarks[23].y * h],
#                 'right_hip': [landmarks[24].x * w, landmarks[24].y * h],
#                 'left_knee': [landmarks[25].x * w, landmarks[25].y * h],
#                 'right_knee': [landmarks[26].x * w, landmarks[26].y * h],
#                 'left_ankle': [landmarks[27].x * w, landmarks[27].y * h],
#                 'right_ankle': [landmarks[28].x * w, landmarks[28].y * h]
#             }
#
#             # Calculate joint angles
#             angles = {
#                 'left_elbow': calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist']),
#                 'right_elbow': calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist']),
#                 'left_knee': calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle']),
#                 'right_knee': calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle']),
#                 'left_shoulder': calculate_angle(points['left_elbow'], points['left_shoulder'], points['left_hip']),
#                 'right_shoulder': calculate_angle(points['right_elbow'], points['right_shoulder'], points['right_hip']),
#                 'left_hip': calculate_angle(points['left_knee'], points['left_hip'], points['left_shoulder']),
#                 'right_hip': calculate_angle(points['right_knee'], points['right_hip'], points['right_shoulder']),
#             }
#
#             # Annotate video frame
#             for joint, angle in angles.items():
#                 position = tuple(map(int, points[joint]))
#                 # Ensure the angle is an integer and format correctly
#                 angle_text = f"{int(round(angle))}°"
#                 cv2.putText(frame, angle_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#             # Detect injuries
#             detected_injuries = []
#             risk_probabilities = []
#             for injury, ranges in basketball_injury_angle_ranges.items():
#                 matching_joints = sum(is_within_range(angles[joint], ranges[joint]) for joint in ranges.keys())
#                 risk_probability = (matching_joints / len(ranges)) * 100
#                 if risk_probability > 0:
#                     detected_injuries.append(injury)
#                     risk_probabilities.append(risk_probability)
#
#
#
#             # Draw risk bar
#             bar_x, bar_y = 50, 50
#             bar_width, bar_height = 300, 20
#             risk_level = int(max(risk_probabilities) if risk_probabilities else 0)
#             cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 255), 2)
#             cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int((bar_width * risk_level) / 100), bar_y + bar_height),
#                           (0, 255, 0), -1)
#             cv2.putText(frame, f"Risk: {risk_level}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (255, 255, 255), 2)
#
#             # Save detected injuries to CSV
#             if detected_injuries:
#                 csv_writer.writerow([timestamp, frame_count, ", ".join(detected_injuries)])
#
#             # Write the frame to the output video
#             out_video.write(frame)
#
# cap.release()
#
# print(f"Injury detection completed. Results saved to {output_csv_path}.")



# import csv
# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# from datetime import datetime
#
# # Initialize MediaPipe pose with optimized settings
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,  # Dynamic video processing
#     model_complexity=2,  # Maximum detail level
#     smooth_landmarks=True,  # Enable landmark smoothing
#     enable_segmentation=True,  # Enable segmentation
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
# mp_draw = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Enhanced visualization settings
# custom_drawing_specs = mp_draw.DrawingSpec(
#     color=(0, 255, 0),  # Green color for landmarks
#     thickness=2,
#     circle_radius=2
# )
#
# custom_connection_specs = mp_draw.DrawingSpec(
#     color=(255, 255, 0),  # Yellow color for connections
#     thickness=2
# )
#
# #Define comprehensive injury thresholds for sports-specific motions
# basketball_injury_angle_ranges = {
#     "Ankle Sprain": {
#         "left_shoulder": (0, 3), "right_shoulder": (87, 93),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (7, 13), "right_knee": (77, 83),
#         "left_hip": (27, 33), "right_hip": (47, 53)
#     },
#     "ACL Tear": {
#         "left_shoulder": (77, 83), "right_shoulder": (87, 93),
#         "left_elbow": (0, 3), "right_elbow": (7, 13),
#         "left_knee": (157, 163), "right_knee": (117, 123),
#         "left_hip": (47, 53), "right_hip": (22, 28)
#     },
#     "MCL Tear": {
#         "left_shoulder": (47, 53), "right_shoulder": (57, 63),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (27, 33), "right_knee": (157, 163),
#         "left_hip": (17, 23), "right_hip": (37, 43)
#     },
#     "Achilles Tendon Rupture": {
#         "left_shoulder": (0, 3), "right_shoulder": (7, 13),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (57, 63), "right_knee": (47, 53),
#         "left_hip": (27, 33), "right_hip": (17, 23)
#     },
#     "Rotator Cuff Tear": {
#         "left_shoulder": (3, 6), "right_shoulder": (117, 123),
#         "left_elbow": (7, 13), "right_elbow": (27, 33),
#         "left_knee": (17, 23), "right_knee": (22, 28),
#         "left_hip": (7, 13), "right_hip": (37, 43)
#     },
#     "Labrum Tear (Shoulder)": {
#         "left_shoulder": (177, 183), "right_shoulder": (27, 33),
#         "left_elbow": (17, 23), "right_elbow": (47, 53),
#         "left_knee": (0, 3), "right_knee": (0, 3),
#         "left_hip": (87, 93), "right_hip": (12, 18)
#     },
#     "Hamstring Strain": {
#         "left_shoulder": (7, 13), "right_shoulder": (57, 63),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (7, 13), "right_knee": (27, 33),
#         "left_hip": (87, 93), "right_hip": (47, 53)
#     },
#     "Quadriceps Contusion": {
#         "left_shoulder": (7, 13), "right_shoulder": (87, 93),
#         "left_elbow": (17, 23), "right_elbow": (7, 13),
#         "left_knee": (77, 83), "right_knee": (87, 93),
#         "left_hip": (37, 43), "right_hip": (57, 63)
#     },
#     "Patellar Tendinitis": {
#         "left_shoulder": (57, 63), "right_shoulder": (67, 73),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (77, 83), "right_knee": (87, 93),
#         "left_hip": (47, 53), "right_hip": (57, 63)
#     },
#     "Wrist Fracture": {
#         "left_shoulder": (97, 103), "right_shoulder": (127, 133),
#         "left_elbow": (0, 3), "right_elbow": (147, 153),
#         "left_knee": (17, 23), "right_knee": (27, 33),
#         "left_hip": (7, 13), "right_hip": (37, 43)
#     },
#     "Finger Dislocation": {
#         "left_shoulder": (87, 93), "right_shoulder": (47, 53),
#         "left_elbow": (0, 3), "right_elbow": (97, 103),
#         "left_knee": (27, 33), "right_knee": (47, 53),
#         "left_hip": (12, 18), "right_hip": (17, 23)
#     },
#     "Labral Tear (Hip)": {
#         "left_shoulder": (7, 13), "right_shoulder": (0, 3),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (57, 63), "right_knee": (87, 93),
#         "left_hip": (97, 103), "right_hip": (27, 33)
#     },
#     "Concussion": {
#         "left_shoulder": (0, 3), "right_shoulder": (0, 3),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (0, 3), "right_knee": (0, 3),
#         "left_hip": (0, 3), "right_hip": (0, 3)  # Neutral angles
#     },
#     "Hip Pointer": {
#         "left_shoulder": (7, 13), "right_shoulder": (27, 33),
#         "left_elbow": (0, 3), "right_elbow": (7, 13),
#         "left_knee": (67, 73), "right_knee": (77, 83),
#         "left_hip": (127, 133), "right_hip": (97, 103)
#     },
#     "Calf Strain": {
#         "left_shoulder": (0, 3), "right_shoulder": (17, 23),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (27, 33), "right_knee": (47, 53),
#         "left_hip": (17, 23), "right_hip": (37, 43)
#     },
#     "Plantar Fasciitis": {
#         "left_shoulder": (7, 13), "right_shoulder": (0, 3),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (17, 23), "right_knee": (27, 33),
#         "left_hip": (47, 53), "right_hip": (57, 63)
#     },
#     "Back Spasm": {
#         "left_shoulder": (27, 33), "right_shoulder": (27, 33),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (47, 53), "right_knee": (57, 63),
#         "left_hip": (67, 73), "right_hip": (77, 83)
#     },
#     "Shoulder Impingement": {
#         "left_shoulder": (107, 113), "right_shoulder": (27, 33),
#         "left_elbow": (57, 63), "right_elbow": (67, 73),
#         "left_knee": (0, 3), "right_knee": (0, 3),
#         "left_hip": (47, 53), "right_hip": (57, 63)
#     },
#     "Stress Fracture": {
#         "left_shoulder": (7, 13), "right_shoulder": (7, 13),
#         "left_elbow": (0, 3), "right_elbow": (0, 3),
#         "left_knee": (87, 93), "right_knee": (47, 53),
#         "left_hip": (27, 33), "right_hip": (77, 83)
#     },
#     "Knee Bursitis": {
#         "left_shoulder": (27, 33), "right_shoulder": (17, 23),
#         "left_elbow": (7, 13), "right_elbow": (0, 3),
#         "left_knee": (97, 103), "right_knee": (87, 93),
#         "left_hip": (47, 53), "right_hip": (57, 63)
#     }
# }
#
#
#
# def calculate_angle(a, b, c):
#     """
#     Calculate angle between three points with enhanced precision
#     Args:
#         a: First point coordinates [x, y]
#         b: Middle point coordinates [x, y] (vertex)
#         c: Last point coordinates [x, y]
#     Returns:
#         angle: Angle in degrees with high precision
#     """
#     try:
#         a = np.array([float(a[0]), float(a[1])], dtype=np.float64)
#         b = np.array([float(b[0]), float(b[1])], dtype=np.float64)
#         c = np.array([float(c[0]), float(c[1])], dtype=np.float64)
#
#         # Calculate vectors
#         ba = a - b
#         bc = c - b
#
#         # Calculate magnitudes
#         ba_norm = np.linalg.norm(ba)
#         bc_norm = np.linalg.norm(bc)
#
#         # Check for zero vectors with high precision threshold
#         if ba_norm < 1e-10 or bc_norm < 1e-10:
#             return 0
#
#         # Calculate angle using dot product
#         cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
#         cosine = np.clip(cosine, -1.0, 1.0)  # Ensure value is in valid range
#         angle = np.degrees(np.arccos(cosine))
#
#         return round(angle, 2)  # Return with 2 decimal precision
#
#     except Exception as e:
#         print(f"Error calculating angle: {e}")
#         return 0
#
#
# def is_within_range(angle, range_tuple):
#     """
#     Check if angle falls within specified range with inclusive boundaries
#     Args:
#         angle: Angle to check
#         range_tuple: (min_angle, max_angle) tuple
#     Returns:
#         bool: True if angle is within range, False otherwise
#     """
#     return range_tuple[0] <= angle <= range_tuple[1]
#
#
# def get_bounding_box(landmarks, frame_shape, padding_factor=0.1):
#     """
#     Calculate dynamic bounding box with adaptive padding
#     Args:
#         landmarks: List of detected pose landmarks
#         frame_shape: Shape of the video frame
#         padding_factor: Factor to determine padding size (default 0.1 = 10%)
#     Returns:
#         tuple: (x_min, y_min, x_max, y_max) or None if no valid landmarks
#     """
#     h, w = frame_shape[:2]
#
#     # Collect coordinates of visible landmarks
#     x_coordinates = []
#     y_coordinates = []
#
#     for landmark in landmarks:
#         if landmark.visibility > 0.6:  # Only use highly visible landmarks
#             x_coordinates.append(landmark.x * w)
#             y_coordinates.append(landmark.y * h)
#
#     if not x_coordinates or not y_coordinates:
#         return None
#
#     # Calculate bounds
#     x_min, x_max = min(x_coordinates), max(x_coordinates)
#     y_min, y_max = min(y_coordinates), max(y_coordinates)
#
#     # Calculate dynamic padding
#     box_width = x_max - x_min
#     box_height = y_max - y_min
#
#     padding_x = box_width * padding_factor
#     padding_y = box_height * padding_factor
#
#     # Apply padding with boundary checks
#     x_min = max(0, int(x_min - padding_x))
#     x_max = min(w, int(x_max + padding_x))
#     y_min = max(0, int(y_min - padding_y))
#     y_max = min(h, int(y_max + padding_y))
#
#     return (x_min, y_min, x_max, y_max)
#
#
# def calculate_motion_speed(current_points, previous_points):
#     """
#     Calculate motion speed between frames
#     Args:
#         current_points: Current frame landmark points
#         previous_points: Previous frame landmark points
#     Returns:
#         float: Average motion speed
#     """
#     if not previous_points:
#         return 0.0
#
#     speeds = []
#     for point_name in current_points:
#         if point_name in previous_points:
#             current = np.array(current_points[point_name][:2])
#             previous = np.array(previous_points[point_name][:2])
#             speed = np.linalg.norm(current - previous)
#             speeds.append(speed)
#
#     return np.mean(speeds) if speeds else 0.0
#
#
# def process_video(input_video_path, output_csv_path, output_video_path):
#     """
#     Process video with enhanced motion tracking and injury detection
#     Args:
#         input_video_path: Path to input video file
#         output_csv_path: Path to output CSV file for analysis data
#         output_video_path: Path to output processed video file
#     """
#     print(f"Starting video processing at {datetime.now()}")
#
#     cap = cv2.VideoCapture(input_video_path)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Optimize for fast movement
#
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
#
#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     # Initialize video writer with high quality
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_video = cv2.VideoWriter(
#         output_video_path,
#         fourcc,
#         fps,
#         (frame_width, frame_height),
#         isColor=True
#     )
#
#     # Initialize CSV writer with comprehensive data logging
#     with open(output_csv_path, mode='w', newline='') as csvfile:
#         csv_writer = csv.writer(csvfile)
#         csv_writer.writerow([
#             'Timestamp',
#             'Frame',
#             'Bounding Box',
#             'Joint Angles',
#             'Confidence Scores',
#             'Detected Injuries',
#             'Risk Level',
#             'Motion Speed',
#             'Overall Assessment'
#         ])
#
#         frame_count = 0
#         previous_points = None
#
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 break
#
#             frame_count += 1
#             timestamp = f"{int((frame_count / fps) // 60):02}:{int((frame_count / fps) % 60):02}.{int((frame_count % fps) * 100 / fps):02}"
#
#             # Process frame with MediaPipe
#             frame.flags.writeable = False
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(rgb_frame)
#             frame.flags.writeable = True
#
#             if results.pose_landmarks:
#                 landmarks = results.pose_landmarks.landmark
#                 bbox = get_bounding_box(landmarks, frame.shape)
#
#                 if bbox:
#                     x_min, y_min, x_max, y_max = bbox
#
#                     # Draw enhanced bounding box
#                     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#
#                     # Process landmarks and calculate points
#                     h, w = frame.shape[:2]
#                     current_points = {}
#                     landmark_indices = {
#                         # Upper body
#                         'nose': 0,
#                         'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
#                         'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
#                         'left_ear': 7, 'right_ear': 8,
#                         'mouth_left': 9, 'mouth_right': 10,
#                         'left_shoulder': 11, 'right_shoulder': 12,
#                         'left_elbow': 13, 'right_elbow': 14,
#                         'left_wrist': 15, 'right_wrist': 16,
#                         'left_pinky': 17, 'right_pinky': 18,
#                         'left_index': 19, 'right_index': 20,
#                         'left_thumb': 21, 'right_thumb': 22,
#
#                         # Lower body
#                         'left_hip': 23, 'right_hip': 24,
#                         'left_knee': 25, 'right_knee': 26,
#                         'left_ankle': 27, 'right_ankle': 28,
#                         'left_heel': 29, 'right_heel': 30,
#                         'left_foot_index': 31, 'right_foot_index': 32
#                     }
#
#                     # Extract all landmark points with enhanced visibility check
#                     for name, idx in landmark_indices.items():
#                         landmark = landmarks[idx]
#                         x = landmark.x * w
#                         y = landmark.y * h
#                         visibility = landmark.visibility
#                         if visibility > 0.5:  # Visibility threshold
#                             current_points[name] = [x, y, visibility]
#
#                     # Calculate all joint angles
#                     angles = {}
#                     confidence_scores = {}
#                     angle_definitions = {
#                         'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
#                         'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
#                         'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
#                         'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
#                         'left_hip': ('left_knee', 'left_hip', 'left_shoulder'),
#                         'right_hip': ('right_knee', 'right_hip', 'right_shoulder'),
#                         'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
#                         'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
#                         'neck': ('left_shoulder', 'nose', 'right_shoulder'),
#                         'left_wrist': ('left_elbow', 'left_wrist', 'left_index'),
#                         'right_wrist': ('right_elbow', 'right_wrist', 'right_index'),
#                         'left_ankle': ('left_knee', 'left_ankle', 'left_foot_index'),
#                         'right_ankle': ('right_knee', 'right_ankle', 'right_foot_index')
#                     }
#
#                     # Calculate angles based on defined joint relationships
#                     for angle_name, (p1, p2, p3) in angle_definitions.items():
#                         if all(p in current_points for p in (p1, p2, p3)):
#                             angle = calculate_angle(
#                                 current_points[p1][:2],
#                                 current_points[p2][:2],
#                                 current_points[p3][:2]
#                             )
#                             angles[angle_name] = angle
#
#                             # Calculate confidence score for this angle
#                             confidence = min(
#                                 current_points[p1][2],
#                                 current_points[p2][2],
#                                 current_points[p3][2]
#                             )
#                             confidence_scores[angle_name] = round(confidence * 100, 2)
#
#                     # Detect injuries and calculate risks
#                     detected_injuries = []
#                     risk_probabilities = []
#                     injury_details = []
#
#                     for injury_name, angle_ranges in basketball_injury_angle_ranges.items():
#                         matching_joints = 0
#                         total_relevant_joints = len(angle_ranges)
#                         problematic_angles = []
#
#                         for joint, range_tuple in angle_ranges.items():
#                             if joint in angles and is_within_range(angles[joint], range_tuple):
#                                 matching_joints += 1
#                                 problematic_angles.append(f"{joint}: {angles[joint]:.1f}°")
#
#                         if matching_joints > 0:
#                             risk_probability = (matching_joints / total_relevant_joints) * 100
#                             detected_injuries.append(injury_name)
#                             risk_probabilities.append(risk_probability)
#                             injury_details.append({
#                                 'name': injury_name,
#                                 'risk': risk_probability,
#                                 'problematic_angles': problematic_angles
#                             })
#
#                     # Calculate motion speed if we have previous points
#                     motion_speed = calculate_motion_speed(current_points, previous_points)
#                     previous_points = current_points.copy()
#
#                     # Draw skeleton with enhanced visualization
#                     mp_draw.draw_landmarks(
#                         frame,
#                         results.pose_landmarks,
#                         mp_pose.POSE_CONNECTIONS,
#                         custom_drawing_specs,
#                         custom_connection_specs
#                     )
#
#                     # Draw angles on frame with enhanced visibility
#                     for joint, angle in angles.items():
#                         if joint in current_points:
#                             position = tuple(map(int, current_points[joint][:2]))
#                             confidence = confidence_scores[joint]
#
#                             # Color-coded confidence visualization
#                             color = (
#                                 int(255 * (1 - confidence / 100)),
#                                 int(255 * (confidence / 100)),
#                                 0
#                             )
#
#                             # Draw angle and confidence
#                             cv2.putText(
#                                 frame,
#                                 f"{angle:.1f}° ({confidence}%)",
#                                 position,
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.5,
#                                 color,
#                                 2
#                             )
#
#                     # Draw risk visualization
#                     if risk_probabilities:
#                         max_risk = max(risk_probabilities)
#                         bar_x = x_min
#                         bar_y = y_min - 40
#                         bar_width = x_max - x_min
#                         bar_height = 20
#
#                         # Draw risk bar background
#                         cv2.rectangle(frame, (bar_x, bar_y),
#                                       (bar_x + bar_width, bar_y + bar_height),
#                                       (0, 0, 255), 2)
#
#                         # Draw risk level fill
#                         fill_width = int((bar_width * max_risk) / 100)
#                         cv2.rectangle(frame, (bar_x, bar_y),
#                                       (bar_x + fill_width, bar_y + bar_height),
#                                       (0, 255, 0), -1)
#
#                         # Add detailed risk information
#                         cv2.putText(frame,
#                                     f"Risk: {int(max_risk)}% - {', '.join(detected_injuries)}",
#                                     (bar_x, bar_y - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX,
#                                     0.5,
#                                     (255, 255, 255),
#                                     2)
#
#                     # Generate overall assessment
#                     assessment = "Normal"
#                     if detected_injuries:
#                         if max(risk_probabilities) > 75:
#                             assessment = "High Risk - Immediate Attention Required"
#                         elif max(risk_probabilities) > 50:
#                             assessment = "Moderate Risk - Monitor Closely"
#                         else:
#                             assessment = "Low Risk - Continue Monitoring"
#
#                     # Record data to CSV
#                     csv_writer.writerow([
#                         timestamp,
#                         frame_count,
#                         f"({x_min}, {y_min}, {x_max}, {y_max})",
#                         str(angles),
#                         str(confidence_scores),
#                         str(detected_injuries),
#                         f"{max(risk_probabilities) if risk_probabilities else 0:.2f}",
#                         f"{motion_speed:.2f}",
#                         assessment
#                     ])
#
#             # Write processed frame
#             out_video.write(frame)
#
#             # Show progress
#             if frame_count % 30 == 0:
#                 progress = (frame_count / total_frames) * 100
#                 print(f"Progress: {progress:.1f}% at {datetime.now()}")
#
#     # Cleanup and finalize
#     cap.release()
#     out_video.release()
#     print(f"Processing completed at {datetime.now()}")
#     print(f"Results saved to {output_csv_path}")
#
#
# if __name__ == "__main__":
#     # Define paths for processing
#     input_video_path = 'video_playback.mp4'  # Replace with your video file
#     output_csv_path = 'pose_analysis_detailed.csv'
#     output_video_path = 'output_analyzed.mp4'
#
#     # Process the video
#     process_video(input_video_path, output_csv_path, output_video_path)
#
#
#


import csv
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import json

# Initialize MediaPipe pose with optimized settings for basketball
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Enhanced visualization settings
custom_drawing_specs = mp_draw.DrawingSpec(
    color=(0, 255, 0),
    thickness=2,
    circle_radius=2
)

custom_connection_specs = mp_draw.DrawingSpec(
    color=(255, 255, 0),
    thickness=2
)

# Basketball-specific injury thresholds
injury_angle_ranges = {
    "Ankle Sprain": {
        "left_ankle": (0, 15),
        "right_ankle": (0, 15),
        "left_knee": (10, 20),
        "right_knee": (10, 20)
    },
    "ACL Injury": {
        "left_knee": (20, 35),
        "right_knee": (20, 35),
        "left_hip": (30, 45),
        "right_hip": (30, 45)
    },
    "Jumper's Knee": {
        "left_knee": (80, 95),
        "right_knee": (80, 95),
        "left_ankle": (60, 75),
        "right_ankle": (60, 75)
    },
    "Lower Back Strain": {
        "left_hip": (85, 100),
        "right_hip": (85, 100),
        "left_shoulder": (75, 90),
        "right_shoulder": (75, 90)
    }
}

# Comprehensive basketball motion patterns
basketball_motion_patterns = {
    "Jump_Shot": {
        "setup_phase": {
            "right_knee": (110, 130),
            "left_knee": (110, 130),
            "right_hip": (100, 120),
            "left_hip": (100, 120),
            "right_ankle": (70, 80),
            "left_ankle": (70, 80)
        },
        "shooting_phase": {
            "right_elbow": (85, 95),
            "right_shoulder": (140, 160),
            "right_wrist": (70, 90),
            "right_hip": (160, 180),
            "right_knee": (160, 180),
            "right_ankle": (140, 160)
        },
        "follow_through": {
            "right_wrist": (160, 180),
            "right_elbow": (160, 180),
            "right_shoulder": (160, 180)
        }
    },
    "Layup": {
        "approach": {
            "leading_knee": (100, 120),
            "trailing_knee": (120, 140),
            "hip_flexion": (80, 100),
            "ankle_flexion": (60, 80)
        },
        "takeoff": {
            "driving_knee": (45, 65),
            "plant_leg": (140, 160),
            "hip_extension": (150, 170),
            "shoulder_flexion": (120, 140)
        },
        "release": {
            "shooting_elbow": (80, 100),
            "wrist_angle": (50, 70),
            "shoulder_height": (150, 170)
        }
    },
    "Defensive_Stance": {
        "basic_position": {
            "knee_flexion": (120, 140),
            "hip_flexion": (120, 140),
            "ankle_dorsiflexion": (70, 90),
            "shoulder_abduction": (10, 30)
        },
        "lateral_movement": {
            "push_off_knee": (100, 120),
            "plant_knee": (110, 130),
            "hip_abduction": (20, 40),
            "ankle_eversion": (10, 20)
        }
    }
}

# Performance metrics for analysis
performance_metrics = {
    "Jump_Shot": {
        "release_height": {
            "optimal_range": (2.1, 2.4),
            "release_time": (0.3, 0.5)
        },
        "arc_angle": {
            "optimal_range": (45, 55),
            "backspin": (120, 180)
        }
    },
    "Defensive_Movement": {
        "lateral_speed": {
            "optimal_range": (3.5, 4.5),
            "recovery_time": (1.2, 1.8)
        },
        "stance_width": {
            "optimal_range": (1.1, 1.4),
            "center_of_gravity": (0.5, 0.6)
        }
    }
}


def calculate_angle(a, b, c):
    """Calculate angle between three points with enhanced precision"""
    try:
        a = np.array([float(a[0]), float(a[1])], dtype=np.float64)
        b = np.array([float(b[0]), float(b[1])], dtype=np.float64)
        c = np.array([float(c[0]), float(c[1])], dtype=np.float64)

        ba = a - b
        bc = c - b

        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)

        if ba_norm < 1e-10 or bc_norm < 1e-10:
            return 0

        cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))

        return round(angle, 2)
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0


def is_within_range(angle, range_tuple):
    """Check if angle falls within specified range"""
    return range_tuple[0] <= angle <= range_tuple[1]


def detect_motion_phase(angles, phase_patterns):
    """Detect specific phase of a basketball motion"""
    matching_angles = 0
    total_angles = len(phase_patterns)

    for joint, (min_angle, max_angle) in phase_patterns.items():
        if joint in angles and min_angle <= angles[joint] <= max_angle:
            matching_angles += 1

    return matching_angles / total_angles if total_angles > 0 else 0




def analyze_layup_form(angles, current_points):
    """
    Analyze basketball layup form and provide feedback
    Args:
        angles: Dictionary of calculated joint angles
        current_points: Dictionary of current landmark points
    Returns:
        dict: Analysis results and recommendations
    """
    analysis = {
        "form_quality": "Good",
        "approach_angle": "Good",
        "knee_drive": "Good",
        "ball_position": "Good",
        "timing": "Good",
        "recommendations": []
    }

    # Check approach angle
    if "right_hip_angle" in angles:
        hip_angle = angles["right_hip_angle"]
        if not (45 <= hip_angle <= 65):
            analysis["approach_angle"] = "Needs Improvement"
            analysis["recommendations"].append(
                "Adjust approach angle to 45-65 degrees for better momentum"
            )

    # Check knee drive
    if "right_knee_angle" in angles:
        knee_angle = angles["right_knee_angle"]
        if not (70 <= knee_angle <= 90):
            analysis["knee_drive"] = "Needs Improvement"
            analysis["recommendations"].append(
                "Drive knee higher for better elevation"
            )

    # Check ball position
    if all(key in current_points for key in ["right_wrist", "right_shoulder"]):
        wrist_height = current_points["right_wrist"][1]
        shoulder_height = current_points["right_shoulder"][1]
        if wrist_height > shoulder_height:
            analysis["ball_position"] = "Needs Improvement"
            analysis["recommendations"].append(
                "Keep ball higher during approach"
            )

    # Overall assessment
    if len(analysis["recommendations"]) > 2:
        analysis["form_quality"] = "Needs Significant Improvement"
    elif len(analysis["recommendations"]) > 0:
        analysis["form_quality"] = "Needs Minor Adjustments"

    return analysis


def analyze_defensive_form(angles, current_points):
    """
    Analyze defensive stance and movement
    Args:
        angles: Dictionary of calculated joint angles
        current_points: Dictionary of current landmark points
    Returns:
        dict: Analysis results and recommendations
    """
    analysis = {
        "form_quality": "Good",
        "stance_width": "Good",
        "knee_bend": "Good",
        "back_straight": "Good",
        "balance": "Good",
        "recommendations": []
    }

    # Check stance width
    if all(key in current_points for key in ["left_ankle", "right_ankle", "right_hip"]):
        left_ankle = np.array(current_points["left_ankle"][:2])
        right_ankle = np.array(current_points["right_ankle"][:2])
        stance_width = np.linalg.norm(left_ankle - right_ankle)
        hip_width = np.linalg.norm(np.array(current_points["right_hip"][:2]) - np.array(current_points["left_hip"][:2]))

        if stance_width < hip_width * 1.5:
            analysis["stance_width"] = "Too Narrow"
            analysis["recommendations"].append(
                "Widen stance to shoulder width plus 6-8 inches"
            )
        elif stance_width > hip_width * 2.5:
            analysis["stance_width"] = "Too Wide"
            analysis["recommendations"].append(
                "Narrow stance for better mobility"
            )

    # Check knee bend
    for side in ["left_knee_angle", "right_knee_angle"]:
        if side in angles:
            knee_angle = angles[side]
            if knee_angle < 100:
                analysis["knee_bend"] = "Too Deep"
                analysis["recommendations"].append(
                    "Reduce knee bend for better mobility"
                )
            elif knee_angle > 140:
                analysis["knee_bend"] = "Too Straight"
                analysis["recommendations"].append(
                    "Increase knee bend for better defensive position"
                )

    # Check back alignment
    if "spine_angle" in angles:
        spine_angle = angles["spine_angle"]
        if not (80 <= spine_angle <= 100):
            analysis["back_straight"] = "Needs Improvement"
            analysis["recommendations"].append(
                "Keep back straight and chest up"
            )

    # Check balance
    if all(key in current_points for key in ["nose", "right_hip", "right_ankle"]):
        nose = np.array(current_points["nose"][:2])
        hip = np.array(current_points["right_hip"][:2])
        ankle = np.array(current_points["right_ankle"][:2])

        vertical_alignment = abs(nose[0] - hip[0]) / hip_width
        if vertical_alignment > 0.3:
            analysis["balance"] = "Off Center"
            analysis["recommendations"].append(
                "Keep weight centered between feet"
            )

    # Overall assessment
    if len(analysis["recommendations"]) > 2:
        analysis["form_quality"] = "Needs Significant Improvement"
    elif len(analysis["recommendations"]) > 0:
        analysis["form_quality"] = "Needs Minor Adjustments"

    return analysis


def analyze_shooting_form(angles, current_points):
    """
    Analyze basketball shooting form and provide feedback
    Args:
        angles: Dictionary of calculated joint angles
        current_points: Dictionary of current landmark points
    Returns:
        dict: Analysis results and recommendations
    """
    analysis = {
        "form_quality": "Good",
        "elbow_alignment": "Good",
        "release_angle": "Good",
        "follow_through": "Good",
        "base": "Good",
        "recommendations": []
    }

    # Check elbow alignment
    if "right_elbow_angle" in angles:
        elbow_angle = angles["right_elbow_angle"]
        if not (85 <= elbow_angle <= 95):
            analysis["elbow_alignment"] = "Needs Improvement"
            analysis["recommendations"].append(
                "Keep elbow at 90 degrees during shot preparation"
            )

    # Check shooting arm alignment
    if "shooting_arm_alignment" in angles:
        arm_angle = angles["shooting_arm_alignment"]
        if not (85 <= arm_angle <= 95):
            analysis["arm_alignment"] = "Needs Improvement"
            analysis["recommendations"].append(
                "Align shooting arm vertically under ball"
            )

    # Check release angle
    if "release_angle" in angles:
        release = angles["release_angle"]
        if not (45 <= release <= 55):
            analysis["release_angle"] = "Needs Improvement"
            analysis["recommendations"].append(
                "Adjust release angle to 45-55 degrees"
            )

    # Check follow through
    if "follow_through_angle" in angles:
        follow = angles["follow_through_angle"]
        if not (160 <= follow <= 180):
            analysis["follow_through"] = "Needs Improvement"
            analysis["recommendations"].append(
                "Complete follow through with full extension"
            )

    # Check base and balance
    if "right_knee_angle" in angles and "left_knee_angle" in angles:
        knee_diff = abs(angles["right_knee_angle"] - angles["left_knee_angle"])
        if knee_diff > 15:
            analysis["base"] = "Unbalanced"
            analysis["recommendations"].append(
                "Maintain balanced stance during shot"
            )

    # Overall assessment
    if len(analysis["recommendations"]) > 2:
        analysis["form_quality"] = "Needs Significant Improvement"
    elif len(analysis["recommendations"]) > 0:
        analysis["form_quality"] = "Needs Minor Adjustments"

    return analysis


def calculate_motion_metrics(current_points, previous_points, frame_time):
    """Calculate performance metrics for basketball movements"""
    metrics = {
        "vertical_displacement": 0,
        "lateral_speed": 0,
        "movement_smoothness": 0
    }

    if not previous_points or not current_points:
        return metrics

    try:
        # Calculate vertical displacement
        if "nose" in current_points and "nose" in previous_points:
            metrics["vertical_displacement"] = (
                    current_points["nose"][1] - previous_points["nose"][1]
            )

        # Calculate lateral speed
        if "hip" in current_points and "hip" in previous_points:
            lateral_distance = abs(
                current_points["hip"][0] - previous_points["hip"][0]
            )
            metrics["lateral_speed"] = lateral_distance / frame_time

        # Calculate movement smoothness
        # Using normalized jerk score
        if "right_wrist" in current_points and "right_wrist" in previous_points:
            jerk = calculate_jerk(
                current_points["right_wrist"],
                previous_points["right_wrist"],
                frame_time
            )
            metrics["movement_smoothness"] = 1 / (1 + jerk)

    except Exception as e:
        print(f"Error calculating metrics: {e}")

    return metrics


def calculate_jerk(current_point, previous_point, time_delta):
    """Calculate jerk (rate of change of acceleration)"""
    try:
        velocity = (np.array(current_point[:2]) - np.array(previous_point[:2])) / time_delta
        acceleration = velocity / time_delta
        jerk = np.linalg.norm(acceleration) / time_delta
        return jerk
    except:
        return 0


def process_basketball_video(input_video_path, output_csv_path, output_video_path):
    """Process basketball video with motion analysis and performance tracking"""
    print(f"Starting basketball motion analysis at {datetime.now()}")

    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_time = 1.0 / fps

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (frame_width, frame_height),
        isColor=True
    )

    # Initialize CSV writer
    with open(output_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Timestamp',
            'Frame',
            'Detected_Motion',
            'Motion_Phase',
            'Joint_Angles',
            'Form_Analysis',
            'Performance_Metrics',
            'Injury_Risks',
            'Recommendations'
        ])

        frame_count = 0
        previous_points = None

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            timestamp = f"{int((frame_count / fps) // 60):02}:{int((frame_count / fps) % 60):02}.{int((frame_count % fps) * 100 / fps):02}"

            # Process frame with MediaPipe
            frame.flags.writeable = False
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            frame.flags.writeable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w = frame.shape[:2]
                current_points = {}

                # Define all landmark indices for basketball analysis
                landmark_indices = {
                    # Upper Body Landmarks
                    'nose': 0,
                    'left_eye_inner': 1,
                    'left_eye': 2,
                    'left_eye_outer': 3,
                    'right_eye_inner': 4,
                    'right_eye': 5,
                    'right_eye_outer': 6,
                    'left_ear': 7,
                    'right_ear': 8,
                    'mouth_left': 9,
                    'mouth_right': 10,

                    # Arms and Shoulders
                    'left_shoulder': 11,
                    'right_shoulder': 12,
                    'left_elbow': 13,
                    'right_elbow': 14,
                    'left_wrist': 15,
                    'right_wrist': 16,
                    'left_pinky': 17,
                    'right_pinky': 18,
                    'left_index': 19,
                    'right_index': 20,
                    'left_thumb': 21,
                    'right_thumb': 22,

                    # Lower Body
                    'left_hip': 23,
                    'right_hip': 24,
                    'left_knee': 25,
                    'right_knee': 26,
                    'left_ankle': 27,
                    'right_ankle': 28,
                    'left_heel': 29,
                    'right_heel': 30,
                    'left_foot_index': 31,
                    'right_foot_index': 32
                }

                # Extract all landmark points with position and visibility
                for name, idx in landmark_indices.items():
                    landmark = landmarks[idx]
                    x = landmark.x * w
                    y = landmark.y * h
                    visibility = landmark.visibility
                    if visibility > 0.5:
                        current_points[name] = [x, y, visibility]

                # Calculate all defined joint angles
                angles = {}
                # Define comprehensive angle calculations for basketball movements
                angle_definitions = {
                    # Shooting Form Angles
                    'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_wrist'),
                    'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_wrist'),
                    'right_shoulder_angle': ('right_elbow', 'right_shoulder', 'right_hip'),
                    'left_shoulder_angle': ('left_elbow', 'left_shoulder', 'left_hip'),

                    # Lower Body Angles
                    'right_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
                    'left_knee_angle': ('left_hip', 'left_knee', 'left_ankle'),
                    'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'),
                    'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
                    'right_ankle_angle': ('right_knee', 'right_ankle', 'right_foot_index'),
                    'left_ankle_angle': ('left_knee', 'left_ankle', 'left_foot_index'),

                    # Body Alignment Angles
                    'spine_angle': ('nose', 'right_hip', 'right_knee'),
                    'neck_angle': ('right_ear', 'right_shoulder', 'right_hip'),
                    'torso_angle': ('right_shoulder', 'right_hip', 'right_knee'),

                    # Basketball-Specific Angles
                    'shooting_arm_alignment': ('right_shoulder', 'right_elbow', 'right_wrist'),
                    'follow_through_angle': ('right_elbow', 'right_wrist', 'right_index'),
                    'release_angle': ('right_elbow', 'right_wrist', 'right_index'),

                    # Defensive Stance Angles
                    'defensive_right_knee': ('right_hip', 'right_knee', 'right_ankle'),
                    'defensive_left_knee': ('left_hip', 'left_knee', 'left_ankle'),
                    'defensive_hips': ('left_hip', 'right_hip', 'right_knee'),

                    # Balance and Stability Angles
                    'base_width': ('left_ankle', 'right_hip', 'right_ankle'),
                    'center_of_mass': ('right_shoulder', 'right_hip', 'right_ankle'),

                    # Jump Shot Analysis
                    'jump_knee_angle': ('right_hip', 'right_knee', 'right_ankle'),
                    'jump_hip_angle': ('right_shoulder', 'right_hip', 'right_knee'),
                    'landing_knee_angle': ('right_hip', 'right_knee', 'right_ankle')
                }
                for angle_name, (p1, p2, p3) in angle_definitions.items():
                    if all(p in current_points for p in (p1, p2, p3)):
                        angle = calculate_angle(
                            current_points[p1][:2],
                            current_points[p2][:2],
                            current_points[p3][:2]
                        )
                        angles[angle_name] = angle

                # Detect basketball-specific motions and phases
                detected_motion = None
                motion_phase = None
                max_phase_confidence = 0

                for motion, phases in basketball_motion_patterns.items():
                    for phase, patterns in phases.items():
                        phase_confidence = detect_motion_phase(angles, patterns)
                        if phase_confidence > max_phase_confidence and phase_confidence > 0.7:
                            max_phase_confidence = phase_confidence
                            detected_motion = motion
                            motion_phase = phase

                # Perform detailed form analysis for detected motions
                form_analysis = None
                if detected_motion == "Jump_Shot":
                    form_analysis = analyze_shooting_form(angles, current_points)
                elif detected_motion == "Layup":
                    form_analysis = analyze_layup_form(angles, current_points)
                elif detected_motion == "Defensive_Stance":
                    form_analysis = analyze_defensive_form(angles, current_points)

                # Calculate performance metrics
                performance = calculate_motion_metrics(
                    current_points,
                    previous_points,
                    frame_time
                )

                # Assess injury risks based on current motion
                injury_risks = []
                for injury, ranges in injury_angle_ranges.items():
                    matching_joints = 0
                    affected_joints = []

                    for joint, range_tuple in ranges.items():
                        if joint in angles and is_within_range(angles[joint], range_tuple):
                            matching_joints += 1
                            affected_joints.append(joint)

                    risk_score = matching_joints / len(ranges)
                    if risk_score > 0.5:
                        injury_risks.append({
                            "type": injury,
                            "risk_score": risk_score,
                            "affected_joints": affected_joints
                        })

                # Generate action recommendations
                recommendations = []
                if form_analysis and "recommendations" in form_analysis:
                    recommendations.extend(form_analysis["recommendations"])

                for risk in injury_risks:
                    recommendations.append(
                        f"Warning: {risk['type']} risk detected. "
                        f"Monitor {', '.join(risk['affected_joints'])}"
                    )

                # Draw visualization overlays
                draw_frame_annotations(
                    frame,
                    current_points,
                    angles,
                    detected_motion,
                    motion_phase,
                    form_analysis,
                    injury_risks,
                    recommendations
                )

                # Record frame data to CSV
                csv_writer.writerow([
                    timestamp,
                    frame_count,
                    detected_motion,
                    motion_phase,
                    json.dumps(angles),
                    json.dumps(form_analysis) if form_analysis else "",
                    json.dumps(performance),
                    json.dumps(injury_risks),
                    "|".join(recommendations)
                ])

                previous_points = current_points.copy()

            # Write processed frame
            out_video.write(frame)

            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% at {datetime.now()}")

    # Cleanup and finalize
    cap.release()
    out_video.release()

    print(f"Results saved to {output_csv_path}")

    return output_video_path, output_csv_path



def draw_frame_annotations(frame, points, angles, detected_motion, motion_phase, form_analysis, injury_risks, recommendations):
    """Draw visual annotations on the frame with corrected point access"""
    h, w = frame.shape[:2]

    # Create a mapping between MediaPipe pose indices and landmark names
    idx_to_name = {
        0: 'nose',
        1: 'left_eye_inner',
        2: 'left_eye',
        3: 'left_eye_outer',
        4: 'right_eye_inner',
        5: 'right_eye',
        6: 'right_eye_outer',
        7: 'left_ear',
        8: 'right_ear',
        9: 'mouth_left',
        10: 'mouth_right',
        11: 'left_shoulder',
        12: 'right_shoulder',
        13: 'left_elbow',
        14: 'right_elbow',
        15: 'left_wrist',
        16: 'right_wrist',
        17: 'left_pinky',
        18: 'right_pinky',
        19: 'left_index',
        20: 'right_index',
        21: 'left_thumb',
        22: 'right_thumb',
        23: 'left_hip',
        24: 'right_hip',
        25: 'left_knee',
        26: 'right_knee',
        27: 'left_ankle',
        28: 'right_ankle',
        29: 'left_heel',
        30: 'right_heel',
        31: 'left_foot_index',
        32: 'right_foot_index'
    }

    # Draw skeleton
    if points:
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx_name = idx_to_name.get(connection[0])
            end_idx_name = idx_to_name.get(connection[1])

            if start_idx_name in points and end_idx_name in points:
                start_point = tuple(map(int, points[start_idx_name][:2]))
                end_point = tuple(map(int, points[end_idx_name][:2]))

                cv2.line(frame, start_point, end_point, (255, 255, 0), 2)
                cv2.circle(frame, start_point, 4, (0, 255, 0), -1)
                cv2.circle(frame, end_point, 4, (0, 255, 0), -1)

    # Draw motion detection info
    if detected_motion:
        cv2.putText(
            frame,
            f"Motion: {detected_motion} - Phase: {motion_phase}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Draw angle measurements
    y_offset = 60
    for joint, angle in angles.items():
        # Only draw angles for visible joints
        joint_name = joint.replace('_angle', '')  # Remove '_angle' suffix if present
        if joint_name in points:
            position = tuple(map(int, points[joint_name][:2]))
            cv2.putText(
                frame,
                f"{angle:.1f}°",
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Draw form analysis
    if form_analysis:
        y_offset = 60
        for key, value in form_analysis.items():
            if key != "recommendations":
                cv2.putText(
                    frame,
                    f"{key}: {value}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 0),
                    2
                )
                y_offset += 25

    # Draw injury risk warnings
    if injury_risks:
        y_offset = h - 120
        for risk in injury_risks:
            cv2.putText(
                frame,
                f"Risk: {risk['type']} ({risk['risk_score']:.2f})",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            y_offset += 25

    # Draw recommendations
    if recommendations:
        y_offset = h - 60
        for rec in recommendations[:2]:  # Show only top 2 recommendations
            cv2.putText(
                frame,
                rec[:50] + "..." if len(rec) > 50 else rec,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2
            )
            y_offset += 25


# if __name__ == "__main__":
#         # Define input/output paths
#     input_video_path = "videoplayback_basketball_2.mp4"  # Replace with your video file
#     output_csv_path = "basketball_analysis_2.csv"
#     output_video_path = "analyzed_basketball_2.mp4"

#         # Process the video
#     process_basketball_video(input_video_path, output_csv_path, output_video_path)



