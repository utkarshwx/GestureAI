import csv
import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Function to calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    bc = c - b
    angle = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    return np.degrees(angle)

# Function to get landmarks
def get_landmark(landmarks, id, width, height):
    if landmarks[id].visibility > 0.5:
        return [int(landmarks[id].x * width), int(landmarks[id].y * height)]
    return None

# Define maximum and minimum allowable angles for each joint
max_angles = {
    'left_elbow': 145, 'right_elbow': 145, 'left_knee': 160, 'right_knee': 160,
    'left_shoulder': 180, 'right_shoulder': 180, 'left_hip': 160, 'right_hip': 160
}
min_angles = {
    'left_elbow': 130, 'right_elbow': 130, 'left_knee': 140, 'right_knee': 140,
    'left_shoulder': 150, 'right_shoulder': 150, 'left_hip': 140, 'right_hip': 140
}

# Input video file and setup
input_video_path = 'test-video.mp4'  # Path to input video
output_video_path = 'processed_video.mp4'  # Path to save processed video
output_csv_path = 'exceeding_joints.csv'  # CSV file to save timestamps and joint data
output_images_dir = 'frames'  # Directory to save images
output_features_csv = 'engineered_features.csv'  # Output CSV file for engineered features

# Feature Engineering Function
def feature_engineering(input_csv, output_csv):
    # Load the existing CSV
    df = pd.read_csv(input_csv)

    # Extract frame-specific data for each joint
    df['Joint'] = df['Joint'].str.lower()  # Standardize joint naming

    # Derived Features
    derived_features = []
    for _, row in df.iterrows():
        joint = row['Joint']
        image_path = row['Image File']

        # Feature: Difference between current angle and mean allowed range
        max_angle = max_angles.get(joint, None)
        min_angle = min_angles.get(joint, None)
        if max_angle is not None and min_angle is not None:
            mean_angle = (max_angle + min_angle) / 2
            angle_difference = max_angle - min_angle
        else:
            mean_angle = np.nan
            angle_difference = np.nan

        # Feature: Symmetry (if paired joint exists)
        paired_joint = None
        if 'left' in joint:
            paired_joint = joint.replace('left', 'right')
        elif 'right' in joint:
            paired_joint = joint.replace('right', 'left')

        symmetry = None
        if paired_joint and paired_joint in df['Joint'].values:
            symmetry = 1  # Indicate symmetry is present
        else:
            symmetry = 0  # No symmetry found

        # Append new feature set for the current row
        derived_features.append({
            'Timestamp (mm:ss)': row['Timestamp (mm:ss)'],
            'Joint': joint,
            'Image File': image_path,
            'Mean Angle': mean_angle,
            'Angle Difference': angle_difference,
            'Symmetry': symmetry
        })

    # Create a DataFrame from the derived features
    features_df = pd.DataFrame(derived_features)

    # Save the new features to a CSV file
    features_df.to_csv(output_csv, index=False)
    print(f"Feature-engineered data saved to {output_csv}")

# Execute feature engineering
feature_engineering(output_csv_path, output_features_csv)



# Create output images directory if not exists
os.makedirs(output_images_dir, exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Open CSV file for writing
with open(output_csv_path, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Timestamp (mm:ss)', 'Joint', 'Image File'])

    print("Processing video. This may take some time...")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        timestamp_seconds = frame_count / fps
        timestamp = f"{int(timestamp_seconds // 60):02}:{int(timestamp_seconds % 60):02}"

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark
            h, w, _ = frame.shape

            points = {
                'left_shoulder': get_landmark(landmarks, 11, w, h),
                'right_shoulder': get_landmark(landmarks, 12, w, h),
                'left_elbow': get_landmark(landmarks, 13, w, h),
                'right_elbow': get_landmark(landmarks, 14, w, h),
                'left_wrist': get_landmark(landmarks, 15, w, h),
                'right_wrist': get_landmark(landmarks, 16, w, h),
                'left_hip': get_landmark(landmarks, 23, w, h),
                'right_hip': get_landmark(landmarks, 24, w, h),
                'left_knee': get_landmark(landmarks, 25, w, h),
                'right_knee': get_landmark(landmarks, 26, w, h),
                'left_ankle': get_landmark(landmarks, 27, w, h),
                'right_ankle': get_landmark(landmarks, 28, w, h)
            }

            angles = {
                'left_elbow': calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist']) if points['left_shoulder'] and points['left_elbow'] and points['left_wrist'] else None,
                'right_elbow': calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist']) if points['right_shoulder'] and points['right_elbow'] and points['right_wrist'] else None,
                'left_knee': calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle']) if points['left_hip'] and points['left_knee'] and points['left_ankle'] else None,
                'right_knee': calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle']) if points['right_hip'] and points['right_knee'] and points['right_ankle'] else None,
                'left_shoulder': calculate_angle(points['left_elbow'], points['left_shoulder'], points['left_hip']) if points['left_elbow'] and points['left_shoulder'] and points['left_hip'] else None,
                'right_shoulder': calculate_angle(points['right_elbow'], points['right_shoulder'], points['right_hip']) if points['right_elbow'] and points['right_shoulder'] and points['right_hip'] else None,
                'left_hip': calculate_angle(points['left_knee'], points['left_hip'], points['left_shoulder']) if points['left_knee'] and points['left_hip'] and points['left_shoulder'] else None,
                'right_hip': calculate_angle(points['right_knee'], points['right_hip'], points['right_shoulder']) if points['right_knee'] and points['right_hip'] and points['right_shoulder'] else None,
            }

            for joint, angle in angles.items():
                if angle and min_angles[joint] <= angle <= max_angles[joint]:
                    # Save frame as image
                    image_file = os.path.join(output_images_dir, f"frame_{frame_count}_{joint}.jpg")
                    cv2.imwrite(image_file, frame)

                    # Write to CSV
                    csv_writer.writerow([timestamp, joint, image_file])

        out.write(frame)

cap.release()
out.release()
print(f"Processed video saved as {output_video_path}")
print(f"Timestamps and joints saved in {output_csv_path}")
