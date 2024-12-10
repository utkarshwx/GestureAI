import csv
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define injury thresholds
#football
football_injury_angle_ranges = {
    "ACL Tear": {
        "left_shoulder": (57, 63), "right_shoulder": (67, 73),
        "left_elbow": (7, 13), "right_elbow": (0, 3),
        "left_knee": (172, 178), "right_knee": (142, 148),
        "left_hip": (22, 28), "right_hip": (12, 18)
    },
    "MCL Tear": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (27, 33), "right_knee": (17, 23),
        "left_hip": (7, 13), "right_hip": (2, 8)
    },
    "Hamstring Strain": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (97, 103), "right_knee": (27, 33),
        "left_hip": (77, 83), "right_hip": (47, 53)
    },
    "Quadriceps Tear": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (17, 23), "right_knee": (-3, 3),
        "left_hip": (2, 8), "right_hip": (0, 3)
    },
    "Shoulder Dislocation": {
        "left_shoulder": (0, 3), "right_shoulder": (157, 163),
        "left_elbow": (0, 3), "right_elbow": (22, 28),
        "left_knee": (0, 3), "right_knee": (0, 3),
        "left_hip": (12, 18), "right_hip": (0, 3)
    },
    "Meniscus Tear": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (92, 98), "right_knee": (82, 88),
        "left_hip": (0, 3), "right_hip": (0, 3)
    },
    "Groin Pull": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (0, 3), "right_knee": (0, 3),
        "left_hip": (67, 73), "right_hip": (12, 18)
    },
    "Achilles Tendon Rupture": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (47, 53), "right_knee": (7, 13),
        "left_hip": (32, 38), "right_hip": (0, 3)
    },
    "Patellar Tendon Rupture": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (152, 158), "right_knee": (32, 38),
        "left_hip": (37, 43), "right_hip": (0, 3)
    },
    "Calf Strain": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (0, 3), "right_knee": (0, 3),
        "left_hip": (77, 83), "right_hip": (27, 33)
    },
    "Labrum Tear (Hip)": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (57, 63), "right_knee": (87, 93),
        "left_hip": (87, 93), "right_hip": (27, 33)
    },
    "Concussion": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (0, 3), "right_knee": (0, 3),
        "left_hip": (0, 3), "right_hip": (0, 3)  # Neutral angles for collision impact
    }
}


# Function to calculate angles
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab, bc = a - b, c - b
    angle = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
    return np.degrees(angle)

# Function to check if angle is within range
def is_within_range(angle, range_tuple):
    return range_tuple[0] <= angle <= range_tuple[1]

# Input/output paths
input_video_path = 'videoplayback_football_2.mp4.mp4'
output_csv_path = 'detected_injuries_football_2.csv'
output_video_path = 'output_video_football_2.mp4'

def process_football_video(input_video_path, output_csv_path, output_video_path):

    print("Processing video for rendering joint angles and risk probabilities...")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Open CSV file for output
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

                # Get bounding box coordinates
                x_coords = [landmark.x for landmark in landmarks if landmark.visibility > 0.5]
                y_coords = [landmark.y for landmark in landmarks if landmark.visibility > 0.5]

                if x_coords and y_coords:
                    # Convert normalized coordinates to pixel values
                    x_min = int(min(x_coords) * w)
                    x_max = int(max(x_coords) * w)
                    y_min = int(min(y_coords) * h)
                    y_max = int(max(y_coords) * h)

                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Get joint positions
                points = {
                    'left_shoulder': [landmarks[11].x * w, landmarks[11].y * h],
                    'right_shoulder': [landmarks[12].x * w, landmarks[12].y * h],
                    'left_elbow': [landmarks[13].x * w, landmarks[13].y * h],
                    'right_elbow': [landmarks[14].x * w, landmarks[14].y * h],
                    'left_wrist': [landmarks[15].x * w, landmarks[15].y * h],
                    'right_wrist': [landmarks[16].x * w, landmarks[16].y * h],
                    'left_hip': [landmarks[23].x * w, landmarks[23].y * h],
                    'right_hip': [landmarks[24].x * w, landmarks[24].y * h],
                    'left_knee': [landmarks[25].x * w, landmarks[25].y * h],
                    'right_knee': [landmarks[26].x * w, landmarks[26].y * h],
                    'left_ankle': [landmarks[27].x * w, landmarks[27].y * h],
                    'right_ankle': [landmarks[28].x * w, landmarks[28].y * h]
                }

                # Calculate joint angles
                angles = {
                    'left_elbow': calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist']),
                    'right_elbow': calculate_angle(points['right_shoulder'], points['right_elbow'], points['right_wrist']),
                    'left_knee': calculate_angle(points['left_hip'], points['left_knee'], points['left_ankle']),
                    'right_knee': calculate_angle(points['right_hip'], points['right_knee'], points['right_ankle']),
                    'left_shoulder': calculate_angle(points['left_elbow'], points['left_shoulder'], points['left_hip']),
                    'right_shoulder': calculate_angle(points['right_elbow'], points['right_shoulder'], points['right_hip']),
                    'left_hip': calculate_angle(points['left_knee'], points['left_hip'], points['left_shoulder']),
                    'right_hip': calculate_angle(points['right_knee'], points['right_hip'], points['right_shoulder']),
                }

                # Detect injuries
                detected_injuries = []
                risk_probabilities = []
                for injury, ranges in football_injury_angle_ranges.items():
                    matching_joints = sum(is_within_range(angles[joint], ranges[joint]) for joint in ranges.keys())
                    risk_probability = (matching_joints / len(ranges)) * 100
                    if risk_probability > 0:
                        detected_injuries.append(injury)
                        risk_probabilities.append(risk_probability)

                # Annotate video frame
                for joint, angle in angles.items():
                    position = tuple(map(int, points[joint]))
                    cv2.putText(frame, f"{int(angle)}°", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw risk bar
                bar_x, bar_y = 50, 50
                bar_width, bar_height = 300, 20
                risk_level = int(max(risk_probabilities) if risk_probabilities else 0)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 255), 2)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int((bar_width * risk_level) / 100), bar_y + bar_height),
                            (0, 255, 0), -1)
                cv2.putText(frame, f"Risk: {risk_level}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

                # Save detected injuries to CSV
                if detected_injuries:
                    csv_writer.writerow([timestamp, frame_count, ", ".join(detected_injuries)])

                # Write the frame to the output video
                out_video.write(frame)

    cap.release()
    out_video.release()

    print(f"Results saved to {output_csv_path}")

    return output_video_path, output_csv_path

