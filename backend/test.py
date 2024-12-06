import csv
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# Initialize MediaPipe pose with optimized settings
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

# Define injury thresholds
injury_angle_ranges = {
    "Rotator Cuff Tear": {
        "left_shoulder": (0, 3), "right_shoulder": (177, 183),
        "left_elbow": (17, 23), "right_elbow": (87, 93),
        "left_knee": (7, 13), "right_knee": (27, 33),
        "left_hip": (17, 23), "right_hip": (47, 53)
    },
    "Tennis Elbow": {
        "left_shoulder": (87, 93), "right_shoulder": (117, 123),
        "left_elbow": (42, 48), "right_elbow": (147, 153),
        "left_knee": (12, 18), "right_knee": (17, 23),
        "left_hip": (0, 3), "right_hip": (27, 33)
    },
    "ACL Tear": {
        "left_shoulder": (57, 63), "right_shoulder": (67, 73),
        "left_elbow": (7, 13), "right_elbow": (0, 3),
        "left_knee": (22, 28), "right_knee": (167, 173),
        "left_hip": (37, 43), "right_hip": (57, 63)
    },
    "Medial Meniscus Tear": {
        "left_shoulder": (0, 3), "right_shoulder": (27, 33),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (0, 3), "right_knee": (157, 163),
        "left_hip": (0, 3), "right_hip": (47, 53)
    },
    "Patellar Tendonitis": {
        "left_shoulder": (87, 93), "right_shoulder": (97, 103),
        "left_elbow": (27, 33), "right_elbow": (87, 93),
        "left_knee": (82, 88), "right_knee": (82, 88),
        "left_hip": (57, 63), "right_hip": (57, 63)
    },
    "Achilles Tendon Rupture": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (7, 13), "right_knee": (27, 33),
        "left_hip": (0, 3), "right_hip": (17, 23)
    },
    "Wrist Sprain": {
        "left_shoulder": (0, 3), "right_shoulder": (137, 143),
        "left_elbow": (0, 3), "right_elbow": (157, 163),
        "left_knee": (7, 13), "right_knee": (17, 23),
        "left_hip": (0, 3), "right_hip": (27, 33)
    },
    "Stress Fracture in Spine": {
        "left_shoulder": (0, 3), "right_shoulder": (177, 183),
        "left_elbow": (0, 3), "right_elbow": (157, 163),
        "left_knee": (67, 73), "right_knee": (47, 53),
        "left_hip": (117, 123), "right_hip": (27, 33)
    },
    "Labral Tear": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (17, 23), "right_knee": (37, 43),
        "left_hip": (0, 3), "right_hip": (117, 123)
    },
    "Hamstring Strain": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (7, 13), "right_knee": (27, 33),
        "left_hip": (0, 3), "right_hip": (147, 153)
    },
    "Groin Strain": {
        "left_shoulder": (0, 3), "right_shoulder": (0, 3),
        "left_elbow": (0, 3), "right_elbow": (0, 3),
        "left_knee": (0, 3), "right_knee": (0, 3),
        "left_hip": (117, 123), "right_hip": (27, 33)
    },
    "Posterior Shoulder Instability": {
        "left_shoulder": (0, 3), "right_shoulder": (147, 153),
        "left_elbow": (0, 3), "right_elbow": (42, 48)
    }
}

# Calculate joint angles
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0
    angle = np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0))
    return round(np.degrees(angle), 2)

# Check if angle is within range
def is_within_range(angle, range_tuple):
    return range_tuple[0] <= angle <= range_tuple[1]

# Process video and detect injuries
def process_video(input_video_path, output_video_path, output_csv_path):
    print(f"Processing started at {datetime.now()}")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return None, None

    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    with open(output_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Timestamp", "Frame", "Joint Angles", "Detected Injuries", "Risk Level"])
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            timestamp = f"{frame_count // fps:02}:{frame_count % fps:02}"

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
                
                landmarks = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                points = {
                    "left_shoulder": (landmarks[11].x * w, landmarks[11].y * h),
                    "right_shoulder": (landmarks[12].x * w, landmarks[12].y * h),
                    "left_elbow": (landmarks[13].x * w, landmarks[13].y * h),
                    "right_elbow": (landmarks[14].x * w, landmarks[14].y * h),
                    "left_knee": (landmarks[25].x * w, landmarks[25].y * h),
                    "right_knee": (landmarks[26].x * w, landmarks[26].y * h),
                }

                # Calculate angles
                angles = {
                    "left_elbow": calculate_angle(points["left_shoulder"], points["left_elbow"], points["right_elbow"]),
                    "right_elbow": calculate_angle(points["right_shoulder"], points["right_elbow"], points["left_elbow"]),
                }

                # Detect injuries
                detected_injuries = []
                for injury, angle_ranges in injury_angle_ranges.items():
                    if any(
                        joint in angles and is_within_range(angles[joint], angle_ranges[joint])
                        for joint in angle_ranges
                    ):
                        detected_injuries.append(injury)

                # Log results
                csv_writer.writerow([timestamp, frame_count, angles, detected_injuries, "High" if detected_injuries else "Low"])

            out_video.write(frame)

            if frame_count % 30 == 0:
                print(f"Progress: {(frame_count / total_frames) * 100:.2f}%")

    cap.release()
    out_video.release()
    print(f"Processing completed at {datetime.now()}")
    return output_video_path, output_csv_path

# Example usage:
# input_video = "input.mp4"
# output_video = "output.mp4"
# output_csv = "output.csv"
# process_video(input_video, output_video, output_csv)
