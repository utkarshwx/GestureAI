import csv
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe pose with optimized settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,  # Dynamic video processing
    model_complexity=2,  # Maximum detail level
    smooth_landmarks=True,  # Enable landmark smoothing
    enable_segmentation=True,  # Enable segmentation
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Enhanced visualization settings
custom_drawing_specs = mp_draw.DrawingSpec(
    color=(0, 255, 0),  # Green color for landmarks
    thickness=2,
    circle_radius=2
)

custom_connection_specs = mp_draw.DrawingSpec(
    color=(255, 255, 0),  # Yellow color for connections
    thickness=2
)

#Define comprehensive injury thresholds for sports-specific motions
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


def calculate_angle(a, b, c):
    
    try:
        a = np.array([float(a[0]), float(a[1])], dtype=np.float64)
        b = np.array([float(b[0]), float(b[1])], dtype=np.float64)
        c = np.array([float(c[0]), float(c[1])], dtype=np.float64)

        # Calculate vectors
        ba = a - b
        bc = c - b

        # Calculate magnitudes
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)

        # Check for zero vectors with high precision threshold
        if ba_norm < 1e-10 or bc_norm < 1e-10:
            return 0

        # Calculate angle using dot product
        cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
        cosine = np.clip(cosine, -1.0, 1.0)  # Ensure value is in valid range
        angle = np.degrees(np.arccos(cosine))

        return round(angle, 2)  # Return with 2 decimal precision

    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0


def is_within_range(angle, range_tuple):
    
    return range_tuple[0] <= angle <= range_tuple[1]


def get_bounding_box(landmarks, frame_shape, padding_factor=0.1):
    
    h, w = frame_shape[:2]

    # Collect coordinates of visible landmarks
    x_coordinates = []
    y_coordinates = []

    for landmark in landmarks:
        if landmark.visibility > 0.6:  # Only use highly visible landmarks
            x_coordinates.append(landmark.x * w)
            y_coordinates.append(landmark.y * h)

    if not x_coordinates or not y_coordinates:
        return None

    # Calculate bounds
    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)

    # Calculate dynamic padding
    box_width = x_max - x_min
    box_height = y_max - y_min

    padding_x = box_width * padding_factor
    padding_y = box_height * padding_factor

    # Apply padding with boundary checks
    x_min = max(0, int(x_min - padding_x))
    x_max = min(w, int(x_max + padding_x))
    y_min = max(0, int(y_min - padding_y))
    y_max = min(h, int(y_max + padding_y))

    return (x_min, y_min, x_max, y_max)


def calculate_motion_speed(current_points, previous_points):
    
    if not previous_points:
        return 0.0

    speeds = []
    for point_name in current_points:
        if point_name in previous_points:
            current = np.array(current_points[point_name][:2])
            previous = np.array(previous_points[point_name][:2])
            speed = np.linalg.norm(current - previous)
            speeds.append(speed)

    return np.mean(speeds) if speeds else 0.0


def process_video(input_video_path, output_video_path , output_csv_path):
    
    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Optimize for fast movement

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer with high quality
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (frame_width, frame_height),
        isColor=True
    )

    # Initialize CSV writer with comprehensive data logging
    with open(output_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Timestamp',
            'Frame',
            'Bounding Box',
            'Joint Angles',
            'Confidence Scores',
            'Detected Injuries',
            'Risk Level',
            'Motion Speed',
            'Overall Assessment'
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
                bbox = get_bounding_box(landmarks, frame.shape)

                if bbox:
                    x_min, y_min, x_max, y_max = bbox

                    # Draw enhanced bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Process landmarks and calculate points
                    h, w = frame.shape[:2]
                    current_points = {}
                    landmark_indices = {
                        # Upper body
                        'nose': 0,
                        'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
                        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
                        'left_ear': 7, 'right_ear': 8,
                        'mouth_left': 9, 'mouth_right': 10,
                        'left_shoulder': 11, 'right_shoulder': 12,
                        'left_elbow': 13, 'right_elbow': 14,
                        'left_wrist': 15, 'right_wrist': 16,
                        'left_pinky': 17, 'right_pinky': 18,
                        'left_index': 19, 'right_index': 20,
                        'left_thumb': 21, 'right_thumb': 22,

                        # Lower body
                        'left_hip': 23, 'right_hip': 24,
                        'left_knee': 25, 'right_knee': 26,
                        'left_ankle': 27, 'right_ankle': 28,
                        'left_heel': 29, 'right_heel': 30,
                        'left_foot_index': 31, 'right_foot_index': 32
                    }

                    # Extract all landmark points with enhanced visibility check
                    for name, idx in landmark_indices.items():
                        landmark = landmarks[idx]
                        x = landmark.x * w
                        y = landmark.y * h
                        visibility = landmark.visibility
                        if visibility > 0.5:  # Visibility threshold
                            current_points[name] = [x, y, visibility]

                    # Calculate all joint angles
                    angles = {}
                    confidence_scores = {}
                    angle_definitions = {
                        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
                        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
                        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
                        'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
                        'left_hip': ('left_knee', 'left_hip', 'left_shoulder'),
                        'right_hip': ('right_knee', 'right_hip', 'right_shoulder'),
                        'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
                        'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
                        'neck': ('left_shoulder', 'nose', 'right_shoulder'),
                        'left_wrist': ('left_elbow', 'left_wrist', 'left_index'),
                        'right_wrist': ('right_elbow', 'right_wrist', 'right_index'),
                        'left_ankle': ('left_knee', 'left_ankle', 'left_foot_index'),
                        'right_ankle': ('right_knee', 'right_ankle', 'right_foot_index')
                    }

                    # Calculate angles based on defined joint relationships
                    for angle_name, (p1, p2, p3) in angle_definitions.items():
                        if all(p in current_points for p in (p1, p2, p3)):
                            angle = calculate_angle(
                                current_points[p1][:2],
                                current_points[p2][:2],
                                current_points[p3][:2]
                            )
                            angles[angle_name] = angle

                            # Calculate confidence score for this angle
                            confidence = min(
                                current_points[p1][2],
                                current_points[p2][2],
                                current_points[p3][2]
                            )
                            confidence_scores[angle_name] = round(confidence * 100, 2)

                    # Detect injuries and calculate risks
                    detected_injuries = []
                    risk_probabilities = []
                    injury_details = []

                    for injury_name, angle_ranges in injury_angle_ranges.items():
                        matching_joints = 0
                        total_relevant_joints = len(angle_ranges)
                        problematic_angles = []

                        for joint, range_tuple in angle_ranges.items():
                            if joint in angles and is_within_range(angles[joint], range_tuple):
                                matching_joints += 1
                                problematic_angles.append(f"{joint}: {angles[joint]:.1f}°")

                        if matching_joints > 0:
                            risk_probability = (matching_joints / total_relevant_joints) * 100
                            detected_injuries.append(injury_name)
                            risk_probabilities.append(risk_probability)
                            injury_details.append({
                                'name': injury_name,
                                'risk': risk_probability,
                                'problematic_angles': problematic_angles
                            })

                    # Calculate motion speed if we have previous points
                    motion_speed = calculate_motion_speed(current_points, previous_points)
                    previous_points = current_points.copy()

                    # Draw skeleton with enhanced visualization
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        custom_drawing_specs,
                        custom_connection_specs
                    )

                    # Draw angles on frame with enhanced visibility
                    for joint, angle in angles.items():
                        if joint in current_points:
                            position = tuple(map(int, current_points[joint][:2]))
                            confidence = confidence_scores[joint]

                            # Color-coded confidence visualization
                            color = (
                                int(255 * (1 - confidence / 100)),
                                int(255 * (confidence / 100)),
                                0
                            )

                            # Draw angle and confidence
                            cv2.putText(
                                frame,
                                f"{angle:.1f}° ({confidence}%)",
                                position,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )

                    # Draw risk visualization
                    if risk_probabilities:
                        max_risk = max(risk_probabilities)
                        bar_x = x_min
                        bar_y = y_min - 40
                        bar_width = x_max - x_min
                        bar_height = 20

                        # Draw risk bar background
                        cv2.rectangle(frame, (bar_x, bar_y),
                                      (bar_x + bar_width, bar_y + bar_height),
                                      (0, 0, 255), 2)

                        # Draw risk level fill
                        fill_width = int((bar_width * max_risk) / 100)
                        cv2.rectangle(frame, (bar_x, bar_y),
                                      (bar_x + fill_width, bar_y + bar_height),
                                      (0, 255, 0), -1)

                        # Add detailed risk information
                        cv2.putText(frame,
                                    f"Risk: {int(max_risk)}% - {', '.join(detected_injuries)}",
                                    (bar_x, bar_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2)

                    # Generate overall assessment
                    assessment = "Normal"
                    if detected_injuries:
                        if max(risk_probabilities) > 75:
                            assessment = "High Risk - Immediate Attention Required"
                        elif max(risk_probabilities) > 50:
                            assessment = "Moderate Risk - Monitor Closely"
                        else:
                            assessment = "Low Risk - Continue Monitoring"

                    # Record data to CSV
                    csv_writer.writerow([
                        timestamp,
                        frame_count,
                        f"({x_min}, {y_min}, {x_max}, {y_max})",
                        str(angles),
                        str(confidence_scores),
                        str(detected_injuries),
                        f"{max(risk_probabilities) if risk_probabilities else 0:.2f}",
                        f"{motion_speed:.2f}",
                        assessment
                    ])

            # Write processed frame
            out_video.write(frame)

            # Show progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100

    # Cleanup and finalize
    cap.release()
    out_video.release()
    print(f"Results saved to {output_csv_path}")

    return output_video_path, output_csv_path


# if _name_ == "_main_":
#     # Define paths for processing
#     input_video_path = 'video_playback.mp4'  # Replace with your video file
#     output_csv_path = 'pose_analysis_detailed.csv'
#     output_video_path = 'output_analyzed.mp4'

#     # Process the video
#     process_video(input_video_path, output_csv_path, output_video_path)