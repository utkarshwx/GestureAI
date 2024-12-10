import os
import logging
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from pose_estimationtennis import process_video  
from gesture_detection_basketball import process_basketball_video
from gesture_detection_football import process_football_video
from report import process_report

# Configure deep logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG for deep logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output logs to stderr
    ]
)
logger = logging.getLogger("PoseEstimationApp")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS for all routes

# Directory to temporarily store processed files
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/health', methods=['GET'])
def ok():
    logger.debug("Health check endpoint called.")
    return "OK"

@app.route('/upload', methods=['POST'])
def process_and_download():
    logger.info("Processing upload request.")
    try:
        # Check if the request contains a video file
        if 'video' not in request.files:
            logger.error("No video file provided in the request.")
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files['video']
        logger.debug(f"Received video file: {video_file.filename}")

        preference = request.form.get('preference', 'tennis')
        # Save the uploaded video temporarily
        input_video_path = os.path.join(TEMP_DIR, video_file.filename)
        video_file.save(input_video_path)
        logger.info(f"Uploaded video saved at: {input_video_path}")

        # Paths for processed outputs
        output_video_path = os.path.join(TEMP_DIR, "processed_video.mp4")
        output_csv_path = os.path.join(TEMP_DIR, "data.csv")

        # Process the video
        logger.debug("Starting video processing.")

        if preference == "tennis" :
            processed_video, processed_csv = process_video(input_video_path, output_video_path, output_csv_path)
        elif preference == "basketball":
            processed_video, processed_csv = process_basketball_video(input_video_path, output_video_path, output_csv_path)
        elif preference == "football":
            processed_video, processed_csv = process_football_video(input_video_path, output_video_path, output_csv_path)


        # processed_video, processed_csv = process_video(input_video_path, output_video_path, output_csv_path)
        processed_report = process_report()
        logger.info(f"Processing completed. Processed video: {processed_video}, Processed CSV: {processed_csv}, Processed Report: {processed_report}")
        output_report_path = os.path.join(TEMP_DIR, "report.pdf")
        # Return download links for the processed files
        return jsonify({
            "processed_video_url": f"/download/video/{os.path.basename(output_video_path)}",
            "processed_csv_url": f"/download/csv/{os.path.basename(output_report_path)}"
        })
    except Exception as e:
        logger.exception("Error during video processing.")
        return jsonify({"error": str(e)}), 500

@app.route('/download/video/<filename>', methods=['GET'])
def download_video(filename):
    file_path = os.path.join(TEMP_DIR, filename)
    logger.info(f"Request to download video: {file_path}")
    if os.path.exists(file_path):
        logger.debug(f"Video file found: {file_path}")
        return send_file(
            file_path,
            as_attachment=True,
            # download_name=filename,  # For Flask 2.x
            # mimetype="video/mp4"
        )
    logger.error(f"Video file not found: {file_path}")
    return jsonify({"error": "File not found"}), 404

@app.route('/download/csv/<filename>', methods=['GET'])
def download_csv(filename):
    file_path = os.path.join(TEMP_DIR, filename)
    logger.info(f"Request to download PDF: {file_path}")
    if os.path.exists(file_path):
        logger.debug(f"PDF file found: {file_path}")
        return send_file(
            file_path,
            as_attachment=True,
            # download_name=filename,  # For Flask 2.x
            # mimetype="text/csv"
        )
    logger.error(f"PDF file not found: {file_path}")
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    logger.info("Starting Flask application.")
    app.run(host="0.0.0.0", port=8000, debug=True)
