# ai_server.py

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import threading
import time
import sys
import os
import io # For image encoding

# --- Try importing your specific AI modules ---
try:
    from face_detection import FaceDetection
    from head_pose import HeadPose
    from object_detection import ObjectDetector
except ImportError as e:
    print(f"----------------------------------------------------")
    print(f"FATAL ERROR: Failed to import AI modules: {e}")
    print(f"Ensure .py files and necessary libraries (ultralytics, mediapipe, opencv-python, numpy) are present and installed in the active environment.")
    print(f"----------------------------------------------------")
    sys.exit("AI Module Import Failed")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Initialize AI Components Globally ---
try:
    print("Initializing AI components...")
    face_detector = FaceDetection()
    head_pose_estimator = HeadPose()
    object_detector = ObjectDetector()
    print("AI components initialized successfully.")
except Exception as init_e:
    print(f"FATAL ERROR during AI component initialization: {init_e}")
    sys.exit("AI Initialization Failed")

# --- Global state for monitoring ---
monitoring = False
latest_status = {
    "faces": 0, "head_pose_status": "N/A", "prohibited_objects": [],
    "warnings_list": [], "warning_count": 0, "terminated": False
}
latest_frame_bytes = None
status_lock = threading.Lock()

# --- Rule Thresholds & Warning Logic ---
HEAD_YAW_THRESHOLD = 0.02
WARNING_THRESHOLD = 3
WARNING_COOLDOWN = 3
last_warning_time = 0

# --- Flask Routes ---

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring_route():
    global monitoring, latest_status, last_warning_time
    with status_lock:
        if monitoring: return jsonify({"status": "already running"})
        print("Start request: Starting monitoring...")
        monitoring = True
        latest_status = {"faces": 0, "head_pose_status": "N/A", "prohibited_objects": [], "warnings_list": ["Initializing..."], "warning_count": 0, "terminated": False}
        last_warning_time = 0
    thread = threading.Thread(target=monitor_exam_task, daemon=True)
    thread.start()
    print("Start request: Monitoring thread initiated.")
    return jsonify({"status": "started"})

@app.route('/stop-monitoring', methods=['POST'])
def stop_monitoring_route():
    global monitoring
    print("Stop request: Setting monitoring flag to False.")
    with status_lock: monitoring = False
    return jsonify({"status": "stopped"})

@app.route('/get-status', methods=['GET'])
def get_status_route():
    global latest_status
    with status_lock: current_status = latest_status.copy()
    return jsonify(current_status)

@app.route('/video_feed')
def video_feed_route():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Frame Generator for MJPEG Stream ---
def generate_frames():
    global latest_frame_bytes, status_lock
    while True:
        frame_bytes = None
        with status_lock:
            if latest_frame_bytes: frame_bytes = latest_frame_bytes
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # ~25 FPS stream rate

# --- Background Monitoring Task ---
def monitor_exam_task():
    global monitoring, latest_status, last_warning_time, latest_frame_bytes
    global face_detector, head_pose_estimator, object_detector
    print("Monitor Task: Started. Opening camera...")
    cap = None
    current_warning_count = 0

    try:
        cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened(): raise IOError("Cannot open video capture device 0")
        print("Monitor Task: Camera opened.")
        with status_lock: current_warning_count = latest_status["warning_count"]

        while True:
            should_continue = False
            with status_lock: should_continue = monitoring
            if not should_continue: print("Monitor Task: Stop signal. Exiting loop."); break

            ret, frame = cap.read()
            if not ret: print("Monitor Task: Warn - Failed frame grab."); time.sleep(0.5); continue

            current_violations_this_frame = []; detected_faces_count = 0
            current_pose_status = "N/A"; detected_prohibited_list = []
            is_terminated_this_frame = False

            try:
                # AI Pipeline
                face_landmarks_list = face_detector.detect_faces(frame)
                detected_faces_count = len(face_landmarks_list)

                if detected_faces_count == 0: current_violations_this_frame.append("No face detected")
                elif detected_faces_count > 1: current_violations_this_frame.append("Multiple faces detected")
                else: # Exactly one face
                    current_pose_status = "OK"
                    landmarks = face_landmarks_list[0]
                    yaw = head_pose_estimator.get_yaw(landmarks)
                    if abs(yaw) > HEAD_YAW_THRESHOLD:
                        current_pose_status = "Looking Away"; current_violations_this_frame.append("Looking away")

                detected_prohibited_list = object_detector.detect_objects(frame)
                if detected_prohibited_list:
                    current_violations_this_frame.append(f"Prohibited item(s): {', '.join(detected_prohibited_list)}")

                # Warning Logic
                if current_violations_this_frame:
                    current_time = time.time()
                    if (current_time - last_warning_time) > WARNING_COOLDOWN:
                        current_warning_count += 1
                        last_warning_time = current_time
                        print(f"Monitor Task: Warn {current_warning_count} for: {', '.join(current_violations_this_frame)}")
                        if current_warning_count >= WARNING_THRESHOLD:
                             print(f"Monitor Task: TERMINATION THRESHOLD REACHED ({current_warning_count}/{WARNING_THRESHOLD})")
                             is_terminated_this_frame = True
                             with status_lock: monitoring = False # Signal stop

                # Update Status
                with status_lock:
                    latest_status["faces"] = detected_faces_count
                    latest_status["head_pose_status"] = current_pose_status
                    latest_status["prohibited_objects"] = detected_prohibited_list
                    latest_status["warnings_list"] = current_violations_this_frame
                    latest_status["warning_count"] = current_warning_count
                    latest_status["terminated"] = is_terminated_this_frame

                # Prepare Frame for Streaming
                display_color = (0, 255, 0); # Green
                if current_warning_count == 1: display_color = (0, 255, 255) # Yellow
                if current_warning_count == 2: display_color = (0, 165, 255) # Orange
                if current_warning_count >= WARNING_THRESHOLD: display_color = (0, 0, 255) # Red

                # Add Overlays
                cv2.putText(frame, f"Warnings: {current_warning_count}/{WARNING_THRESHOLD}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2, cv2.LINE_AA)
                if is_terminated_this_frame:
                     text_size = cv2.getTextSize("TERMINATED", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                     text_x = (frame.shape[1] - text_size[0]) // 2; text_y = (frame.shape[0] + text_size[1]) // 2
                     cv2.putText(frame, "TERMINATED", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

                # Encode and store frame
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    with status_lock: latest_frame_bytes = buffer.tobytes()

            except Exception as e:
                error_msg = f"AI Processing Error: {type(e).__name__}"
                print(f"Monitor Task: {error_msg} - {e}")
                with status_lock: latest_status["warnings_list"] = [error_msg]
                time.sleep(1)

            if is_terminated_this_frame: break # Exit loop immediately on termination

    # --- FIX: Correctly Indented Error Handling ---
    except IOError as e:
        error_msg = f"Camera Error: {e}"
        print(f"Monitor Task: {error_msg}")
        with status_lock:
            latest_status["warnings_list"] = [error_msg]
    except Exception as e:
        error_msg = f"Unhandled Error in Monitor Task: {e}"
        print(error_msg)
        with status_lock:
            latest_status["warnings_list"] = [error_msg]
    # --- End of Fix ---
    finally:
        if cap is not None and cap.isOpened(): cap.release(); print("Monitor Task: Camera released.")
        print("Monitor Task: Exiting.")
        with status_lock:
            final_terminated_status = latest_status.get("terminated", False)
            monitoring = False
            latest_status = {"faces": 0, "head_pose_status": "N/A", "prohibited_objects": [], "warnings_list": ["Monitoring Stopped"], "warning_count": latest_status.get("warning_count", 0), "terminated": final_terminated_status}

# --- Main Execution Guard ---
if __name__ == '__main__':
    print("Starting Flask AI Server on http://localhost:5001 (accessible via 0.0.0.0)")
    app.run(port=5001, debug=True, use_reloader=False, host='0.0.0.0', threaded=True)
