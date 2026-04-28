#!/usr/bin/env python3
"""
Module 11 — Human Detection & Person Recognition (OpenCV)
---------------------------------------------------------

- Detects multiple humans in camera feed.
- Recognizes known persons from pre-encoded data.
- Labels unknown persons automatically.
- Works standalone and ROS2 compatible (optional).

Run:
  python3 person_node.py --camera 0
"""

import cv2
import face_recognition
import os
import pickle
import argparse
import logging
import numpy as np

# Optional ROS imports
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, String
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False

# ---------------- Config ----------------
ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), "face_recognition_model", "encodings.pkl")
FRAME_RESIZE = 0.25  # 1/4 size for speed
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("person_node")


# ---------------- Core Recognition ----------------
class FaceRecognizer:
    def __init__(self, encodings_path=ENCODINGS_PATH):
        if not os.path.exists(encodings_path):
            raise FileNotFoundError(f"Encodings file not found at {encodings_path}. Run encode_faces.py first.")
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        self.known_encodings = data["encodings"]
        self.known_names = data["names"]
        logger.info(f"Loaded {len(self.known_names)} known face encodings.")

    def recognize_faces(self, frame):
        """Return (boxes, names) for all detected faces."""
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_small, model="hog")
        encs = face_recognition.face_encodings(rgb_small, boxes)

        names = []
        for encoding in encs:
            matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=0.45)
            name = "Unknown"
            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matched_idxs:
                    counts[self.known_names[i]] = counts.get(self.known_names[i], 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)

        boxes = [(int(top / FRAME_RESIZE), int(right / FRAME_RESIZE),
                  int(bottom / FRAME_RESIZE), int(left / FRAME_RESIZE)) for (top, right, bottom, left) in boxes]
        return boxes, names


# ---------------- Standalone Runner ----------------
def run_standalone(camera_index=0):
    logger.info("📸 Starting human detection and recognition (OpenCV)...")
    recog = FaceRecognizer()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("❌ Camera not found or busy.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        boxes, names = recog.recognize_faces(frame)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 4, bottom - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Human Detection & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- ROS2 Node (Optional) ----------------
class PersonNode(Node):
    def __init__(self, cam_index=0):
        super().__init__('person_node')
        self.recog = FaceRecognizer()
        self.pub_detected = self.create_publisher(Bool, '/person_detected', 10)
        self.pub_name = self.create_publisher(String, '/person_name', 10)
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not available.")
        self.timer = self.create_timer(0.3, self.loop)
        self.get_logger().info("PersonNode (OpenCV) started.")

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        boxes, names = self.recog.recognize_faces(frame)
        detected = len(names) > 0
        msg_detected = Bool()
        msg_detected.data = detected
        self.pub_detected.publish(msg_detected)

        msg_name = String()
        msg_name.data = ", ".join(names) if names else "None"
        self.pub_name.publish(msg_name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 5, bottom - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("ROS2 Person Detection", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Person Detection & Recognition (OpenCV)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--ros", action="store_true", help="Run as ROS2 node")
    args = parser.parse_args()

    if args.ros:
        if not ROS_AVAILABLE:
            logger.error("❌ ROS2 not available. Run standalone instead.")
            return
        rclpy.init()
        node = PersonNode(cam_index=args.camera)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_standalone(args.camera)


if __name__ == "__main__":
    main()
