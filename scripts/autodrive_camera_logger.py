import bge
import numpy as np
import os
import time
import onnxruntime as ort
import math
import csv
from mathutils import Euler

# ─────────────────────────────
# Paths (relative to project)
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH      = os.path.join(BASE_DIR, "model", "model_best.onnx")
SCREENSHOT_PATH = os.path.join(BASE_DIR, "tmp_frame.png")
SAVE_FOLDER     = os.path.join(BASE_DIR, "output", "telemetry")
CSV_PATH        = os.path.join(SAVE_FOLDER, "autodrive_camrot.csv")

ORIGIN_NAME = "Empty"

# ─────────────────────────────
# Driving params
# ─────────────────────────────
BASE_SPEED   = 0.005
MAX_STEER    = 0.06
STEER_SMOOTH = 0.4
DEAD_ZONE    = 0.001
BIAS_CORRECT = 0.0

# Camera
CAM_MAX_DEGREES = 80.0
CAM_SMOOTH      = 0.9

# Image
IMG_W = 405
IMG_H = 466


class AutoDriveController(bge.types.KX_PythonComponent):

    args = {}

    def start(self, args):
        try:
            self.smooth_steer = 0.0
            self.smooth_cam_z = 0.0
            self.frame = 0
            self.ready = False
            self.camera = None

            scene = bge.logic.getCurrentScene()

            # Origin reference
            if ORIGIN_NAME not in scene.objects:
                raise Exception(f"Empty '{ORIGIN_NAME}' not found")
            self.origin = scene.objects[ORIGIN_NAME]

            # CSV setup
            os.makedirs(SAVE_FOLDER, exist_ok=True)
            self.csv_file = open(CSV_PATH, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)

            self.csv_writer.writerow([
                "frame", "steering", "rel_x", "rel_y", "car_rot_z_deg"
            ])

            print("[AutoDrive] Starting...")

            # Camera setup
            self.camera = scene.active_camera
            base_e = self.camera.localOrientation.to_euler('XYZ')

            self.cam_base_x = base_e.x
            self.cam_base_y = base_e.y
            self.cam_base_z = base_e.z

            # OpenCV
            import cv2
            self.cv2 = cv2

            # Model
            if not os.path.isfile(MODEL_PATH):
                print(f"[ERROR] Model not found: {MODEL_PATH}")
                return

            self.session = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name

            self.ready = True
            print("[AutoDrive] READY")

        except Exception as e:
            print(f"[ERROR start()] {e}")

    # ─────────────────────────────
    def build_input(self, path):
        img = self.cv2.imread(path)
        if img is None:
            return None

        img = self.cv2.resize(img, (IMG_W, IMG_H))
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        return img[np.newaxis, ...]

    # ─────────────────────────────
    def rotate_camera(self, norm_steer):
        target = math.copysign(
            abs(norm_steer) ** 0.5,
            norm_steer
        ) * math.radians(CAM_MAX_DEGREES)

        alpha = 1.0 - CAM_SMOOTH
        self.smooth_cam_z = alpha * target + CAM_SMOOTH * self.smooth_cam_z

        new_euler = Euler(
            (self.cam_base_x,
             self.cam_base_y,
             self.cam_base_z + self.smooth_cam_z),
            'XYZ'
        )

        self.camera.localOrientation = new_euler.to_matrix()

    # ─────────────────────────────
    def update(self):
        try:
            self.frame += 1

            # Move forward
            self.object.applyMovement((0, BASE_SPEED, 0), True)

            if not self.ready:
                return

            # Screenshot
            bge.render.makeScreenshot(SCREENSHOT_PATH)
            time.sleep(0.02)

            X = self.build_input(SCREENSHOT_PATH)
            if X is None:
                return

            # Model inference
            output = self.session.run(None, {self.input_name: X})[0]
            norm_steer = float(output[0, 0]) - BIAS_CORRECT

            if abs(norm_steer) < DEAD_ZONE:
                norm_steer = 0.0

            # Smooth steering
            raw_steer = norm_steer * MAX_STEER
            alpha = 1.0 - STEER_SMOOTH
            self.smooth_steer = alpha * raw_steer + STEER_SMOOTH * self.smooth_steer

            steer = float(np.clip(self.smooth_steer, -MAX_STEER, MAX_STEER))
            self.object.applyRotation((0, 0, steer), True)

            # Camera follow
            self.rotate_camera(norm_steer)

            # Position
            car_pos = self.object.worldPosition
            origin  = self.origin.worldPosition

            rel_x = float(car_pos.x - origin.x)
            rel_y = float(car_pos.y - origin.y)

            rot_z = round(
                math.degrees(self.object.worldOrientation.to_euler('XYZ').z),
                4
            )

            # Save log
            self.csv_writer.writerow([
                self.frame, steer, rel_x, rel_y, rot_z
            ])
            self.csv_file.flush()

            if self.frame % 10 == 0:
                print(f"[AutoDrive] f={self.frame} | steer={steer:+.4f}")

        except Exception as e:
            print(f"[ERROR update()] frame={self.frame}: {e}")

    # ─────────────────────────────
    def end(self):
        if hasattr(self, "csv_file"):
            self.csv_file.close()

        if self.camera:
            self.camera.localOrientation = Euler(
                (self.cam_base_x, self.cam_base_y, self.cam_base_z),
                'XYZ'
            ).to_matrix()

        print(f"[AutoDrive] Stopped | frames={self.frame}")