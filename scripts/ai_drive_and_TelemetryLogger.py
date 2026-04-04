import bge
import numpy as np
import os
import time
import onnxruntime as ort
import csv
import math

IMG_W, IMG_H, CHANNELS = 405, 466, 3

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH      = os.path.join(BASE_DIR, "model", "model_best.onnx")
SCREENSHOT_PATH = os.path.join(BASE_DIR, "tmp_frame.png")
SAVE_FOLDER     = os.path.join(BASE_DIR, "output")
CSV_PATH        = os.path.join(SAVE_FOLDER, "BlindDrive_test.csv")

ORIGIN_NAME = "Empty"


class AutoDriveController(bge.types.KX_PythonComponent):

    args = {
        "base_speed":   0.005,
        "max_steer":    0.06,
        "steer_smooth": 0.4,
        "dead_zone":    0.02,
        "bias_correct": 0.0,
    }

    def start(self, args):
        try:
            self.base_speed   = float(args.get("base_speed", 0.005))
            self.max_steer    = float(args.get("max_steer", 0.06))
            self.steer_smooth = float(args.get("steer_smooth", 0.4))
            self.dead_zone    = float(args.get("dead_zone", 0.02))
            self.bias_correct = float(args.get("bias_correct", 0.0))

            self.smooth_steer = 0.0
            self.frame        = 0
            self.ready        = False

            scene = bge.logic.getCurrentScene()

            if ORIGIN_NAME not in scene.objects:
                raise Exception(f"Empty '{ORIGIN_NAME}' not found")

            self.origin = scene.objects[ORIGIN_NAME]

            os.makedirs(SAVE_FOLDER, exist_ok=True)

            self.csv_file = open(CSV_PATH, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["frame", "steering", "car_x", "car_y", "car_rot_z"])

            if not os.path.isfile(MODEL_PATH):
                print(f"[ERROR] Model not found: {MODEL_PATH}")
                return

            import cv2
            self.cv2 = cv2

            self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name

            self.ready = True
            print("=== AutoDrive READY ===")

        except Exception as e:
            print(f"[ERROR start()] {e}")

    def build_input(self, path):
        img = self.cv2.imread(path)
        if img is None:
            return None

        img = self.cv2.resize(img, (IMG_W, IMG_H))
        img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        return img[np.newaxis, ...]

    def update(self):
        try:
            self.frame += 1

            self.object.applyMovement((0, self.base_speed, 0), True)

            if not self.ready:
                return

            bge.render.makeScreenshot(SCREENSHOT_PATH)
            time.sleep(0.02)

            X = self.build_input(SCREENSHOT_PATH)
            if X is None:
                return

            output = self.session.run(None, {self.input_name: X})[0]
            norm_steer = float(output[0, 0]) - self.bias_correct

            if abs(norm_steer) < self.dead_zone:
                norm_steer = 0.0

            raw_steer = norm_steer * self.max_steer

            alpha = 1.0 - self.steer_smooth
            self.smooth_steer = alpha * raw_steer + self.steer_smooth * self.smooth_steer

            steer = float(np.clip(self.smooth_steer, -self.max_steer, self.max_steer))
            self.object.applyRotation((0, 0, steer), True)

            # Position logging
            car_pos = self.object.worldPosition
            origin  = self.origin.worldPosition

            rel_x = float(car_pos.x - origin.x)
            rel_y = float(car_pos.y - origin.y)

            rot_z = math.degrees(self.object.worldOrientation.to_euler('XYZ').z)

            self.csv_writer.writerow([self.frame, steer, rel_x, rel_y, rot_z])

        except Exception as e:
            print(f"[ERROR update()] frame={self.frame}: {e}")

    def end(self):
        if hasattr(self, "csv_file"):
            self.csv_file.close()
        print(f"Saved {self.frame} frames")