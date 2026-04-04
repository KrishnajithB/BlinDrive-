import bge
import csv
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SAVE_FOLDER = os.path.join(BASE_DIR, "output", "telemetry")
CSV_PATH = os.path.join(SAVE_FOLDER, "manual_drive.csv")

ORIGIN_NAME = "Empty"


class ManualTelemetryLogger(bge.types.KX_PythonComponent):

    args = {
        "base_speed":  0.005,
        "steer_accel": 0.003,
        "steer_decay": 0.85,
        "max_steer":   0.06,
    }

    def start(self, args):
        self.base_speed  = float(args.get("base_speed", 0.005))
        self.steer_accel = float(args.get("steer_accel", 0.003))
        self.steer_decay = float(args.get("steer_decay", 0.85))
        self.max_steer   = float(args.get("max_steer", 0.06))

        self.steering = 0.0
        self.frame    = 0

        scene = bge.logic.getCurrentScene()

        if ORIGIN_NAME not in scene.objects:
            raise Exception(f"Empty '{ORIGIN_NAME}' not found")

        self.origin = scene.objects[ORIGIN_NAME]

        os.makedirs(SAVE_FOLDER, exist_ok=True)

        self.csv_file = open(CSV_PATH, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "frame",
            "steering",
            "car_x",
            "car_y",
            "car_rot_z"
        ])

        print("=== TELEMETRY LOGGER STARTED ===")

    def update(self):
        keyboard  = bge.logic.keyboard.events
        KX_ACTIVE = bge.logic.KX_INPUT_ACTIVE

        left_held  = keyboard[bge.events.AKEY] == KX_ACTIVE
        right_held = keyboard[bge.events.DKEY] == KX_ACTIVE

        # Steering
        if left_held and not right_held:
            self.steering += self.steer_accel
        elif right_held and not left_held:
            self.steering -= self.steer_accel
        else:
            self.steering *= self.steer_decay
            if abs(self.steering) < 0.0001:
                self.steering = 0.0

        self.steering = max(-self.max_steer, min(self.max_steer, self.steering))

        # Movement
        self.object.applyMovement((0, self.base_speed, 0), True)
        self.object.applyRotation((0, 0, self.steering), True)

        # Position
        car_pos = self.object.worldPosition
        origin  = self.origin.worldPosition

        rel_x = float(car_pos.x - origin.x)
        rel_y = float(car_pos.y - origin.y)

        # Rotation
        rot_z = math.degrees(self.object.worldOrientation.to_euler('XYZ').z)

        # Normalize steering
        steering_norm = round(self.steering / self.max_steer, 6)

        frame_name = f"frame_{self.frame:06d}"

        self.csv_writer.writerow([
            frame_name,
            steering_norm,
            rel_x,
            rel_y,
            rot_z
        ])

        self.csv_file.flush()

        if self.frame % 10 == 0:
            print(f"[LOG] {frame_name} | steer={steering_norm:+.3f}")

        self.frame += 1

    def end(self):
        if hasattr(self, "csv_file"):
            self.csv_file.close()

        print(f"=== LOGGER STOPPED | frames={self.frame} ===")