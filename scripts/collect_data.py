import bge
import csv
import os

class CarController(bge.types.KX_PythonComponent):

    args = {
        "save_dir": r"./data",
        "base_speed": 0.005,
        "steer_accel": 0.003,
        "steer_decay": 0.85,
        "max_steer": 0.06,
        "save_every": 1,

        # Data balance
        "dead_zone": 0.005,
        "straight_skip": 6,
    }

    def start(self, args):
        self.base_speed    = float(args.get("base_speed", 0.005))
        self.steer_accel   = float(args.get("steer_accel", 0.003))
        self.steer_decay   = float(args.get("steer_decay", 0.85))
        self.max_steer     = float(args.get("max_steer", 0.06))
        self.save_every    = int(args.get("save_every", 1))
        self.dead_zone     = float(args.get("dead_zone", 0.005))
        self.straight_skip = int(args.get("straight_skip", 6))
        self.save_dir      = args.get("save_dir")

        self.steering = 0.0
        self.frame = 0
        self.saved = 0
        self.straight_ctr = 0

        os.makedirs(self.save_dir, exist_ok=True)

        csv_path = os.path.join(self.save_dir, "labels.csv")
        self.csv_file = open(csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        if os.path.getsize(csv_path) == 0:
            self.csv_writer.writerow(["image", "steering"])

        print("=== Data Collector started ===")

    def update(self):
        keyboard = bge.logic.keyboard.events
        KX_ACTIVE = bge.logic.KX_INPUT_ACTIVE

        left_held  = keyboard[bge.events.AKEY] == KX_ACTIVE
        right_held = keyboard[bge.events.DKEY] == KX_ACTIVE

        # Steering logic
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

        # Save logic
        if self.frame % self.save_every == 0:
            is_straight = abs(self.steering) < self.dead_zone

            if is_straight:
                self.straight_ctr += 1
                should_save = (self.straight_ctr % self.straight_skip == 0)
            else:
                should_save = True

            if should_save:
                steering_norm = round(self.steering / self.max_steer, 6)

                img_name = f"frame_{self.saved:06d}.png"
                img_path = os.path.join(self.save_dir, img_name)

                bge.render.makeScreenshot(img_path)

                self.csv_writer.writerow([img_name, steering_norm])
                self.csv_file.flush()

                self.saved += 1

        self.frame += 1

    def end(self):
        self.csv_file.close()
        print(f"Saved: {self.saved} frames")