# feedback_display_udp_SL8_paired.py
# Separate lightweight live biofeedback display for VETTA predicted vGRF.
#
# Run this script in a separate terminal BEFORE running the acquisition script:
#     python feedback_display_udp.py
#
# The acquisition script sends small UDP messages to 127.0.0.1:5055 whenever a
# new predicted first-peak vGRF is calculated. This script receives those
# messages and updates a simple two-bar display.
#
# This script does not open COM3, does not read sensors, and does not save data.
# If this display crashes or is closed, the acquisition script should continue.

import json
import socket
import tkinter as tk
from collections import deque
from datetime import datetime
import time

# -----------------------------
# Display/settings
# -----------------------------
UDP_HOST = "127.0.0.1"
UDP_PORT = 5055
TARGET_VGRF = 1.15
ROLLING_WINDOW_STEPS = 2       # Change to 1 for fastest display response; 2 mimics MATLAB-style smoothing.
DISPLAY_MIN_UPDATE_MS = 150  # Rate-limit visual updates so bars do not flicker/change too fast.
Y_MIN = 0.90
Y_MAX = 1.30
POLL_MS = 20                   # How often the GUI checks for new UDP messages.

class UDPBiofeedbackDisplay:
    """Display-only application for predicted first-peak vGRF feedback.

    Expected UDP message format from acquisition script:
        {"type":"step", "side":"left", "value":1.12, "step_id":"L3", "time":12.34}

    The display uses a MATLAB-like paired update: it updates both bars only when
    both legs have at least ROLLING_WINDOW_STEPS values available.
    """

    def __init__(self):
        self.left_peaks = deque(maxlen=ROLLING_WINDOW_STEPS)
        self.right_peaks = deque(maxlen=ROLLING_WINDOW_STEPS)
        self.last_left_value = None
        self.last_right_value = None
        self.message_count = 0
        self.left_new_since_update = False
        self.right_new_since_update = False
        self.last_display_update_time = 0.0

        # UDP socket is non-blocking so the GUI never waits for network data.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.sock.setblocking(False)

        # Tkinter window.
        self.root = tk.Tk()
        self.root.title("VETTA Predicted vGRF Biofeedback")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.width = 560
        self.height = 470
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="white")
        self.canvas.pack(padx=10, pady=10)

        self.status_var = tk.StringVar(value=f"Listening on UDP {UDP_HOST}:{UDP_PORT} ...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Arial", 11))
        self.status_label.pack(pady=(0, 8))

        self.top_margin = 45
        self.bottom_margin = 70
        self.left_x_center = 185
        self.right_x_center = 385
        self.bar_width = 85
        self.plot_bottom = self.height - self.bottom_margin
        self.plot_height = self.height - self.top_margin - self.bottom_margin

        self._draw_static_elements()
        self._create_dynamic_elements()

        self.running = True
        self.root.after(POLL_MS, self.poll_udp)

    def y_to_canvas(self, value):
        """Convert vGRF value to canvas y coordinate."""
        value = max(Y_MIN, min(Y_MAX, float(value)))
        frac = (value - Y_MIN) / (Y_MAX - Y_MIN)
        return self.plot_bottom - frac * self.plot_height

    def _draw_static_elements(self):
        # Title
        self.canvas.create_text(
            self.width // 2, 18,
            text="Real-Time Predicted First-Peak vGRF",
            font=("Arial", 15, "bold")
        )

        # Axes
        self.canvas.create_line(75, self.top_margin, 75, self.plot_bottom, width=2)
        self.canvas.create_line(75, self.plot_bottom, self.width - 55, self.plot_bottom, width=2)

        # Y ticks
        tick_values = [Y_MIN, round((Y_MIN + TARGET_VGRF) / 2, 2), TARGET_VGRF, round((TARGET_VGRF + Y_MAX) / 2, 2), Y_MAX]
        seen = set()
        for val in tick_values:
            if val in seen:
                continue
            seen.add(val)
            y = self.y_to_canvas(val)
            self.canvas.create_line(68, y, 82, y, width=1)
            self.canvas.create_text(43, y, text=f"{val:.2f}", font=("Arial", 10))

        # Target line
        target_y = self.y_to_canvas(TARGET_VGRF)
        self.canvas.create_line(75, target_y, self.width - 55, target_y, fill="red", width=4)
        self.canvas.create_text(
            self.width - 135, target_y - 12,
            text=f"Target = {TARGET_VGRF:.2f} BW",
            fill="red",
            font=("Arial", 11, "bold")
        )

        # Bar labels
        self.canvas.create_text(self.left_x_center, self.plot_bottom + 25, text="Left", font=("Arial", 13, "bold"))
        self.canvas.create_text(self.right_x_center, self.plot_bottom + 25, text="Right", font=("Arial", 13, "bold"))
        self.canvas.create_text(25, self.top_margin + self.plot_height / 2, text="BW", font=("Arial", 11, "bold"), angle=90)

    def _create_dynamic_elements(self):
        self.left_bar = self.canvas.create_rectangle(
            self.left_x_center - self.bar_width // 2,
            self.plot_bottom,
            self.left_x_center + self.bar_width // 2,
            self.plot_bottom,
            fill="#2f77d0",
            outline="black"
        )
        self.right_bar = self.canvas.create_rectangle(
            self.right_x_center - self.bar_width // 2,
            self.plot_bottom,
            self.right_x_center + self.bar_width // 2,
            self.plot_bottom,
            fill="#2f77d0",
            outline="black"
        )
        self.left_text = self.canvas.create_text(self.left_x_center, self.plot_bottom - 12, text="--", font=("Arial", 16, "bold"))
        self.right_text = self.canvas.create_text(self.right_x_center, self.plot_bottom - 12, text="--", font=("Arial", 16, "bold"))
        self.last_message_text = self.canvas.create_text(
            self.width // 2, self.height - 20,
            text="No predicted steps received yet",
            font=("Arial", 10)
        )

    def add_step(self, side, value, step_id=None, acquisition_time=None):
        side = str(side).lower()
        value = float(value)
        if side == "left":
            self.left_peaks.append(value)
            self.last_left_value = value
            self.left_new_since_update = True
        elif side == "right":
            self.right_peaks.append(value)
            self.last_right_value = value
            self.right_new_since_update = True
        else:
            return

        self.message_count += 1
        now = datetime.now().strftime("%H:%M:%S")
        msg = f"Last: {side.upper()} {step_id or ''} = {value:.3f} BW"
        if acquisition_time is not None:
            msg += f" at t={float(acquisition_time):.2f}s"
        msg += f" | display time {now} | messages={self.message_count}"
        self.canvas.itemconfig(self.last_message_text, text=msg)

    def ready_to_update(self):
        return len(self.left_peaks) >= ROLLING_WINDOW_STEPS and len(self.right_peaks) >= ROLLING_WINDOW_STEPS

    def update_bars(self, force=False):
        if not self.ready_to_update():
            self.status_var.set(
                f"Waiting for {ROLLING_WINDOW_STEPS} step(s)/leg... "
                f"Left={len(self.left_peaks)}, Right={len(self.right_peaks)}"
            )
            return

        # MATLAB-like paired display behavior: wait until both legs have received
        # at least one new accepted step since the last display update. This avoids
        # rapid bar flicker if several messages arrive from one side before the
        # other side.
        if not force and not (self.left_new_since_update and self.right_new_since_update):
            self.status_var.set(
                f"Waiting for next paired update... "
                f"Left new={self.left_new_since_update}, Right new={self.right_new_since_update}"
            )
            return

        now_t = time.perf_counter()
        if not force and (now_t - self.last_display_update_time) * 1000.0 < DISPLAY_MIN_UPDATE_MS:
            return

        left_avg = sum(self.left_peaks) / len(self.left_peaks)
        right_avg = sum(self.right_peaks) / len(self.right_peaks)
        left_y = self.y_to_canvas(left_avg)
        right_y = self.y_to_canvas(right_avg)

        self.canvas.coords(
            self.left_bar,
            self.left_x_center - self.bar_width // 2,
            left_y,
            self.left_x_center + self.bar_width // 2,
            self.plot_bottom
        )
        self.canvas.coords(
            self.right_bar,
            self.right_x_center - self.bar_width // 2,
            right_y,
            self.right_x_center + self.bar_width // 2,
            self.plot_bottom
        )
        self.canvas.coords(self.left_text, self.left_x_center, left_y - 18)
        self.canvas.coords(self.right_text, self.right_x_center, right_y - 18)
        self.canvas.itemconfig(self.left_text, text=f"{left_avg:.2f}")
        self.canvas.itemconfig(self.right_text, text=f"{right_avg:.2f}")
        self.status_var.set(f"Rolling average of last {ROLLING_WINDOW_STEPS} step(s)/leg")

        self.left_new_since_update = False
        self.right_new_since_update = False
        self.last_display_update_time = now_t

    def poll_udp(self):
        if not self.running:
            return

        got_message = False
        while True:
            try:
                data, _addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                return

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception as e:
                self.status_var.set(f"Invalid UDP message: {e}")
                continue

            if msg.get("type") == "step":
                self.add_step(
                    msg.get("side"),
                    msg.get("value"),
                    step_id=msg.get("step_id"),
                    acquisition_time=msg.get("time"),
                )
                got_message = True

        if got_message:
            self.update_bars()

        self.root.after(POLL_MS, self.poll_udp)

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        print(f"Listening for VETTA feedback messages on UDP {UDP_HOST}:{UDP_PORT}")
        print("Start the acquisition script in another terminal. Close this window to stop the display only.")
        self.root.mainloop()

if __name__ == "__main__":
    app = UDPBiofeedbackDisplay()
    app.run()
