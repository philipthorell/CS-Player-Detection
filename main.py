import os
import signal
import subprocess
import sys
import threading

# gtk4-layer-shell must be preloaded before libwayland-client, otherwise
# layer surfaces silently fall back to ordinary toplevel windows.
_LAYER_SHELL_LIB = "/usr/lib/libgtk4-layer-shell.so"
if _LAYER_SHELL_LIB not in os.environ.get("LD_PRELOAD", ""):
    os.environ["LD_PRELOAD"] = (
        f"{_LAYER_SHELL_LIB}:{os.environ.get('LD_PRELOAD', '')}".rstrip(":")
    )
    os.execv(sys.executable, [sys.executable, *sys.argv])

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gdk, GLib, Gtk, Gtk4LayerShell  # noqa: E402

import cairo  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

MODEL_PATH = "runs/detect/models/csgo-yolov5-2/weights/best.pt"
MODEL_CONFIDENCE = 0.6
OUTPUT_MONITOR = "DP-1"
MONITOR_SIZE = (1920, 1080)


def grab(output: str) -> np.ndarray:
    data = subprocess.run(
        ["grim", "-t", "ppm", "-o", output, "-"],
        check=True, capture_output=True,
    ).stdout
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class Overlay(Gtk.ApplicationWindow):
    def __init__(self, app, output):
        super().__init__(application=app)
        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.OVERLAY)
        for edge in (
            Gtk4LayerShell.Edge.TOP,
            Gtk4LayerShell.Edge.BOTTOM,
            Gtk4LayerShell.Edge.LEFT,
            Gtk4LayerShell.Edge.RIGHT,
        ):
            Gtk4LayerShell.set_anchor(self, edge, True)
        Gtk4LayerShell.set_keyboard_mode(self, Gtk4LayerShell.KeyboardMode.NONE)
        Gtk4LayerShell.set_exclusive_zone(self, -1)

        for mon in Gdk.Display.get_default().get_monitors():
            if mon.get_connector() == output:
                Gtk4LayerShell.set_monitor(self, mon)
                break

        self.boxes: list[tuple[float, float, float, float, str, float]] = []
        self.lock = threading.Lock()
        self.area = Gtk.DrawingArea()
        self.area.set_draw_func(self.on_draw)
        self.set_child(self.area)
        self.connect("realize", self._make_clickthrough)

    def _make_clickthrough(self, _):
        self.get_surface().set_input_region(cairo.Region())

    def on_draw(self, _area, cr, _w, _h):
        with self.lock:
            boxes = list(self.boxes)
        cr.set_line_width(2)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(14)

        # Draw the boxes with the label and confidence
        for x, y, bw, bh, label, conf in boxes:
            cr.set_source_rgba(0, 1, 0, 1)
            cr.rectangle(x, y, bw, bh)
            cr.stroke()
            cr.move_to(x, max(y - 4, 14))
            cr.show_text(f"{label} {conf:.2f}")

        # Draw outline of the monitor
        cr.set_source_rgba(0, 1, 0, 1)
        cr.rectangle(0, 0, MONITOR_SIZE[0], MONITOR_SIZE[1])
        cr.stroke()

    def update(self, boxes):
        with self.lock:
            self.boxes = boxes
        self.area.queue_draw()
        return False


def detect_loop(overlay: Overlay, output_monitor: str):
    model = YOLO(MODEL_PATH)
    while True:
        frame = grab(output_monitor)
        results = model(frame, verbose=False, conf=MODEL_CONFIDENCE, device="cuda")[0]
        names = results.names
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append((
                x1, y1, x2 - x1, y2 - y1,
                names[int(box.cls[0].item())],
                box.conf[0].item(),
            ))
        GLib.idle_add(overlay.update, out)


def on_activate(app):
    css = Gtk.CssProvider()
    css.load_from_string("window { background: transparent; }")
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        css,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )
    overlay = Overlay(app, OUTPUT_MONITOR)
    overlay.present()
    threading.Thread(target=detect_loop, args=(overlay, OUTPUT_MONITOR), daemon=True).start()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = Gtk.Application(application_id="com.cs.detect.overlay")
    app.connect("activate", on_activate)
    app.run([])


if __name__ == "__main__":
    main()
