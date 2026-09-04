#!/usr/bin/env python3
"""Calibrate camera positions around a 70 x 40 cm arena."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
import tkinter as tk
from typing import Any

from camera_grid import Camera, EventRecorder, MotionDetector, discover


CAGE_WIDTH_CM = 70.0
CAGE_HEIGHT_CM = 40.0
FOV_DEGREES = 50.0
ARENA_MARGIN_SCALE = 1.10
DEFAULT_LAYOUT = (
    (17.5, 0.0), (35.0, 0.0), (52.5, 0.0),
    (17.5, 40.0), (35.0, 40.0), (52.5, 40.0),
    (0.0, 13.33), (0.0, 26.67),
    (70.0, 13.33), (70.0, 26.67),
)
# Inner wall corners in the 480x270 overhead preview: top-left clockwise to
# bottom-left.
DEFAULT_ARENA_CORNERS = ((136.5, 86.6), (360.0, 86.6),
                         (363.0, 231.8), (135.0, 230.6))


def clip_polygon_to_rectangle(
        points: list[tuple[float, float]], left: float, top: float,
        right: float, bottom: float) -> list[tuple[float, float]]:
    """Clip a polygon to an axis-aligned rectangle."""
    def clip(edge: str, value: float,
             polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not polygon:
            return []

        def inside(point: tuple[float, float]) -> bool:
            x, y = point
            return (x >= value if edge == "left" else
                    x <= value if edge == "right" else
                    y >= value if edge == "top" else y <= value)

        def intersection(start: tuple[float, float],
                         end: tuple[float, float]) -> tuple[float, float]:
            sx, sy = start
            ex, ey = end
            if edge in ("left", "right"):
                ratio = (value - sx) / (ex - sx)
                return value, sy + ratio * (ey - sy)
            ratio = (value - sy) / (ey - sy)
            return sx + ratio * (ex - sx), value

        result: list[tuple[float, float]] = []
        previous = polygon[-1]
        for current in polygon:
            if inside(current):
                if not inside(previous):
                    result.append(intersection(previous, current))
                result.append(current)
            elif inside(previous):
                result.append(intersection(previous, current))
            previous = current
        return result

    result = points
    for edge, value in (("left", left), ("right", right),
                        ("top", top), ("bottom", bottom)):
        result = clip(edge, value, result)
    return result


def expand_corners(corners: list[list[float]],
                   scale: float = ARENA_MARGIN_SCALE) -> list[list[float]]:
    """Expand a calibration quadrilateral about its centroid."""
    center_x = sum(point[0] for point in corners) / len(corners)
    center_y = sum(point[1] for point in corners) / len(corners)
    return [[center_x + (x - center_x) * scale,
             center_y + (y - center_y) * scale] for x, y in corners]


def map_pixel_to_arena(x: float, y: float,
                       corners: list[list[float]]) -> tuple[float, float]:
    """Perspective-map one overhead-preview pixel into cage centimetres."""
    import cv2
    import numpy as np
    source = np.asarray(corners, dtype=np.float32)
    target = np.asarray(((0, 0), (CAGE_WIDTH_CM, 0),
                         (CAGE_WIDTH_CM, CAGE_HEIGHT_CM),
                         (0, CAGE_HEIGHT_CM)), dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, target)
    point = np.asarray([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, transform)[0, 0]
    return float(mapped[0]), float(mapped[1])


def snap_to_cage(x: float, y: float) -> tuple[float, float, float]:
    """Snap a position to the closest wall and return its inward angle."""
    x = min(CAGE_WIDTH_CM, max(0.0, x))
    y = min(CAGE_HEIGHT_CM, max(0.0, y))
    distances = (y, CAGE_HEIGHT_CM - y, x, CAGE_WIDTH_CM - x)
    wall = min(range(4), key=distances.__getitem__)
    if wall == 0:
        return x, 0.0, 90.0
    if wall == 1:
        return x, CAGE_HEIGHT_CM, -90.0
    if wall == 2:
        return 0.0, y, 0.0
    return CAGE_WIDTH_CM, y, 180.0


def point_in_fov(camera_x: float, camera_y: float, angle_degrees: float,
                 point_x: float, point_y: float,
                 fov_degrees: float = FOV_DEGREES,
                 max_depth: float | None = None) -> bool:
    dx, dy = point_x - camera_x, point_y - camera_y
    if dx == 0 and dy == 0:
        return True
    point_angle = math.degrees(math.atan2(dy, dx))
    difference = (point_angle - angle_degrees + 180) % 360 - 180
    if abs(difference) > fov_degrees / 2:
        return False
    if max_depth is not None:
        direction = math.radians(angle_degrees)
        forward_depth = dx * math.cos(direction) + dy * math.sin(direction)
        if forward_depth > max_depth:
            return False
    return True


def stable_index0(device: str) -> str:
    import glob
    import os
    target = os.path.realpath(device)
    matches = [path for path in glob.glob("/dev/v4l/by-path/*-video-index0")
               if os.path.realpath(path) == target and "-usb-" in path]
    return sorted(matches)[0] if matches else device


class MjpegPreview:
    def __init__(self, device: str, width: int = 480, height: int = 270):
        self.device, self.width, self.height = device, width, height
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.paused = threading.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        size = self.width * self.height * 3
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-fflags", "nobuffer", "-flags", "low_delay",
            # Keep the night-vision sentinel at its full compressed frame rate;
            # only scale the decoded image for the UI and motion analysis.
            "-f", "v4l2", "-input_format", "mjpeg", "-framerate", "30",
            "-video_size", "1920x1080", "-i", self.device, "-an",
            "-vf", f"scale={self.width}:{self.height}:flags=fast_bilinear",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=size)
        assert self.process.stdout is not None
        while not self.stop.is_set():
            if self.paused.is_set():
                self.stop.wait(0.1)
                continue
            frame = self.process.stdout.read(size)
            if len(frame) != size:
                break
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(frame)

    def close(self) -> None:
        self.stop.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()


class BufferedPreview:
    """Decode completed RAM-ring segments without reopening the camera."""

    def __init__(self, directory: Path, width: int = 480, height: int = 270):
        self.directory, self.width, self.height = directory, width, height
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        last: Path | None = None
        frame_size = self.width * self.height * 3
        while not self.stop.is_set():
            segments = sorted(path for path in self.directory.glob("*.ts")
                              if path.stat().st_size > 0)
            # The newest file may still be written by the capture process.
            segment = segments[-2] if len(segments) >= 2 else None
            if segment is None or segment == last:
                self.stop.wait(0.15)
                continue
            last = segment
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", str(segment), "-an", "-vf",
                f"fps=5,scale={self.width}:{self.height}:flags=fast_bilinear",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
            ]
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL,
                                            bufsize=frame_size)
            assert self.process.stdout is not None
            while not self.stop.is_set():
                frame = self.process.stdout.read(frame_size)
                if len(frame) != frame_size:
                    break
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(frame)
            self._terminate()

    def _terminate(self) -> None:
        process, self.process = self.process, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()

    def close(self) -> None:
        self.stop.set()
        self._terminate()


class CompressedCapture:
    """Continuously buffer one camera's untouched 1080p H.264 stream."""

    def __init__(self, camera: Camera, number: int, event_dir: Path,
                 buffer_dir: Path, pre_seconds: float, post_seconds: float,
                 startup_delay: float):
        self.camera, self.number = camera, number
        self.startup_delay = startup_delay
        self.recorder = EventRecorder(event_dir, buffer_dir, number,
                                      pre_seconds, post_seconds)
        self.stop = threading.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        if self.stop.wait(self.startup_delay):
            return
        while not self.stop.is_set():
            command = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-f", "v4l2", "-input_format", "h264", "-framerate", "30",
                "-video_size", "1920x1080", "-i", self.camera.device, "-an",
                "-map", "0:v:0", "-c:v", "copy", "-f", "segment",
                "-segment_time", "1", "-segment_atclocktime", "1",
                "-reset_timestamps", "1", "-strftime", "1",
                self.recorder.segment_pattern,
            ]
            self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
            while not self.stop.wait(0.25):
                process = self.process
                if process is None or process.poll() is not None:
                    break
            self._terminate()
            self.stop.wait(2)

    def update(self, active: bool, now: float) -> None:
        self.recorder.update(active, now)

    def pause(self) -> None:
        self.paused.set()
        self._terminate()

    def resume(self) -> None:
        self.paused.clear()

    @property
    def online(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _terminate(self) -> None:
        process, self.process = self.process, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self) -> None:
        self.stop.set()
        self._terminate()
        self.thread.join(timeout=2)
        self.recorder.close()


class SpatialLayout:
    def __init__(self, root: tk.Tk, cameras: list[Camera], sentinel: Camera,
                 layout_file: Path, event_dir: Path, buffer_dir: Path,
                 pre_seconds: float, post_seconds: float):
        self.root, self.cameras, self.sentinel = root, cameras, sentinel
        self.layout_file = layout_file
        self.margin, self.scale = 70, 12.0
        self.cage_left, self.cage_top = self.margin, self.margin
        self.cage_right = self.cage_left + CAGE_WIDTH_CM * self.scale
        self.cage_bottom = self.cage_top + CAGE_HEIGHT_CM * self.scale
        self.config = self._read_config()
        self.recording_enabled = bool(self.config.get("recording_enabled", True))
        self.sentinel_flipped = bool(self.config.get("sentinel_flipped", False))
        saved_flips = self.config.get("camera_flips", {})
        self.camera_flips = {
            camera.bus: bool(saved_flips.get(camera.bus, False))
            for camera in cameras
        }
        self.positions = self._load()
        self.arena_corners = [list(point) for point in self.config.get(
            "arena_corners", DEFAULT_ARENA_CORNERS)]
        self.drag_corner: int | None = None
        self.selected: int | None = None
        self.dragging: int | None = None
        self.detail_stream: BufferedPreview | None = None
        self.detail_image: tk.PhotoImage | None = None
        self.sentinel_stream = MjpegPreview(stable_index0(sentinel.device))
        self.sentinel_image: tk.PhotoImage | None = None
        self.motion_detector = MotionDetector(480, 270, 20, 1.0)
        self.motion_frames = 0
        self.motion_history: deque[bool] = deque(maxlen=30)
        self.box_hold_until = 0.0
        self.last_confirmed_boxes: list[tuple[int, int, int, int]] = []
        self.motion_boxes: list[tuple[float, float, float, float]] = []
        self.active_cameras: set[int] = set()
        self.test_point: tuple[float, float] | None = None
        self.captures = [
            CompressedCapture(camera, index + 1, event_dir, buffer_dir,
                              pre_seconds, post_seconds, index * 0.35)
            for index, camera in enumerate(cameras)
        ]
        for capture in self.captures:
            capture.recorder.set_enabled(self.recording_enabled)
        self.closed = False

        root.title("Mouse arena — spatial motion dispatcher")
        root.configure(bg="#111")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _event: self.close())
        root.bind("q", lambda _event: self.close())

        body = tk.Frame(root, bg="#111")
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, width=980, height=630, bg="#15191d",
                                highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        sidebar = tk.Frame(body, width=500, bg="#202020")
        sidebar.pack(side="right", fill="y", padx=(0, 8), pady=8)
        sidebar.pack_propagate(False)
        self.preview_title = tk.Label(
            sidebar, text="OVERHEAD SENTINEL", fg="#86d7ff", bg="#202020",
            font=("TkDefaultFont", 12, "bold"))
        self.preview_title.pack(pady=(10, 4))
        preview_frame = tk.Frame(sidebar, width=480, height=270, bg="#111")
        preview_frame.pack(padx=10)
        preview_frame.pack_propagate(False)
        self.sentinel_label = tk.Label(preview_frame, text="connecting…", fg="#aaa",
                                       bg="#111", width=480, height=270)
        self.sentinel_label.pack(fill="both", expand=True)
        self.flip_button = tk.Button(
            preview_frame, command=self._toggle_preview_flip,
            fg="white", bg="#3c5365", activeforeground="white",
            activebackground="#58758b", padx=5, pady=2, bd=1,
            cursor="hand2")
        self.flip_button.place(relx=1.0, x=-6, y=6, anchor="ne")
        self._update_flip_button()
        self.back_button = tk.Button(
            sidebar, text="← Back to motion controls",
            command=self._close_detail, fg="white", bg="#3c5365",
            activeforeground="white", activebackground="#58758b",
            padx=12, pady=5)
        self.motion_controls = tk.LabelFrame(
            sidebar, text=" Motion detection ", fg="white",
            bg="#202020", bd=1, padx=8, pady=6,
            font=("TkDefaultFont", 11, "bold"))
        self.motion_controls.pack(fill="x", padx=10, pady=(14, 4))
        settings = self.config.get("motion_settings", {})
        self.threshold_value = tk.IntVar(value=settings.get("pixel_delta", 20))
        self.percent_value = tk.DoubleVar(value=settings.get("changed_percent", 1.0))
        self.frames_value = tk.IntVar(value=settings.get("motion_frames", 3))
        self._scale(self.motion_controls, "Pixel delta", self.threshold_value, 1, 100, 1)
        self._scale(self.motion_controls, "Changed area %", self.percent_value, 0.1, 15, 0.1)
        self._scale(self.motion_controls, "Motion frames", self.frames_value, 1, 15, 1)
        self.recording_button = tk.Button(
            sidebar, command=self._toggle_recording,
            fg="white", activeforeground="white", relief="raised", bd=2,
            padx=12, pady=8, font=("TkDefaultFont", 11, "bold"),
            cursor="hand2")
        self.recording_button.pack(fill="x", padx=10, pady=(10, 2))
        self._update_recording_button()
        self.dispatch_status = tk.Label(
            sidebar, text="Waiting for motion", fg="#ddd", bg="#171717",
            justify="left", anchor="w", padx=10, pady=10,
            font=("TkDefaultFont", 10, "bold"))
        self.dispatch_status.pack(fill="x", padx=10, pady=8)
        tk.Label(sidebar,
                 text="Drag cameras to their physical wall positions.\n"
                      "Drag the four cyan handles onto the inner cage corners.\n"
                      "Click inside the cage to test coverage.\n"
                      "1080p streams remain compressed in RAM; only\n"
                      "the overhead sentinel is decoded continuously.",
                 fg="#bbb", bg="#202020", justify="left").pack(padx=12, pady=12,
                                                                  anchor="w")
        self.status = tk.Label(root, text="", fg="white", bg="#303840", anchor="w",
                               padx=10, pady=5)
        self.status.pack(fill="x")
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.sentinel_label.bind("<Button-1>", self._corner_press)
        self.sentinel_label.bind("<B1-Motion>", self._corner_drag)
        self.sentinel_label.bind("<ButtonRelease-1>", self._corner_release)
        self.sentinel_stream.start()
        for capture in self.captures:
            capture.start()
        self._draw()
        root.after(40, self._refresh)

    def _read_config(self) -> dict[str, Any]:
        try:
            loaded = json.loads(self.layout_file.read_text())
            return loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError, AttributeError):
            return {}

    def _load(self) -> dict[str, list[float]]:
        saved = self.config.get("positions", {})
        result = {}
        for index, camera in enumerate(self.cameras):
            result[camera.bus] = saved.get(camera.bus, list(DEFAULT_LAYOUT[index]))
        return result

    def _save(self) -> None:
        payload = {"cage_cm": [70, 40], "fov_degrees": FOV_DEGREES,
                   "sentinel_bus": self.sentinel.bus,
                   "sentinel_flipped": self.sentinel_flipped,
                   "camera_flips": self.camera_flips,
                   "recording_enabled": self.recording_enabled,
                   "positions": self.positions,
                   "arena_corners": self.arena_corners,
                   "motion_settings": {
                       "pixel_delta": self.threshold_value.get(),
                       "changed_percent": self.percent_value.get(),
                       "motion_frames": self.frames_value.get(),
                   }}
        self.layout_file.write_text(json.dumps(payload, indent=2) + "\n")

    def _toggle_preview_flip(self) -> None:
        if self.selected is None:
            self.sentinel_flipped = not self.sentinel_flipped
            self.motion_detector.previous = None
            self.motion_history.clear()
            self.last_confirmed_boxes.clear()
            self.box_hold_until = 0.0
        else:
            bus = self.cameras[self.selected].bus
            self.camera_flips[bus] = not self.camera_flips[bus]
        self._update_flip_button()
        self._save()

    def _update_flip_button(self) -> None:
        flipped = (self.sentinel_flipped if self.selected is None else
                   self.camera_flips[self.cameras[self.selected].bus])
        self.flip_button.configure(
            text="↻ 180°", bg="#8b5a2b" if flipped else "#3c5365")

    def _update_recording_button(self) -> None:
        if self.recording_enabled:
            self.recording_button.configure(
                text="Recording: ON", bg="#8b1e1e", activebackground="#a52a2a")
        else:
            self.recording_button.configure(
                text="Recording: OFF", bg="#444", activebackground="#555")

    def _toggle_recording(self) -> None:
        self.recording_enabled = not self.recording_enabled
        for capture in self.captures:
            capture.recorder.set_enabled(self.recording_enabled)
        self._update_recording_button()
        self._save()

    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.cage_left + x * self.scale, self.cage_top + y * self.scale

    def _to_cm(self, x: float, y: float) -> tuple[float, float]:
        return ((x - self.cage_left) / self.scale,
                (y - self.cage_top) / self.scale)

    def _scale(self, parent: tk.Widget, label: str, variable: tk.Variable,
               minimum: float, maximum: float, resolution: float) -> None:
        row = tk.Frame(parent, bg="#202020")
        row.pack(fill="x")
        tk.Label(row, text=label, width=15, anchor="w", fg="white",
                 bg="#202020").pack(side="left")
        scale = tk.Scale(row, from_=minimum, to=maximum, resolution=resolution,
                         orient="horizontal", length=285, variable=variable,
                         showvalue=True, fg="white", bg="#202020",
                         troughcolor="#555", highlightthickness=0)
        scale.pack(side="left")
        scale.bind("<ButtonRelease-1>", lambda _event: self._save())

    def _draw(self, test_point: tuple[float, float] | None = None) -> None:
        if test_point is not None:
            self.test_point = test_point
        self.canvas.delete("all")
        self.canvas.create_text(self.cage_left, 28, anchor="w", fill="white",
                                text=f"70 × 40 cm arena — {FOV_DEGREES:.0f}° horizontal FoV",
                                font=("TkDefaultFont", 14, "bold"))
        self.canvas.create_rectangle(self.cage_left, self.cage_top,
                                     self.cage_right, self.cage_bottom,
                                     fill="#26342e", outline="#b8c8bd", width=3)
        self.canvas.create_text((self.cage_left + self.cage_right) / 2,
                                (self.cage_top + self.cage_bottom) / 2,
                                text="TOP-DOWN SENTINEL", fill="#7ca691",
                                font=("TkDefaultFont", 16, "bold"))
        active = self.active_cameras
        if self.test_point:
            tx, ty = self.test_point
            cx, cy = self._to_canvas(tx, ty)
            self.canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7,
                                    fill="#ffdf5d", outline="white", width=2)
        for index, camera in enumerate(self.cameras):
            x, y = self.positions[camera.bus]
            x, y, angle = snap_to_cage(x, y)
            covered = index in active
            half = math.radians(FOV_DEGREES / 2)
            direction = math.radians(angle)
            is_end_camera = x == 0.0 or x == CAGE_WIDTH_CM
            # End cameras only contribute through the nearest half of the
            # 70 cm cage. Convert axial depth to ray length for drawing.
            length_cm = ((CAGE_WIDTH_CM / 2) / math.cos(half)
                         if is_end_camera else 62)
            ox, oy = self._to_canvas(x, y)
            cone = [(x, y)]
            for ray in (direction - half, direction + half):
                cone.append((x + math.cos(ray) * length_cm,
                             y + math.sin(ray) * length_cm))
            clipped = clip_polygon_to_rectangle(
                cone, 0.0, 0.0, CAGE_WIDTH_CM, CAGE_HEIGHT_CM)
            points = [coordinate for point in clipped
                      for coordinate in self._to_canvas(*point)]
            self.canvas.create_polygon(points, fill="#547c69" if covered else "#344940",
                                       outline="#77aa91", stipple="gray50")
        for x, y, width, height in self.motion_boxes:
            left, top = self._to_canvas(x, y)
            right, bottom = self._to_canvas(x + width, y + height)
            self.canvas.create_rectangle(left, top, right, bottom,
                                         outline="#ff4d4d", width=4)
        for index, camera in enumerate(self.cameras):
            x, y = self.positions[camera.bus]
            ox, oy = self._to_canvas(x, y)
            selected = index == self.selected
            covered = index in active
            online = self.captures[index].online
            self.canvas.create_oval(ox - 17, oy - 17, ox + 17, oy + 17,
                                    fill=("#6b5540" if not online else
                                          "#e03e31" if covered else
                                          "#d65c4a" if selected else "#3479a8"),
                                    outline="#ff9f43" if not online else "white",
                                    width=3, tags=("camera", f"cam{index}"))
            self.canvas.create_text(ox, oy, text=str(index + 1), fill="white",
                                    font=("TkDefaultFont", 11, "bold"),
                                    tags=("camera", f"cam{index}"))
        if self.test_point:
            self.status.configure(text="Dispatcher selection: " +
                                  (", ".join(f"Camera {n + 1}" for n in sorted(active))
                                   if active else "no camera"))

    def _camera_at(self, x: float, y: float) -> int | None:
        for index, camera in enumerate(self.cameras):
            cx, cy = self._to_canvas(*self.positions[camera.bus])
            if (x - cx) ** 2 + (y - cy) ** 2 <= 24 ** 2:
                return index
        return None

    def _press(self, event: tk.Event[Any]) -> None:
        index = self._camera_at(event.x, event.y)
        if index is not None:
            if self.selected == index:
                self.dragging = None
                self._close_detail()
                return
            self.dragging = index
            self._select(index)
            return
        x, y = self._to_cm(event.x, event.y)
        if 0 <= x <= CAGE_WIDTH_CM and 0 <= y <= CAGE_HEIGHT_CM:
            self.active_cameras = self._coverage([(x, y)])
            self._draw((x, y))

    def _drag(self, event: tk.Event[Any]) -> None:
        if self.dragging is None:
            return
        x, y = self._to_cm(event.x, event.y)
        x, y, _angle = snap_to_cage(x, y)
        self.positions[self.cameras[self.dragging].bus] = [x, y]
        self._draw()

    def _release(self, _event: tk.Event[Any]) -> None:
        if self.dragging is not None:
            self.dragging = None
            self._save()

    def _select(self, index: int) -> None:
        if self.selected == index and self.detail_stream:
            return
        self._close_detail(redraw=False)
        self.selected = index
        self.preview_title.configure(text=f"CAMERA {index + 1}")
        self.sentinel_label.configure(image="", text="loading buffered preview…")
        self._update_flip_button()
        self.detail_image = None
        self.motion_controls.pack_forget()
        self.back_button.pack(fill="x", padx=10, pady=(4, 0))
        self.detail_stream = BufferedPreview(
            self.captures[index].recorder.rolling, 480, 270)
        self.detail_stream.start()
        self.dispatch_status.configure(
            text=f"Camera {index + 1} identification preview — buffer continues")
        self._draw()

    def _close_detail(self, redraw: bool = True) -> None:
        stream = self.detail_stream
        self.detail_stream = None
        if stream:
            stream.close()
        self.selected = None
        self.detail_image = None
        if hasattr(self, "preview_title"):
            self.preview_title.configure(text="OVERHEAD SENTINEL")
            self.sentinel_label.configure(image="", text="connecting…")
            self._update_flip_button()
        if hasattr(self, "back_button"):
            self.back_button.pack_forget()
            self.motion_controls.pack(fill="x", padx=10, pady=(14, 4),
                                      before=self.dispatch_status)
        if redraw:
            self._draw()

    def _corner_press(self, event: tk.Event[Any]) -> None:
        distances = [(event.x - x) ** 2 + (event.y - y) ** 2
                     for x, y in self.arena_corners]
        nearest = min(range(4), key=distances.__getitem__)
        if distances[nearest] <= 24 ** 2:
            self.drag_corner = nearest

    def _corner_drag(self, event: tk.Event[Any]) -> None:
        if self.drag_corner is None:
            return
        self.arena_corners[self.drag_corner] = [
            min(479.0, max(0.0, float(event.x))),
            min(269.0, max(0.0, float(event.y))),
        ]

    def _corner_release(self, _event: tk.Event[Any]) -> None:
        if self.drag_corner is not None:
            self.drag_corner = None
            self._save()

    def _coverage(self, points: list[tuple[float, float]]) -> set[int]:
        active: set[int] = set()
        for index, camera in enumerate(self.cameras):
            x, y = self.positions[camera.bus]
            x, y, angle = snap_to_cage(x, y)
            max_depth = (CAGE_WIDTH_CM / 2
                         if x == 0.0 or x == CAGE_WIDTH_CM else None)
            if any(point_in_fov(x, y, angle, px, py,
                                max_depth=max_depth) for px, py in points):
                active.add(index)
        return active

    def _process_motion(self, frame: bytes) -> bytes:
        self.motion_detector.configure(self.threshold_value.get(),
                                       self.percent_value.get())
        changed = self.motion_detector.changed_percent(frame)
        effective_corners = expand_corners(self.arena_corners)
        polygon = self.motion_detector.np.asarray(
            effective_corners, dtype=self.motion_detector.np.float32)
        self.motion_detector.boxes = [
            box for box in self.motion_detector.boxes
            if self.motion_detector.cv2.pointPolygonTest(
                polygon, (box[0] + box[2] / 2, box[1] + box[3] / 2), False) >= 0
        ]
        local_motion = (self.motion_detector.is_motion(changed)
                        and bool(self.motion_detector.boxes))
        self.motion_history.append(local_motion)
        required = self.frames_value.get()
        # Require N positive frames in a short 2N-frame window instead of N
        # perfectly consecutive frames. A mouse often produces alternating
        # strong/weak differences as it crosses a uniform floor.
        window = list(self.motion_history)[-max(required, required * 2):]
        self.motion_frames = sum(window)
        sustained = self.motion_frames >= required
        now_monotonic = time.monotonic()
        if sustained and self.motion_detector.boxes:
            self.last_confirmed_boxes = list(self.motion_detector.boxes)
            self.box_hold_until = now_monotonic + 1.5
        boxes = (self.last_confirmed_boxes
                 if now_monotonic < self.box_hold_until else [])
        self.motion_boxes = []
        for x, y, width, height in boxes:
            mapped = [map_pixel_to_arena(px, py, effective_corners)
                      for px, py in ((x, y), (x + width, y),
                                     (x + width, y + height), (x, y + height))]
            xs, ys = [point[0] for point in mapped], [point[1] for point in mapped]
            left, right = max(0.0, min(xs)), min(CAGE_WIDTH_CM, max(xs))
            top, bottom = max(0.0, min(ys)), min(CAGE_HEIGHT_CM, max(ys))
            if right > left and bottom > top:
                self.motion_boxes.append((left, top, right - left, bottom - top))
        # Corners and edge midpoints avoid losing a camera when the mouse box
        # straddles the boundary of two adjacent FoV cones.
        points: list[tuple[float, float]] = []
        for x, y, width, height in self.motion_boxes:
            points.extend([
                (x + width / 2, y + height / 2),
                (x, y), (x + width, y),
                (x, y + height), (x + width, y + height),
                (x + width / 2, y), (x + width / 2, y + height),
                (x, y + height / 2), (x + width, y + height / 2),
            ])
        self.test_point = points[0] if points else None
        self.active_cameras = self._coverage(points) if points else set()
        available = {index for index in self.active_cameras
                     if self.captures[index].online}
        recording = (self.recording_enabled and bool(boxes)
                     and len(available) >= 3)
        now = time.time()
        for index, capture in enumerate(self.captures):
            capture.update(recording and index in available, now)
        selected = ", ".join(str(index + 1) for index in sorted(self.active_cameras))
        offline = [str(index + 1) for index, capture in enumerate(self.captures)
                   if not capture.online]
        self.dispatch_status.configure(
            text=(f"Changed: {changed:.2f}%   Motion: "
                  f"{min(self.motion_frames, required)}/{required} frames\n"
                  f"FoV cameras: {selected or 'none'}\n"
                  f"Recording: {'DISABLED' if not self.recording_enabled else 'YES' if recording else 'armed'}"
                  f"{'   Offline: ' + ', '.join(offline) if offline else ''}"),
            fg="#ff7676" if recording else "#ddd")
        self._draw()
        # Use the same held boxes in the video and arena views.
        self.motion_detector.boxes = list(boxes)
        annotated = self.motion_detector.preview(frame, 480, 270)
        rgb = self.motion_detector.np.frombuffer(
            annotated, dtype=self.motion_detector.np.uint8).reshape(270, 480, 3).copy()
        polygon = self.motion_detector.np.asarray(
            self.arena_corners, dtype=self.motion_detector.np.int32)
        self.motion_detector.cv2.polylines(rgb, [polygon], True, (30, 220, 255), 2)
        expanded = self.motion_detector.np.asarray(
            effective_corners, dtype=self.motion_detector.np.int32)
        self.motion_detector.cv2.polylines(rgb, [expanded], True, (255, 210, 40), 2)
        for index, (x, y) in enumerate(self.arena_corners, 1):
            center = (round(x), round(y))
            self.motion_detector.cv2.circle(rgb, center, 7, (30, 220, 255), -1)
            self.motion_detector.cv2.putText(
                rgb, str(index), (center[0] + 8, center[1] - 5),
                self.motion_detector.cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (30, 220, 255), 1, self.motion_detector.cv2.LINE_AA)
        return rgb.tobytes()

    @staticmethod
    def _photo(frame: bytes, width: int, height: int) -> tk.PhotoImage:
        return tk.PhotoImage(data=f"P6 {width} {height} 255\n".encode() + frame,
                             format="PPM")

    def _refresh(self) -> None:
        try:
            frame = self.sentinel_stream.frames.get_nowait()
            if self.sentinel_flipped:
                frame = self.motion_detector.np.frombuffer(
                    frame, dtype=self.motion_detector.np.uint8
                ).reshape(270, 480, 3)[::-1, ::-1].tobytes()
            frame = self._process_motion(frame)
            self.sentinel_image = self._photo(frame, 480, 270)
            if self.selected is None:
                self.sentinel_label.configure(image=self.sentinel_image, text="")
        except queue.Empty:
            pass
        if self.detail_stream:
            try:
                frame = self.detail_stream.frames.get_nowait()
                assert self.selected is not None
                if self.camera_flips[self.cameras[self.selected].bus]:
                    frame = self.motion_detector.np.frombuffer(
                        frame, dtype=self.motion_detector.np.uint8
                    ).reshape(270, 480, 3)[::-1, ::-1].tobytes()
                self.detail_image = self._photo(frame, 480, 270)
                self.sentinel_label.configure(image=self.detail_image, text="")
            except queue.Empty:
                pass
        if not self.closed:
            self.root.after(40, self._refresh)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._save()
        if self.detail_stream:
            self.detail_stream.close()
            self.detail_stream = None
        # Remove the window immediately. Stopping ten V4L2/FFmpeg processes
        # serially can otherwise make Tk appear frozen for several seconds.
        cleanup = threading.Thread(target=self._shutdown, daemon=False)
        cleanup.start()
        self.root.destroy()

    def _shutdown(self) -> None:
        self.sentinel_stream.close()
        workers = [threading.Thread(target=capture.close, daemon=False)
                   for capture in self.captures]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-file", default="camera_layout.json")
    parser.add_argument("--event-dir", default="/mnt/ssd/camera-events")
    parser.add_argument("--buffer-dir", default="/dev/shm/camera-grid")
    parser.add_argument("--pre-seconds", type=float, default=5)
    parser.add_argument("--post-seconds", type=float, default=10)
    args = parser.parse_args()
    cameras, rejected = discover()
    sentinels = [camera for camera in rejected
                 if "MJPG" in camera.formats and "Arducam" not in camera.name]
    if len(cameras) != 10 or not sentinels:
        raise SystemExit(f"Expected 10 H.264 perimeter cameras and one MJPEG "
                         f"sentinel; found {len(cameras)} and {len(sentinels)}")
    sentinel = sentinels[-1]
    sentinel = Camera(stable_index0(sentinel.device), sentinel.name,
                      sentinel.bus, sentinel.formats)
    root = tk.Tk()
    viewer = SpatialLayout(root, cameras, sentinel, Path(args.layout_file),
                           Path(args.event_dir), Path(args.buffer_dir),
                           args.pre_seconds, args.post_seconds)
    signal.signal(signal.SIGINT, lambda *_args: root.after(0, viewer.close))
    signal.signal(signal.SIGTERM, lambda *_args: root.after(0, viewer.close))
    try:
        root.mainloop()
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
