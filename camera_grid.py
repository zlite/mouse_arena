#!/usr/bin/env python3
"""Discover H.264 V4L2 cameras and display them in a Tk grid."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import glob
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Camera:
    device: str
    name: str
    bus: str
    formats: tuple[str, ...]

    @property
    def h264(self) -> bool:
        return "H264" in self.formats


def run_text(command: list[str], timeout: float = 4) -> str:
    try:
        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
        return result.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def parse_camera(device: str, details: str, formats: str) -> Camera:
    def field(label: str, fallback: str) -> str:
        match = re.search(rf"^\s*{label}\s*:\s*(.+?)\s*$", details, re.MULTILINE)
        return match.group(1) if match else fallback

    fourccs = tuple(dict.fromkeys(re.findall(r"'([A-Za-z0-9]{4})'", formats)))
    return Camera(
        device=device,
        name=field("Card type", os.path.basename(device)),
        bus=field("Bus info", "unknown bus"),
        formats=fourccs,
    )


def discover() -> tuple[list[Camera], list[Camera]]:
    supported, rejected = [], []
    candidates = sorted(glob.glob("/dev/video*"), key=device_sort_key)
    for device in candidates:
        # Raspberry Pi systems expose many codec/ISP nodes. Avoid probing those:
        # UVC control calls can be slow when ten cameras share USB hubs.
        if not is_uvc_device(device):
            continue
        formats = run_text(["v4l2-ctl", "-d", device, "--list-formats-ext"])
        # Empty-format UVC nodes are metadata/control endpoints.
        if "Type: Video Capture" not in formats:
            continue
        details = run_text(["v4l2-ctl", "-d", device, "--info"])
        camera = parse_camera(device, details, formats)
        if camera.h264:
            camera = Camera(stable_device_path(device), camera.name, camera.bus, camera.formats)
        (supported if camera.h264 else rejected).append(camera)
    return deduplicate(supported), rejected


def is_uvc_device(device: str) -> bool:
    name = os.path.basename(device)
    driver_link = f"/sys/class/video4linux/{name}/device/driver"
    try:
        return os.path.basename(os.path.realpath(driver_link)) == "uvcvideo"
    except OSError:
        return False


def stable_device_path(device: str) -> str:
    """Return the physical-port symlink for an endpoint when udev provides one."""
    target = os.path.realpath(device)
    matches = [
        path for path in glob.glob("/dev/v4l/by-path/*-video-index2")
        if os.path.realpath(path) == target and "-usb-" in path
    ]
    return sorted(matches)[0] if matches else device


def device_sort_key(path: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path)
    return (int(match.group(1)) if match else 1_000_000, path)


def deduplicate(cameras: list[Camera]) -> list[Camera]:
    """Keep one H.264 capture endpoint per physical USB bus path."""
    result: list[Camera] = []
    seen: set[str] = set()
    for camera in cameras:
        key = camera.bus if camera.bus != "unknown bus" else camera.device
        if key not in seen:
            seen.add(key)
            result.append(camera)
    return result


def controller_name(bus: str) -> str:
    """Return the xHCI controller portion of a V4L2 USB bus path."""
    match = re.match(r"usb-(.+?)-\d+(?:[.:].*)?$", bus)
    return match.group(1) if match else "unknown"


def root_port_name(bus: str) -> str:
    """Return the xHCI controller and root-port portion of a USB bus path."""
    match = re.match(r"usb-(.+?)-(\d+)(?:[.:].*)?$", bus)
    return f"{match.group(1)}-{match.group(2)}" if match else "unknown"


def video_filter(width: int, height: int, rotated: bool,
                 preview_fps: int | None = None) -> str:
    # Reduce scaling, color conversion, pipe traffic, and Tk work without
    # changing the camera's full-resolution capture frame rate.
    filters = [f"fps={preview_fps}"] if preview_fps else []
    if rotated:
        filters.extend(("hflip", "vflip"))
    filters.extend((
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=fast_bilinear",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
    ))
    return ",".join(filters)


def stream_overlay_text(stream: Any) -> str:
    parts = [stream.capture_size, f"{stream.measured_fps:.1f} fps"]
    if stream.motion_detector:
        if stream.recorder and not stream.recorder.enabled:
            parts.append("REC OFF")
        else:
            parts.append("REC" if stream.recorder and stream.recorder.recording
                         else f"{stream.changed_percent:.1f}%")
        if stream.coordinator:
            parts.append(
                f"{stream.motion_camera_count}/{stream.coordinator.minimum_cameras} cams")
    if stream.rotated:
        parts.append("180°")
    return "  •  ".join(parts)


class MotionDetector:
    """Detect sustained changes in reduced RGB preview frames."""

    def __init__(self, width: int, height: int, threshold: int,
                 minimum_percent: float):
        try:
            import cv2
            import numpy
        except ImportError as exc:
            raise RuntimeError(
                "motion detection requires OpenCV: "
                "sudo apt install python3-opencv"
            ) from exc
        self.cv2, self.np = cv2, numpy
        self.width, self.height = width, height
        self.threshold = 0
        self.minimum_percent = 0.0
        self.minimum_pixels = 1
        self.configure(threshold, minimum_percent)
        self.previous: Any | None = None
        self.boxes: list[tuple[int, int, int, int]] = []

    def configure(self, threshold: int | None = None,
                  minimum_percent: float | None = None) -> None:
        if threshold is not None:
            self.threshold = max(0, min(255, int(threshold)))
        if minimum_percent is not None:
            self.minimum_percent = max(0.01, float(minimum_percent))
            self.minimum_pixels = max(
                1, int(self.width * self.height * self.minimum_percent / 100))

    def changed_percent(self, frame: bytes) -> float:
        rgb = self.np.frombuffer(frame, dtype=self.np.uint8).reshape(
            self.height, self.width, 3)
        gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
        gray = self.cv2.GaussianBlur(gray, (9, 9), 0)
        if self.previous is None:
            self.previous = gray
            self.boxes = []
            return 0.0
        difference = self.cv2.absdiff(self.previous, gray)
        self.previous = gray
        mask = self.cv2.threshold(difference, self.threshold, 255,
                                  self.cv2.THRESH_BINARY)[1]
        mask = self.cv2.dilate(mask, None, iterations=2)
        changed = int(self.cv2.countNonZero(mask))
        contours = self.cv2.findContours(
            mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)[0]
        minimum_box_area = max(16, self.minimum_pixels // 10)
        self.boxes = [self.cv2.boundingRect(contour) for contour in contours
                      if self.cv2.contourArea(contour) >= minimum_box_area]
        return changed * 100.0 / (self.width * self.height)

    def is_motion(self, changed_percent: float) -> bool:
        return changed_percent * self.width * self.height / 100 >= self.minimum_pixels

    def preview(self, frame: bytes, output_width: int,
                output_height: int) -> bytes:
        rgb = self.np.frombuffer(frame, dtype=self.np.uint8).reshape(
            self.height, self.width, 3)
        if self.boxes:
            rgb = rgb.copy()
        for x, y, width, height in self.boxes:
            self.cv2.rectangle(rgb, (x, y), (x + width, y + height),
                               (255, 48, 48), 2)
        if (output_width, output_height) != (self.width, self.height):
            rgb = self.cv2.resize(rgb, (output_width, output_height),
                                  interpolation=self.cv2.INTER_AREA)
        return rgb.tobytes()


class MotionCoordinator:
    """Require near-simultaneous motion from multiple cameras."""

    def __init__(self, minimum_cameras: int, window_seconds: float):
        self.minimum_cameras = minimum_cameras
        self.window_seconds = window_seconds
        self.last_motion: dict[int, float] = {}
        self.lock = threading.Lock()

    def report(self, camera_number: int, motion: bool,
               now: float) -> tuple[bool, int]:
        with self.lock:
            if motion:
                self.last_motion[camera_number] = now
            cutoff = now - self.window_seconds
            self.last_motion = {
                number: timestamp for number, timestamp in self.last_motion.items()
                if timestamp >= cutoff
            }
            count = len(self.last_motion)
            return count >= self.minimum_cameras, count


class EventRecorder:
    """Retain rolling H.264 segments around motion and join them losslessly."""

    def __init__(self, root: Path, buffer_root: Path, camera_number: int,
                 pre_seconds: float, post_seconds: float, extension: str = "ts"):
        self.extension = extension
        self.root = root
        self.camera_number = camera_number
        self.pre_seconds, self.post_seconds = pre_seconds, post_seconds
        self.rolling = buffer_root / f"camera-{camera_number:02d}"
        self.rolling.mkdir(parents=True, exist_ok=True)
        for stale in self.rolling.glob(f"*.{self.extension}"):
            stale.unlink(missing_ok=True)
        self.event_dir: Path | None = None
        self.event_segments: list[Path] = []
        self.finalizers: list[threading.Thread] = []
        self.last_motion = 0.0
        self.enabled = True

    @property
    def segment_pattern(self) -> str:
        return str(self.rolling / f"%Y%m%d-%H%M%S.{self.extension}")

    @property
    def recording(self) -> bool:
        return self.event_dir is not None

    def update(self, motion: bool, now: float) -> None:
        if not self.enabled:
            if self.event_dir is not None:
                self._collect_completed(now)
                self._finish()
            self._trim(now)
            return
        if motion:
            self.last_motion = now
            if self.event_dir is None:
                stamp = datetime.fromtimestamp(now).strftime("%Y%m%d-%H%M%S-%f")[:-3]
                self.event_dir = self.root / f"camera-{self.camera_number:02d}" / stamp
                self.event_dir.mkdir(parents=True, exist_ok=True)
        self._collect_completed(now)
        if self.event_dir is not None and now - self.last_motion >= self.post_seconds:
            self._finish()
        elif self.event_dir is None:
            self._trim(now)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def _completed(self, now: float) -> list[Path]:
        # Leave FFmpeg's newest segment alone while it may still be open.
        return sorted(path for path in self.rolling.glob(f"*.{self.extension}")
                      if path.stat().st_size > 0
                      and now - path.stat().st_mtime >= 1.5)

    def _collect_completed(self, now: float) -> None:
        if self.event_dir is None:
            return
        cutoff = now - self.pre_seconds if not self.event_segments else 0
        for source in self._completed(now):
            if source.stat().st_mtime < cutoff:
                source.unlink(missing_ok=True)
                continue
            target = self.event_dir / source.name
            shutil.move(source, target)
            self.event_segments.append(target)

    def _trim(self, now: float) -> None:
        cutoff = now - self.pre_seconds - 2
        for path in self._completed(now):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)

    def _finish(self) -> None:
        event_dir, segments = self.event_dir, self.event_segments
        self.event_dir, self.event_segments = None, []
        if event_dir and segments:
            thread = threading.Thread(target=self._join, args=(event_dir, segments),
                                      daemon=False)
            self.finalizers.append(thread)
            thread.start()

    def close(self) -> None:
        if self.event_dir is not None:
            # FFmpeg has stopped, so its last segment is closed and safe to keep.
            self._collect_completed(time.time() + 2)
            self._finish()
        for thread in self.finalizers:
            thread.join(timeout=10)

    def _join(self, event_dir: Path, segments: list[Path]) -> None:
        output = event_dir.with_suffix(f".{self.extension}")
        partial = output.with_suffix(f".partial.{self.extension}")
        manifest = event_dir / "segments.txt"
        ordered = sorted(segments)
        entries = [f"file '{segment.name}'\n" for segment in ordered]
        if self.extension == "mkv":
            # Spatial captures retain absolute kernel-derived timestamps. Preserve
            # inter-segment timing and expose the shared epoch for multi-view sync.
            starts = []
            for segment in ordered:
                probe = subprocess.run([
                    "nice", "-n", "10", "ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
                    "-read_intervals", "%+#1", "-show_entries", "packet=pts_time",
                    "-of", "json", str(segment),
                ], capture_output=True, text=True)
                try:
                    starts.append(float(json.loads(probe.stdout)["packets"][0]["pts_time"]))
                except (ValueError, KeyError, IndexError):
                    return  # Keep source segments if their timeline cannot be read.
            for i in range(len(entries) - 1):
                entries[i] += f"duration {starts[i + 1] - starts[i]:.6f}\n"
            output.with_suffix(".timestamps.json").write_text(json.dumps({
                "camera_number": self.camera_number,
                "timestamp_source": "V4L2 kernel timestamps converted to host wall clock",
                "first_frame_unix_seconds": starts[0],
                "mapping": "frame Unix time = first_frame_unix_seconds + video packet PTS seconds",
                "precision_seconds": 0.001,
                "hardware_synchronized": False,
                "uvc_timing_manifest": getattr(self, "timing_manifest", None),
            }, indent=2))
        manifest.write_text("".join(entries))
        result = subprocess.run([
            "nice", "-n", "10", "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", "1", "-f", "concat", "-safe", "0", "-i", str(manifest),
            "-map", "0:v:0", "-c:v", "copy", str(partial),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode == 0 and partial.stat().st_size > 0:
            partial.replace(output)
        else:
            partial.unlink(missing_ok=True)
            if self.extension != "ts":
                # Matroska files cannot be recovered by byte concatenation.
                # Preserve the original segments and manifest for recovery.
                return
            # Keep an event recoverable even if remuxing an interrupted segment fails.
            with output.open("wb") as destination:
                for segment in ordered:
                    with segment.open("rb") as source:
                        shutil.copyfileobj(source, destination)
        manifest.unlink(missing_ok=True)
        for segment in segments:
            segment.unlink(missing_ok=True)
        try:
            event_dir.rmdir()
        except OSError:
            pass


class Stream:
    def __init__(self, camera: Camera, width: int, height: int, fps: int,
                 analysis_width: int, analysis_height: int,
                 capture_size: str, preview_fps: int,
                 startup_delay: float = 0, motion: bool = False,
                 motion_fps: float = 5, motion_threshold: int = 20,
                 motion_percent: float = 3.0, event_dir: Path | None = None,
                 buffer_dir: Path | None = None,
                 camera_number: int = 1, pre_seconds: float = 5,
                 post_seconds: float = 10,
                 coordinator: MotionCoordinator | None = None):
        self.camera, self.width, self.height, self.fps = camera, width, height, fps
        self.analysis_width, self.analysis_height = analysis_width, analysis_height
        self.preview_fps = preview_fps
        self.requested_capture_size = capture_size
        self.capture_modes = (capture_size,)
        self.capture_mode_index = 0
        self.capture_size = self.capture_modes[0]
        self.startup_delay = startup_delay
        self.rotated = False
        self.motion_fps = motion_fps
        self.motion_detector = (MotionDetector(analysis_width, analysis_height,
                                                motion_threshold, motion_percent)
                                if motion else None)
        self.recorder = (EventRecorder(event_dir, buffer_dir, camera_number,
                                       pre_seconds, post_seconds)
                         if motion and event_dir and buffer_dir else None)
        self.coordinator = coordinator
        self.camera_number = camera_number
        self.motion_camera_count = 0
        self.changed_percent = 0.0
        self._last_motion_check = 0.0
        self.measured_fps = 0.0
        self._frame_count = 0
        self._fps_started = time.monotonic()
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.status: queue.Queue[str] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.process: subprocess.Popen[bytes] | None = None
        self.thread = threading.Thread(target=self._supervise, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _set_status(self, message: str) -> None:
        try:
            self.status.get_nowait()
        except queue.Empty:
            pass
        self.status.put_nowait(message)

    def _supervise(self) -> None:
        frame_size = self.analysis_width * self.analysis_height * 3
        if self.startup_delay and self.stop_event.wait(self.startup_delay):
            return
        while not self.stop_event.is_set():
            self.capture_size = self.capture_modes[self.capture_mode_index]
            self._set_status(f"connecting at {self.capture_size}…")
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-fflags", "nobuffer", "-flags", "low_delay", "-flags2", "fast",
                # Decode the analysis branch at half resolution and omit
                # non-reference IDCT/loop-filter work.  The segment output is
                # stream-copied before these decoder options, so saved events
                # remain the camera's untouched 1080p H.264 bitstream.
                "-lowres", "1", "-skip_idct", "noref",
                "-skip_loop_filter", "all", "-threads", "1",
                "-f", "v4l2", "-input_format", ("h264" if self.camera.h264 else "mjpeg"), "-framerate", str(self.fps),
                "-video_size", self.capture_size,
                "-i", self.camera.device, "-an",
            ]
            if self.recorder:
                command.extend([
                    "-map", "0:v:0", "-c:v", "copy", "-f", "segment",
                    "-segment_time", "1", "-segment_atclocktime", "1",
                    "-reset_timestamps", "1", "-strftime", "1",
                    self.recorder.segment_pattern,
                ])
            command.extend([
                "-map", "0:v:0", "-vf",
                video_filter(self.analysis_width, self.analysis_height,
                             self.rotated, self.preview_fps),
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
            ])
            try:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    bufsize=frame_size,
                )
                self.process = process
                assert process.stdout is not None
                while not self.stop_event.is_set():
                    frame = process.stdout.read(frame_size)
                    if len(frame) != frame_size:
                        break
                    now = time.monotonic()
                    if (self.motion_detector and self.recorder
                            and now - self._last_motion_check >= 1 / self.motion_fps):
                        self._last_motion_check = now
                        self.changed_percent = self.motion_detector.changed_percent(frame)
                        wall_time = time.time()
                        local_motion = self.motion_detector.is_motion(
                            self.changed_percent)
                        if self.coordinator:
                            consensus, self.motion_camera_count = self.coordinator.report(
                                self.camera_number, local_motion, wall_time)
                        else:
                            consensus, self.motion_camera_count = local_motion, int(local_motion)
                        self.recorder.update(consensus, wall_time)
                    if self.motion_detector:
                        frame = self.motion_detector.preview(
                            frame, self.width, self.height)
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    self.frames.put_nowait(frame)
                    self._frame_count += 1
                    elapsed = now - self._fps_started
                    if elapsed >= 1:
                        self.measured_fps = self._frame_count / elapsed
                        self._frame_count = 0
                        self._fps_started = now
                    self._set_status("")
                if not self.stop_event.is_set() and process.stderr is not None:
                    error = process.stderr.read().decode(errors="replace").strip()
                    if error:
                        last_error = error.splitlines()[-1]
                        self._set_status(last_error)
                        print(f"{self.camera.device}: {last_error}",
                              file=sys.stderr, flush=True)
            except OSError as exc:
                self._set_status(str(exc))
            finally:
                self._terminate_process()
            if not self.stop_event.wait(2):
                self._set_status(f"reconnecting at {self.capture_modes[self.capture_mode_index]}…")

    def _terminate_process(self) -> None:
        process, self.process = self.process, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self) -> None:
        self.stop_event.set()
        self._terminate_process()
        if self.recorder:
            self.recorder.close()

    def toggle_rotation(self) -> None:
        self.rotated = not self.rotated
        self.measured_fps = 0.0
        self._frame_count = 0
        self._fps_started = time.monotonic()
        self._set_status("rotating 180°…")
        self._terminate_process()

    def configure_motion(self, threshold: int | None = None,
                         minimum_percent: float | None = None,
                         motion_fps: float | None = None) -> None:
        if self.motion_detector:
            self.motion_detector.configure(threshold, minimum_percent)
        if motion_fps is not None:
            self.motion_fps = max(0.1, float(motion_fps))

class Viewer:
    def __init__(self, root: tk.Tk, cameras: list[Camera], args: argparse.Namespace):
        self.root, self.streams, self.images = root, [], []
        self.closed = False
        self.recording_enabled = True
        self.coordinator = (MotionCoordinator(args.min_motion_cameras,
                                              args.motion_window)
                            if args.motion else None)
        root.title("USB Camera Grid — H.264")
        root.configure(bg="#111")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<KeyPress-q>", lambda _event: self.close())
        root.bind("<KeyPress-Q>", lambda _event: self.close())
        root.bind("<Escape>", lambda _event: self.close())
        for index, camera in enumerate(cameras):
            row, column = divmod(index, args.columns)
            panel = tk.Frame(root, bg="#181818", highlightbackground="#333", highlightthickness=1)
            panel.grid(row=row, column=column, padx=2, pady=2, sticky="nsew")
            label = tk.Label(panel, text="connecting…", fg="#aaa", bg="#181818")
            label.pack(fill="both", expand=True)
            overlay = tk.Label(label, text=f"{args.capture_size}  •  -- fps",
                               fg="white", bg="#202020", padx=5, pady=2,
                               font=("TkDefaultFont", 9, "bold"))
            overlay.place(x=6, y=6)
            caption = tk.Label(panel, text=f"{index + 1}. {camera.name}  •  {camera.device}",
                               fg="#ddd", bg="#222", anchor="w", padx=6, width=1)
            caption.pack(fill="x")
            stream = Stream(camera, args.width, args.height, args.fps,
                            args.analysis_width, args.analysis_height,
                            args.capture_size, args.preview_fps,
                            index * args.startup_interval, args.motion,
                            args.motion_fps, args.motion_threshold,
                            args.motion_percent, Path(args.event_dir),
                            Path(args.buffer_dir), index + 1,
                            args.pre_seconds, args.post_seconds, self.coordinator)
            stream.label = label  # type: ignore[attr-defined]
            stream.overlay = overlay  # type: ignore[attr-defined]
            rotate = tk.Button(label, text="↻", command=stream.toggle_rotation,
                               fg="white", bg="#202020", activeforeground="white",
                               activebackground="#444", relief="flat", bd=0,
                               padx=7, pady=2, font=("TkDefaultFont", 12, "bold"),
                               cursor="hand2")
            rotate.place(relx=1.0, x=-6, y=6, anchor="ne")
            stream.rotate_button = rotate  # type: ignore[attr-defined]
            self.streams.append(stream)
            self.images.append(None)
            stream.start()
        self.recording_button = tk.Button(
            root, text="Recording: ON", command=self.toggle_recording,
            fg="white", bg="#8b1e1e", activeforeground="white",
            activebackground="#a52a2a", relief="raised", bd=2,
            padx=12, pady=7, font=("TkDefaultFont", 11, "bold"),
            cursor="hand2")
        self.recording_button.place(relx=1.0, x=-12, y=12, anchor="ne")
        self.motion_controls = tk.Frame(root, bg="#202020", bd=2, relief="raised")
        self.motion_controls.place(relx=0.5, y=8, anchor="n")
        self.threshold_value = tk.IntVar(value=args.motion_threshold)
        self.percent_value = tk.DoubleVar(value=args.motion_percent)
        self.motion_fps_value = tk.DoubleVar(value=args.motion_fps)
        self._add_motion_scale("Pixel delta", self.threshold_value, 1, 100, 1,
                               self.update_motion_controls, 0)
        self._add_motion_scale("Trigger area %", self.percent_value, 0.1, 20, 0.1,
                               self.update_motion_controls, 1)
        self._add_motion_scale("Analysis FPS", self.motion_fps_value, 1, 15, 1,
                               self.update_motion_controls, 2)
        rows = max(1, (len(cameras) + args.columns - 1) // args.columns)
        for column in range(args.columns):
            root.grid_columnconfigure(column, weight=1)
        for row in range(rows):
            root.grid_rowconfigure(row, weight=1)
        # Fill the usable desktop immediately; avoids initially hiding the
        # second row on screens smaller than the grid's requested dimensions.
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        try:
            root.attributes("-zoomed", True)
        except tk.TclError:
            pass
        root.after(30, self.refresh)

    def refresh(self) -> None:
        for index, stream in enumerate(self.streams):
            try:
                frame = stream.frames.get_nowait()
            except queue.Empty:
                frame = None
            if frame:
                ppm = f"P6 {stream.width} {stream.height} 255\n".encode() + frame
                image = tk.PhotoImage(data=ppm, format="PPM")
                self.images[index] = image
                stream.label.configure(image=image, text="")  # type: ignore[attr-defined]
                stream.overlay.lift()  # type: ignore[attr-defined]
                stream.rotate_button.lift()  # type: ignore[attr-defined]
                self.recording_button.lift()
                self.motion_controls.lift()
                stream.overlay.configure(  # type: ignore[attr-defined]
                    text=stream_overlay_text(stream)
                )
            elif self.images[index] is None:
                try:
                    message = stream.status.get_nowait()
                    if message:
                        stream.label.configure(text=message)  # type: ignore[attr-defined]
                except queue.Empty:
                    pass
        self.root.after(30, self.refresh)

    def _add_motion_scale(self, label: str, variable: tk.Variable,
                          minimum: float, maximum: float, resolution: float,
                          command: Any, column: int) -> None:
        frame = tk.Frame(self.motion_controls, bg="#202020")
        frame.grid(row=0, column=column, padx=6, pady=3)
        tk.Label(frame, text=label, fg="white", bg="#202020").pack()
        tk.Scale(frame, from_=minimum, to=maximum, resolution=resolution,
                 orient="horizontal", length=190, variable=variable,
                 command=command, showvalue=True, fg="white", bg="#202020",
                 troughcolor="#555", highlightthickness=0).pack()

    def update_motion_controls(self, _value: str = "") -> None:
        threshold = self.threshold_value.get()
        minimum_percent = self.percent_value.get()
        motion_fps = self.motion_fps_value.get()
        for stream in self.streams:
            stream.configure_motion(threshold, minimum_percent, motion_fps)

    def toggle_recording(self) -> None:
        self.recording_enabled = not self.recording_enabled
        for stream in self.streams:
            if stream.recorder:
                stream.recorder.set_enabled(self.recording_enabled)
        if self.recording_enabled:
            self.recording_button.configure(text="Recording: ON", bg="#8b1e1e")
        else:
            self.recording_button.configure(text="Recording: OFF", bg="#444")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for stream in self.streams:
            stream.close()
        self.root.destroy()


def print_inventory(supported: list[Camera], rejected: list[Camera]) -> None:
    print(f"H.264 cameras: {len(supported)}")
    for index, camera in enumerate(supported, 1):
        print(f"  {index:2}. {camera.device:<12} {camera.name} [{camera.bus}]")
    groups: dict[str, int] = {}
    root_ports: dict[str, int] = {}
    for camera in supported:
        controller = controller_name(camera.bus)
        groups[controller] = groups.get(controller, 0) + 1
        root_port = root_port_name(camera.bus)
        root_ports[root_port] = root_ports.get(root_port, 0) + 1
    if groups:
        print("USB controller distribution: " +
              ", ".join(f"{name}: {count}" for name, count in groups.items()))
        print("USB root-port distribution: " +
              ", ".join(f"{name}: {count}" for name, count in root_ports.items()))
        if len(supported) >= 8 and max(root_ports.values()) > 3:
            print("Warning: more than three cameras share a USB 2 root port.",
                  file=sys.stderr)
    if rejected:
        print(f"Capture endpoints without H.264: {len(rejected)}")
        for camera in rejected:
            print(f"      {camera.device:<12} {camera.name} ({', '.join(camera.formats) or 'no formats'})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="probe devices and exit")
    parser.add_argument("--devices", nargs="*", help="explicit /dev/video devices (skip discovery)")
    parser.add_argument("--input-format", choices=("h264", "mjpeg"), default="h264",
                        help="capture format for explicitly supplied devices")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--width", type=int, default=352, help="decoded tile width")
    parser.add_argument("--height", type=int, default=198, help="decoded tile height")
    parser.add_argument("--analysis-width", type=int, default=640,
                        help="OpenCV analysis width (default: 640)")
    parser.add_argument("--analysis-height", type=int, default=360,
                        help="OpenCV analysis height (default: 360)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--capture-size",
                        help="H.264 camera mode (default: 1080p with --motion, otherwise 640x360)")
    parser.add_argument("--preview-fps", type=int, default=5,
                        help="grid/analysis decode rate, up to --fps (default: 5)")
    parser.add_argument("--startup-interval", type=float, default=0.4,
                        help="seconds between camera starts (default: 0.4)")
    parser.add_argument("--motion", action="store_true",
                        help="detect motion and retain full-resolution H.264 events")
    parser.add_argument("--motion-fps", type=float, default=5,
                        help="motion checks per second (default: 5)")
    parser.add_argument("--motion-threshold", type=int, default=20,
                        help="per-pixel grayscale difference (default: 20)")
    parser.add_argument("--motion-percent", type=float, default=3.0,
                        help="changed frame area that triggers motion (default: 3.0)")
    parser.add_argument("--event-dir", default="/mnt/ssd/camera-events",
                        help="event recording directory")
    parser.add_argument("--buffer-dir", default="/dev/shm/camera-grid",
                        help="RAM-backed rolling-buffer directory")
    parser.add_argument("--min-motion-cameras", type=int, default=3,
                        help="cameras required to trigger recording (default: 3)")
    parser.add_argument("--motion-window", type=float, default=1.0,
                        help="seconds allowed between camera triggers (default: 1.0)")
    parser.add_argument("--pre-seconds", type=float, default=5,
                        help="seconds retained before motion (default: 5)")
    parser.add_argument("--post-seconds", type=float, default=10,
                        help="quiet seconds retained after motion (default: 10)")
    args = parser.parse_args()
    if args.capture_size is None:
        args.capture_size = "1920x1080" if args.motion else "640x360"
    if (args.columns < 1 or args.width < 1 or args.height < 1
            or args.analysis_width < 1 or args.analysis_height < 1 or args.fps < 1
            or args.preview_fps < 1 or args.preview_fps > args.fps
            or args.startup_interval < 0 or args.motion_fps <= 0
            or not 0 <= args.motion_threshold <= 255 or args.motion_percent <= 0
            or args.pre_seconds < 0 or args.post_seconds < 0
            or args.min_motion_cameras < 1 or args.motion_window <= 0):
        parser.error("columns, dimensions, and fps must be positive")
    if args.devices is not None:
        fourcc = "H264" if args.input_format == "h264" else "MJPG"
        cameras = [Camera(d, os.path.basename(d), "explicit", (fourcc,)) for d in args.devices]
        rejected: list[Camera] = []
    else:
        cameras, rejected = discover()
    print_inventory(cameras, rejected)
    if args.list:
        return 0 if cameras else 1
    if not cameras:
        print("No H.264 V4L2 cameras found. Try --list or pass --devices.", file=sys.stderr)
        return 1
    if len(cameras) != 10:
        print(f"Warning: expected 10 cameras, found {len(cameras)}.", file=sys.stderr)
    root = tk.Tk()
    viewer = Viewer(root, cameras[:10], args)
    signal.signal(signal.SIGINT, lambda *_args: root.after(0, viewer.close))
    try:
        root.mainloop()
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
