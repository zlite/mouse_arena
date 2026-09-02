import unittest
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from camera_grid import (Camera, EventRecorder, MotionCoordinator, MotionDetector,
                         controller_name, deduplicate, device_sort_key, parse_camera,
                         root_port_name, stream_overlay_text, video_filter)
from spatial_layout import (expand_corners, map_pixel_to_arena, point_in_fov,
                            snap_to_cage)


class CameraTests(unittest.TestCase):
    def test_spatial_fov(self):
        self.assertTrue(point_in_fov(0, 20, 0, 35, 20))
        self.assertFalse(point_in_fov(0, 20, 0, 1, 40))
        self.assertTrue(point_in_fov(0, 20, 0, 35, 20, max_depth=35))
        self.assertFalse(point_in_fov(0, 20, 0, 36, 20, max_depth=35))

    def test_spatial_snap(self):
        self.assertEqual(snap_to_cage(20, 2), (20, 0.0, 90.0))
        self.assertEqual(snap_to_cage(69, 20), (70.0, 20, 180.0))

    def test_spatial_perspective_mapping(self):
        corners = [[10, 20], [110, 20], [110, 80], [10, 80]]
        x, y = map_pixel_to_arena(60, 50, corners)
        self.assertAlmostEqual(x, 35.0, places=3)
        self.assertAlmostEqual(y, 20.0, places=3)

    def test_spatial_calibration_margin(self):
        expanded = expand_corners([[0, 0], [10, 0], [10, 10], [0, 10]], 1.1)
        self.assertEqual(expanded, [[-0.5, -0.5], [10.5, -0.5],
                                    [10.5, 10.5], [-0.5, 10.5]])

    def test_parse_h264_camera(self):
        details = "Card type : USB Camera\nBus info : usb-0000:00:14.0-2"
        formats = "Type: Video Capture\n[0]: 'H264' (H.264)\n[1]: 'MJPG' (Motion-JPEG)"
        camera = parse_camera("/dev/video2", details, formats)
        self.assertEqual(camera.name, "USB Camera")
        self.assertEqual(camera.bus, "usb-0000:00:14.0-2")
        self.assertEqual(camera.formats, ("H264", "MJPG"))
        self.assertTrue(camera.h264)

    def test_natural_device_order(self):
        paths = ["/dev/video10", "/dev/video2", "/dev/video0"]
        self.assertEqual(sorted(paths, key=device_sort_key),
                         ["/dev/video0", "/dev/video2", "/dev/video10"])

    def test_deduplicate_physical_bus(self):
        cameras = [Camera("/dev/video0", "A", "usb-1", ("H264",)),
                   Camera("/dev/video1", "A", "usb-1", ("H264",)),
                   Camera("/dev/video2", "B", "usb-2", ("H264",))]
        self.assertEqual([c.device for c in deduplicate(cameras)],
                         ["/dev/video0", "/dev/video2"])

    def test_rotation_filter(self):
        normal = video_filter(352, 198, False)
        rotated = video_filter(352, 198, True)
        self.assertNotIn("hflip", normal)
        self.assertTrue(rotated.startswith("hflip,vflip,"))
        self.assertIn("scale=352:198", rotated)

    def test_preview_filter_runs_before_scaling(self):
        filters = video_filter(352, 198, False, 10)
        self.assertTrue(filters.startswith("fps=10,scale=352:198"))

    def test_controller_name(self):
        self.assertEqual(controller_name("usb-xhci-hcd.1-2.1.4"), "xhci-hcd.1")
        self.assertEqual(controller_name("explicit"), "unknown")

    def test_root_port_name(self):
        self.assertEqual(root_port_name("usb-xhci-hcd.1-2.1.4"), "xhci-hcd.1-2")
        self.assertEqual(root_port_name("explicit"), "unknown")

    def test_motion_overlay(self):
        stream = SimpleNamespace(capture_size="1920x1080", measured_fps=10.0,
                                 motion_detector=object(), recorder=None,
                                 changed_percent=2.5, rotated=True,
                                 coordinator=None)
        self.assertEqual(stream_overlay_text(stream),
                         "1920x1080  •  10.0 fps  •  2.5%  •  180°")

    def test_recording_disabled_overlay(self):
        recorder = SimpleNamespace(enabled=False, recording=False)
        stream = SimpleNamespace(capture_size="1920x1080", measured_fps=10.0,
                                 motion_detector=object(), recorder=recorder,
                                 changed_percent=9.0, rotated=False,
                                 coordinator=None)
        self.assertEqual(stream_overlay_text(stream),
                         "1920x1080  •  10.0 fps  •  REC OFF")

    def test_motion_parameters_update_and_clamp(self):
        detector = MotionDetector.__new__(MotionDetector)
        detector.width, detector.height = 100, 50
        detector.threshold, detector.minimum_percent = 0, 0.0
        detector.minimum_pixels = 1
        detector.configure(300, 2.5)
        self.assertEqual(detector.threshold, 255)
        self.assertEqual(detector.minimum_percent, 2.5)
        self.assertEqual(detector.minimum_pixels, 125)

    def test_motion_detector_draws_preview_boxes(self):
        import numpy as np

        detector = MotionDetector(100, 50, 20, 1.0)
        first = np.zeros((50, 100, 3), dtype=np.uint8)
        second = first.copy()
        second[10:30, 20:50] = 255
        detector.changed_percent(first.tobytes())
        changed = detector.changed_percent(second.tobytes())
        annotated = np.frombuffer(
            detector.preview(second.tobytes(), 100, 50), dtype=np.uint8)
        self.assertGreater(changed, 1.0)
        self.assertTrue(detector.boxes)
        self.assertTrue(np.any(annotated.reshape(50, 100, 3)[:, :, 0] == 255))

    def test_event_recorder_keeps_pretrigger_and_joins_segments(self):
        with tempfile.TemporaryDirectory() as directory, \
                tempfile.TemporaryDirectory() as buffer_directory:
            recorder = EventRecorder(Path(directory), Path(buffer_directory),
                                     1, 5, 10)
            now = time.time()
            for index, age in enumerate((4, 3, 2)):
                segment = recorder.rolling / f"segment-{index}.ts"
                segment.write_bytes(bytes([index + 1]))
                os.utime(segment, (now - age, now - age))
            recorder.update(True, now)
            recorder._finish()
            deadline = time.time() + 2
            output = None
            while time.time() < deadline:
                outputs = list(Path(directory).glob("camera-01/*.ts"))
                if outputs and outputs[0].stat().st_size == 3:
                    output = outputs[0]
                    break
                time.sleep(0.01)
            self.assertIsNotNone(output)
            self.assertEqual(output.read_bytes(), b"\x01\x02\x03")

    def test_three_camera_motion_consensus(self):
        coordinator = MotionCoordinator(3, 1.0)
        self.assertEqual(coordinator.report(1, True, 10.0), (False, 1))
        self.assertEqual(coordinator.report(2, True, 10.2), (False, 2))
        self.assertEqual(coordinator.report(3, True, 10.4), (True, 3))
        self.assertEqual(coordinator.report(3, False, 11.5), (False, 0))


if __name__ == "__main__":
    unittest.main()
