# Five-camera spatial layout

Run `python3 spatial_layout.py` for four HD USB side cameras and one Global
Shutter overhead Sentinel. Side capture uses 1920×1080 MJPEG at 30 FPS; the
Sentinel uses 1920×1200 MJPEG at 60 FPS. OpenCV analyzes 480×270 frames at the
incoming rate, while the UI redraws at 10 FPS. The status shows capture and CV
rates separately. Live testing achieved approximately 59.7 CV FPS.

Sustained motion within the calibrated overhead area triggers both long-side
cameras for every event. Each end camera records only motion in its half of the
arena; boxes crossing the midpoint select both ends. Optical FoV cones show
140° HFOV, independently of event routing. Side streams retain five
seconds of pre-trigger footage and ten seconds after motion in losslessly copied
MJPEG Matroska files under `/mnt/ssd/camera-events`. Click a side camera for a
buffered identification preview; Sentinel analysis continues. Drag cameras to
their wall positions and adjust the cyan corners to the cage floor. Initial
corners are an estimate for the new overhead view. Settings are saved separately
in `camera_layout_five.json`, preserving the previous layout.

Side thumbnails are decoded directly by the existing capture processes at 10 FPS,
using reduced-resolution JPEG decoding. Full-resolution recording is stream-copy;
preview frames never replace recorded frames. Recording disk work runs outside
the Sentinel CV/UI loop.

Each new spatial recording has a `.timestamps.json` sidecar. To align views,
add `first_frame_unix_seconds` to each packet PTS in the associated MKV. Capture
uses kernel-derived timestamps converted to the host wall clock; segment joining
preserves inter-segment offsets. Timestamp storage resolution is 1 ms, not a claim
of exposure accuracy. Cameras free-run at about 29.8 FPS: compare nearest timestamps
rather than matching frame numbers. Simultaneous sensor exposures require hardware
synchronization; no such synchronization is enabled here. Older recordings lack
this mapping. Keep each sidecar with its video.

Raw device and receive timestamps are saved for all five cameras throughout each
app session under `/mnt/ssd/camera-events/timing/<session>/`. Each `.uvch.gz` file
contains original Linux UVC payload headers: host receive monotonic nanoseconds,
host USB SOF, raw device PTS, device SCR and device USB SOF (when supplied).
`session.json` identifies cameras and endpoints and reports collection errors,
record counts and any incomplete tail discarded on shutdown. `clock.jsonl`
samples host realtime between two monotonic readings once per second, allowing
conversion to the video sidecars' wall-clock timeline without assuming a fixed
wall-clock offset. Event `.timestamps.json` files link to their timing manifest.
These are session logs, including periods when recording is OFF; keep the linked
timing directory with retained clips. Shutdown closes the compressed logs.

Decode a completed log with `python3 uvc_timestamps.py <file.uvch.gz>` to JSON
lines. Headers are per USB payload, so many rows can share one frame PTS. Raw
32-bit camera counters can wrap and must be calibrated per device; they are not
UTC values or proof of synchronized exposures. Logs preserve headers delivered
by Linux; they do not assert that the driver supplied every USB payload header.
Thumbnails redraw at 2 FPS while recording, with full-rate capture and Sentinel
motion analysis maintained independently.

The following documents the original ten-camera viewer configuration.

# Ten-camera H.264 viewer

A lightweight Linux viewer for ten USB/UVC cameras. It discovers Video4Linux
capture endpoints that advertise H.264, opens the cameras in compressed mode,
and displays them in a 5×2 grid. A failed feed reconnects automatically.

## Requirements

- Linux with the cameras visible as `/dev/video*`
- `v4l2-ctl` (`v4l-utils`), `ffmpeg`, and Python 3 with Tk
- OpenCV (`sudo apt install python3-opencv`) when using motion detection
- Permission to read the video devices (often membership in the `video` group)

## Verify and run

```bash
./camera_grid.py --list
./camera_grid.py
```

The first command should report `H.264 cameras: 10`. To bypass discovery or
enforce a specific order, list the capture endpoints explicitly:

```bash
./camera_grid.py --devices /dev/video0 /dev/video2 /dev/video4 /dev/video6 \
  /dev/video8 /dev/video10 /dev/video12 /dev/video14 /dev/video16 /dev/video18
```

Discovery automatically uses persistent `/dev/v4l/by-path/...-video-index2`
names so feeds recover into the same tile after a USB reset. The overlay reports
the actual mode and measured display frame rate. Only a tile-sized RGB frame is
transferred to the UI. Click the ↻ button on a tile to rotate that feed 180°.
The window opens maximized, and `Q` or Escape quits.

The ten-camera USB topology was tested at 1080p and 720p, but the controller
rejects excess concurrent feeds with `No space left on device` and USB protocol
errors. `--list` prints the distribution by xHCI controller. For ten 1080p30
feeds, distribute the cameras across the Pi's USB 2 root ports, with no more
than three cameras behind any one root port. A USB 3 hub does not make USB 2
cameras use its SuperSpeed bus, and several hubs daisy-chained below one root
port still share that port's 480 Mb/s schedule.

The Pi 5 exposes hardware HEVC decoding but no H.264 decoder. Ten full-resolution
software decoders cannot all render at 30 fps on its four CPU cores. Since each
grid tile is only 352×198, the low-latency default requests 640×360 H.264 and
renders at 30 fps. This preserves more detail than the tile can display while
avoiding a growing decoder backlog. Use `--capture-size 1920x1080` explicitly
for a resolution-first, lower-frame-rate view, and tune the UI independently
with `--preview-fps`.

Camera starts are staggered by 0.4 seconds so identical UVC devices do not all
negotiate bandwidth simultaneously. Change this with `--startup-interval`.

Other useful options include `--columns 5`, `--fps 30`, `--preview-fps 5`,
`--capture-size 640x360`, and `--width 352 --height 198`.

## Full-resolution motion events

Run motion mode with:

```bash
./camera_grid.py --motion --preview-fps 10
```

To calibrate the physical camera arrangement around the 70 × 40 cm arena, run:

```bash
./spatial_layout.py
```

Drag the ten numbered perimeter cameras along the cage walls. Their 50° fields
of view point inward. Clicking inside the cage tests which cameras cover that
location. The separate MJPEG camera is displayed as the overhead sentinel.
Placements persist in
`camera_layout.json` using the cameras' physical USB paths.

The cyan quadrilateral over the sentinel feed marks the detected inner cage
walls. Drag its four numbered handles if the walls move or the automatic
initial estimate needs correction. A perspective transform maps that
quadrilateral to the 70 × 40 cm diagram. Camera positions, calibration corners,
and motion-slider settings are all restored from `camera_layout.json`.
The yellow outline is the effective motion area, expanded 10% beyond the cyan
wall calibration so activity at the cage edges and corners is not discarded.
The four cameras positioned on the short ends have a 35 cm maximum coverage
depth (half the cage length); they are not selected for motion farther away.

The overhead sentinel runs OpenCV motion detection and mirrors detected boxes
onto the arena diagram. Pixel-difference, changed-area, and sustained-frame
thresholds are adjustable in the UI. All ten perimeter cameras continuously
keep untouched 1080p30 H.264 pre-trigger buffers in RAM without decoding. When
a sustained motion box lies in at least three configured FoVs, only those
cameras retain their full-resolution pre/post-trigger video on the SSD.

Motion mode captures the cameras' original 1920×1080 H.264 at 30 fps while
decoding 640×360 RGB analysis frames. Those are reduced to screen-sized tiles
after OpenCV analysis. OpenCV checks them five times per second. FFmpeg copies
the untouched, already-compressed H.264 packets into a RAM-backed rolling buffer
under `/dev/shm/camera-grid`; no Pi-side video encoding is performed. When at
least three cameras detect motion within one second, the app retains full-HD
footage from every camera, including five seconds before the trigger and ten
quiet seconds afterward. Completed recordings are losslessly remuxed to MPEG-TS
files under `/mnt/ssd/camera-events/camera-NN/`.

The overlay displays the changed-frame percentage and switches to `REC` during
an event. Useful tuning options are `--motion-fps`, `--motion-threshold`,
`--motion-percent`, `--pre-seconds`, `--post-seconds`, and `--event-dir`.
Lowering `--preview-fps` reduces display work without changing event resolution
or recorded frame rate.

The live control panel applies to every camera and exposes pixel-difference
threshold, minimum changed-frame area, and motion-analysis rate. Higher pixel
delta or trigger-area values reduce sensitivity; higher analysis FPS reacts
faster but uses more CPU. The `Recording: ON/OFF` button controls whether
detected events are retained while previews and motion analysis continue.
Consensus and buffer behavior can be tuned with `--min-motion-cameras`,
`--motion-window`, and `--buffer-dir`.
