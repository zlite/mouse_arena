"""Capture original UVCH metadata without opening the video endpoint.

Each block: little-endian uint64 host monotonic nanoseconds, uint16 host USB
SOF, uint8 header length, uint8 flags, then optional uint32 PTS and SCR
(uint32 device clock, uint16 device SOF). Header length excludes the first
10 host bytes. Raw counters are intentionally not converted using descriptors.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import struct
import subprocess
import threading
import time


def blocks(data: bytes):
    offset = 0
    while offset + 12 <= len(data):
        host, sof, length, flags = struct.unpack_from('<QHBB', data, offset)
        if length < 2 or offset + 10 + length > len(data):
            raise ValueError('Incomplete or invalid UVCH block')
        cursor = offset + 12
        end = offset + 10 + length
        row = dict(host_receive_monotonic_ns=host, host_usb_sof=sof, flags=flags)
        if flags & 4:
            if cursor + 4 > end:
                raise ValueError('Truncated PTS')
            row['device_pts_raw'] = struct.unpack_from('<I', data, cursor)[0]
            cursor += 4
        if flags & 8:
            if cursor + 6 > end:
                raise ValueError('Truncated SCR')
            row['device_scr_raw'], row['device_usb_sof'] = struct.unpack_from('<IH', data, cursor)
        yield row
        offset = end
    if offset != len(data):
        raise ValueError('Trailing incomplete UVCH block')


class TimestampSession:
    def __init__(self, cameras, root: Path):
        self.directory = root / 'timing' / (time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000000:09d}')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.processes = []
        self.threads = []
        self.errors = []
        self.sources = []
        for label, camera in cameras:
            video = Path(os.path.realpath(camera.device))
            # Match physical USB interface rather than relying on numbering.
            interface = (Path('/sys/class/video4linux') / video.name / 'device').resolve()
            candidates = []
            for node in Path('/sys/class/video4linux').glob('video*'):
                if (node / 'device').resolve() == interface and node.name != video.name:
                    info = subprocess.run(['v4l2-ctl', '-d', '/dev/' + node.name, '--get-fmt-meta'],
                                          capture_output=True, text=True).stdout
                    if 'UVCH' in info:
                        candidates.append('/dev/' + node.name)
            if len(candidates) != 1:
                raise RuntimeError(f'Cannot identify UVCH endpoint for {camera.device}')
            self.sources.append(dict(label=label, video=camera.device, bus=camera.bus,
                                     metadata_device=candidates[0], file=f'{label}.uvch.gz'))
        self.manifest = self.directory / 'session.json'
        self._manifest('starting')

    def _manifest(self, status):
        self.manifest.write_text(json.dumps(dict(
            status=status, sources=self.sources, errors=self.errors,
            format='Concatenated Linux UVCH payload-header blocks, gzip compressed',
            host_clock='CLOCK_MONOTONIC nanoseconds from UVC driver at payload reception',
            device_clock='Raw unsigned 32-bit PTS/SCR; independent clock, wraps, frequency not assumed',
            clock_mapping='clock.jsonl contains bracketed host realtime/monotonic samples',
            association='Match by time; blocks are payload headers, not one row per video frame',
            hardware_synchronized=False), indent=2))

    def start(self):
        for source in self.sources:
            process = subprocess.Popen(['v4l2-ctl', '-d', source['metadata_device'],
                                        '--stream-mmap', '--stream-to=-'],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.processes.append(process)
            thread = threading.Thread(target=self._save, args=(process, source), daemon=True)
            self.threads.append(thread)
            thread.start()
        self.clock_thread = threading.Thread(target=self._clocks, daemon=True)
        self.clock_thread.start()
        self._manifest('capturing')

    def _save(self, process, source):
        try:
            with gzip.open(self.directory / source['file'], 'wb', compresslevel=1) as output:
                pending = b""
                source['blocks_saved'] = 0
                while True:
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    pending += chunk
                    offset = 0
                    while offset + 12 <= len(pending):
                        length = pending[offset + 10]
                        if length < 2:
                            raise ValueError('Invalid UVCH header length')
                        end = offset + 10 + length
                        if end > len(pending):
                            break
                        source['blocks_saved'] += 1
                        offset = end
                    output.write(pending[:offset])
                    pending = pending[offset:]
                source['shutdown_partial_bytes_discarded'] = len(pending)
            if not self.stop.is_set():
                self.errors.append(f"{source['label']}: metadata stream stopped unexpectedly")
                self._manifest('error')
        except Exception as exc:
            self.errors.append(f"{source['label']}: {exc}")
            self._manifest('error')

    def _clocks(self):
        with (self.directory / 'clock.jsonl').open('w') as output:
            while not self.stop.is_set():
                before = time.monotonic_ns()
                realtime = time.time_ns()
                after = time.monotonic_ns()
                output.write(json.dumps(dict(monotonic_before_ns=before,
                                             realtime_ns=realtime,
                                             monotonic_after_ns=after)) + '\n')
                output.flush()
                self.stop.wait(1)

    def close(self):
        self.stop.set()
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for thread in self.threads:
            thread.join()
        self.clock_thread.join()
        self._manifest('closed' if not self.errors else 'error')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode saved UVCH timestamps into JSON lines')
    parser.add_argument('file', type=Path)
    args = parser.parse_args()
    with gzip.open(args.file, 'rb') as source:
        while True:
            prefix = source.read(12)
            if not prefix:
                break
            if len(prefix) != 12 or prefix[10] < 2:
                raise ValueError('Incomplete UVCH header')
            payload = source.read(prefix[10] - 2)
            for row in blocks(prefix + payload):
                print(json.dumps(row))
