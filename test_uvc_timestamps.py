import struct
import unittest
from uvc_timestamps import blocks


class MetadataTests(unittest.TestCase):
    def test_host_and_device_clocks_remain_distinct(self):
        raw = struct.pack('<QHBBIIH', 1234567890123, 17, 12, 12,
                          0xffffffff, 42, 16)
        row, = blocks(raw)
        self.assertEqual(row['host_receive_monotonic_ns'], 1234567890123)
        self.assertEqual(row['device_pts_raw'], 0xffffffff)
        self.assertEqual(row['device_scr_raw'], 42)
        self.assertEqual(row['device_usb_sof'], 16)

    def test_optional_fields_and_wrap_are_preserved(self):
        raw = struct.pack('<QHBBI', 100, 4, 6, 4, 0xffffffff)
        raw += struct.pack('<QHBBI', 200, 5, 6, 4, 0)
        rows = list(blocks(raw))
        self.assertEqual([r['device_pts_raw'] for r in rows], [0xffffffff, 0])
        self.assertNotIn('device_scr_raw', rows[0])

    def test_truncation_is_reported(self):
        with self.assertRaises(ValueError):
            list(blocks(struct.pack('<QHBB', 100, 4, 12, 12)))


if __name__ == '__main__':
    unittest.main()
