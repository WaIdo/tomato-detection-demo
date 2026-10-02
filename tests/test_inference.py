import io
import unittest

import numpy as np
from PIL import Image

from inference import MAX_BYTES, decode_image, letterbox, nms, postprocess, results_csv, safe_cell


class ImageTests(unittest.TestCase):
    def test_bad_image(self):
        with self.assertRaises(ValueError):
            decode_image(b"not an image")

    def test_size_cap(self):
        with self.assertRaises(ValueError):
            decode_image(b"x" * (MAX_BYTES + 1))

    def test_pixel_cap(self):
        buffer = io.BytesIO()
        Image.new("RGB", (2001, 2000)).save(buffer, "PNG")
        with self.assertRaises(ValueError):
            decode_image(buffer.getvalue())

    def test_exif_orientation(self):
        image = Image.new("RGB", (20, 10), "red")
        exif = image.getexif()
        exif[274] = 6
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", exif=exif)
        self.assertEqual(decode_image(buffer.getvalue()).shape, (20, 10, 3))

    def test_grayscale(self):
        buffer = io.BytesIO()
        Image.new("L", (10, 20)).save(buffer, "PNG")
        self.assertEqual(decode_image(buffer.getvalue()).shape, (20, 10, 3))

    def test_rgb_and_letterbox(self):
        image = np.full((320, 640, 3), (255, 0, 0), dtype=np.uint8)
        tensor, ratio, pad = letterbox(image)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(pad, (0, 160))
        self.assertEqual(ratio, 1)
        np.testing.assert_array_equal(tensor[0, :, 320, 320], [1, 0, 0])

    def test_coordinates_restore(self):
        output = np.zeros((1, 10, 1), np.float32)
        output[0, :4, 0] = [320, 320, 200, 100]
        output[0, 5, 0] = 0.9
        found = postprocess(output, (320, 640, 3), 1, (0, 160))
        self.assertEqual(found[0]["class_id"], 1)
        np.testing.assert_allclose(found[0]["box"], [220, 110, 420, 210])

    def test_non_square_roundtrip(self):
        for height, width in [(401, 967), (1000, 333), (480, 640)]:
            _, ratio, pad = letterbox(np.zeros((height, width, 3), np.uint8))
            box = np.array([12, 23, width-20, height-17], np.float32)
            scaled = box * ratio + np.array([*pad, *pad])
            output = np.zeros((1, 10, 1), np.float32)
            output[0, :4, 0] = [*(scaled[:2]+scaled[2:])/2, *(scaled[2:]-scaled[:2])]
            output[0, 4, 0] = 0.8
            found = postprocess(output, (height, width, 3), ratio, pad)
            np.testing.assert_allclose(found[0]["box"], box, atol=0.0002)

    def test_class_aware_nms(self):
        boxes = np.array([[0, 0, 10, 10]] * 3, np.float32)
        kept = nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]))
        self.assertEqual(kept.tolist(), [0, 2])

    def test_empty(self):
        self.assertEqual(postprocess(np.zeros((1, 10, 1)), (640, 640, 3), 1, (0, 0)), [])
        self.assertIn("confidence", results_csv([]).decode("utf-8-sig"))

    def test_nan_output(self):
        self.assertEqual(postprocess(np.full((1, 10, 1), np.nan), (640, 640, 3), 1, (0, 0)), [])

    def test_csv_escape(self):
        for name in ["=1+1.jpg", "+abc.jpg", "@test.png", "  =bad.jpg"]:
            self.assertTrue(safe_cell(name).startswith("'"))
        self.assertEqual(safe_cell("leaf.jpg"), "leaf.jpg")

    def test_wrong_shape(self):
        with self.assertRaises(ValueError):
            postprocess(np.zeros((1, 84, 8400)), (640, 640, 3), 1, (0, 0))


if __name__ == "__main__":
    unittest.main()
