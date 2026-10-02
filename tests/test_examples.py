import hashlib
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from inference import decode_image


ROOT = Path(__file__).resolve().parents[1]


class ExampleTests(unittest.TestCase):
    def test_example_integrity(self):
        expected = {
            "IMG20220323084648_aug3.jpg": "0c23864181ce77a858d4de9394ebda1eb1ea0e5edff6bfc1a395e4048c589798",
            "IMG20220324093257_aug5.jpg": "bd25f1afa6a6f71d4cceaf00fa0764225c341689b78238efd80a27ca936a9bac",
            "IMG20220325085806_aug1.jpg": "2f8d3c45152ca6cfae5fd54268edd3676c4348c9e9ca80aa1b9a2fbf199730b6",
        }
        for filename, digest in expected.items():
            data = (ROOT / "examples" / filename).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
            self.assertEqual(decode_image(data).shape, (640, 640, 3))

    def test_real_examples_threshold_and_clear(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        for index, count in [(1, 14), (2, 12), (3, 8)]:
            app.button(key=f"example-{index}").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.metric[1].value, str(count))
        app.slider[0].set_value(0.95).run()
        app.button(key="example-2").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[1].value, "0")
        self.assertEqual(app.session_state["artifacts"][0]["confidence"], 0.95)
        next(b for b in app.button if b.label == "清空结果").click().run()
        self.assertFalse(app.metric)
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == "检测上传图片").click().run()
        self.assertEqual(app.warning[0].value, "请先选择图片。")


if __name__ == "__main__":
    unittest.main()
