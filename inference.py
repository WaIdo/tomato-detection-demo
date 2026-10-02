"""CPU inference for the two frozen, six-class thesis models."""
from __future__ import annotations

import ast
import csv
import gc
import hashlib
import io
import json
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
NAMES = ("Late_blight", "Leaf_Miner", "Magnesium_Deficiency",
         "Nitrogen_Deficiency", "Pottassium_Deficiency", "Spotted_Wilt_Virus")
CHINESE = ("晚疫病", "潜叶虫害", "缺镁", "缺氮", "缺钾", "斑萎病毒病")
COLORS = ((32, 132, 105), (40, 108, 183), (173, 108, 12),
          (162, 62, 104), (91, 97, 164), (15, 126, 142))
MAX_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 4_000_000
CSV_FIELDS = ["image", "object", "class_id", "class", "class_cn", "confidence",
              "x1", "y1", "x2", "y2", "model", "inference_ms"]
cv2.setNumThreads(1)


def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("单张文件须为非空图片，且不超过 2 MiB。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("仅支持 JPEG 和 PNG 图片。")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("图片总像素不能超过 400 万，请先缩小尺寸。")
                return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("图片分辨率过大。") from exc
    except (OSError, SyntaxError) as exc:
        raise ValueError("图片无法解码，请重新导出为 JPEG 或 PNG。") from exc


def letterbox(rgb: np.ndarray, size: int = 640):
    height, width = rgb.shape[:2]
    ratio = min(size / height, size / width)
    nw, nh = round(width * ratio), round(height * ratio)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    padded = cv2.copyMakeBorder(resized, top, round(dh + 0.1), left,
                               round(dw + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, ratio, (left, top)


def nms(boxes, scores, classes, threshold=0.7, max_det=300):
    order = np.argsort(-scores, kind="stable")[:30000]
    keep = []
    while len(order) and len(keep) < max_det:
        i, rest = int(order[0]), order[1:]
        keep.append(i)
        lo = np.maximum(boxes[i, :2], boxes[rest, :2])
        hi = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        area = np.maximum(hi - lo, 0).prod(axis=1)
        union = (np.maximum(boxes[i, 2:] - boxes[i, :2], 0).prod()
                 + np.maximum(boxes[rest, 2:] - boxes[rest, :2], 0).prod(axis=1) - area)
        overlap = area / np.maximum(union, 1e-7)
        order = rest[(classes[rest] != classes[i]) | (overlap <= threshold)]
    return np.asarray(keep, dtype=np.int64)


def postprocess(output, shape, ratio, pad, confidence=0.25, iou=0.7):
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 10:
        raise ValueError("模型输出与六类别检测配置不一致。")
    predictions = output[0].T
    scores = predictions[:, 4:].max(axis=1)
    valid = np.isfinite(predictions).all(axis=1) & (scores > confidence)
    predictions, scores = predictions[valid], scores[valid]
    classes = predictions[:, 4:].argmax(axis=1)
    boxes = np.concatenate((predictions[:, :2] - predictions[:, 2:4] / 2,
                            predictions[:, :2] + predictions[:, 2:4] / 2), axis=1)
    keep = nms(boxes, scores, classes, iou)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
    return [{"class_id": int(c), "class": NAMES[c], "class_cn": CHINESE[c],
             "confidence": float(s), "box": [float(x) for x in b]}
            for b, s, c in zip(boxes, scores, classes) if b[2] > b[0] and b[3] > b[1]]


class Detector:
    """One model and one active inference across all Streamlit sessions."""

    def __init__(self):
        self.lock = threading.Lock()
        self.session = None
        self.model = None

    def _load(self, model):
        if model not in {"gtrfskd", "p2d"}:
            raise ValueError("未知模型。")
        if self.model == model:
            return
        manifest = json.loads((ROOT / "models/manifest.json").read_text())
        path = ROOT / "models" / f"{model}.onnx"
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != manifest[model]["sha256"]:
            raise ValueError("模型校验失败，请联系系统维护人员。")
        self.session, self.model = None, None
        gc.collect()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [1, 3, 640, 640]:
            raise ValueError("模型输入尺寸不是 1×3×640×640。")
        names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])
        if names != dict(enumerate(NAMES)):
            raise ValueError("模型类别顺序不一致。")
        self.session, self.model = session, model

    def predict(self, rgb, model="gtrfskd", confidence=0.25, iou=0.7):
        if not 0.05 <= confidence <= 0.95 or not 0.1 <= iou <= 0.95:
            raise ValueError("检测阈值超出范围。")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("当前有检测任务运行，请稍后重试。")
        try:
            load_start = time.perf_counter()
            self._load(model)
            load_ms = (time.perf_counter() - load_start) * 1000
            start = time.perf_counter()
            tensor, ratio, pad = letterbox(rgb)
            prepared = time.perf_counter()
            output = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0]
            inferred = time.perf_counter()
            detections = postprocess(output, rgb.shape, ratio, pad, confidence, iou)
            end = time.perf_counter()
            return detections, {"load_ms": load_ms, "preprocess_ms": (prepared-start)*1000,
                                "inference_ms": (inferred-prepared)*1000,
                                "postprocess_ms": (end-inferred)*1000,
                                "pipeline_ms": (end-start)*1000}
        finally:
            self.lock.release()


def annotate(rgb, detections):
    canvas = rgb.copy()
    scale = max(0.42, min(canvas.shape[:2]) / 1200)
    for detection in detections:
        x1, y1, x2, y2 = map(round, detection["box"])
        color = COLORS[detection["class_id"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f'{detection["class"]} {detection["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        lx = max(0, min(x1, canvas.shape[1] - tw - 5))
        ly = max(th + 5, y1)
        cv2.rectangle(canvas, (lx, ly-th-4), (lx+tw+4, ly+base), (255, 255, 255), -1)
        cv2.putText(canvas, label, (lx+2, ly-2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def jpeg_preview(rgb):
    image = Image.fromarray(rgb)
    image.thumbnail((1280, 1280))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=93)
    return buffer.getvalue()


def safe_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def results_csv(artifacts):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for artifact in artifacts:
        for index, det in enumerate(artifact["detections"], 1):
            row = {"image": safe_cell(artifact["name"]), "object": index,
                   **{k: det[k] for k in ["class_id", "class", "class_cn"]},
                   "confidence": round(det["confidence"], 6),
                   **dict(zip(["x1", "y1", "x2", "y2"], [round(x, 3) for x in det["box"]])),
                   "model": artifact["model"], "inference_ms": round(artifact["timing"]["inference_ms"], 3)}
            writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")
