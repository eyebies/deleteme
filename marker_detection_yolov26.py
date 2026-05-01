import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


# =========================
# Config
# =========================
@dataclass
class Config:
    model_path: str
    image_paths: List[str]
    class_names: List[str]
    output_dir: str = "outputs"
    input_size: int = 640
    conf_thres: float = 0.25
    warmup_runs: int = 5


# =========================
# Runner
# =========================
class YOLOONNXBenchmark:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if os.path.exists(cfg.model_path)==False:
            raise FileNotFoundError(cfg.model_path)
    

        self.session = ort.InferenceSession(
            cfg.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        
        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255,
            size=(len(cfg.class_names), 3),
            dtype=np.uint8
        )

    # -------------------------
    # Preprocess
    # -------------------------
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (self.cfg.input_size, self.cfg.input_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, 0)

    # -------------------------
    # Inference
    # -------------------------
    def infer(self, image: np.ndarray):
        inputs = {self.input_name: self.preprocess(image)}
        return self.session.run(None, inputs)[0]

    # -------------------------
    # Postprocess
    # -------------------------
    def postprocess(self, outputs, orig_shape):
        h, w = orig_shape
        sx = w / self.cfg.input_size
        sy = h / self.cfg.input_size

        dets = []
        for b in outputs[0]:
            conf = float(b[4])
            label = int(b[5])
            if conf < self.cfg.conf_thres:
                continue

            x1, y1, x2, y2 = b[:4]
            dets.append((
                int(x1 * sx),
                int(y1 * sy),
                int(x2 * sx),
                int(y2 * sy),
                conf,
                label
            ))
        return dets

    # -------------------------
    # Draw
    # -------------------------

    def draw(self, img, dets):
        for x1, y1, x2, y2, conf, cls_id in dets:

            color = tuple(int(c) for c in self.colors[cls_id])
            label = self.cfg.class_names[cls_id] if cls_id < len(self.cfg.class_names) else str(cls_id)

            text = f"{label} {conf:.2f}"

            # ======================
            # BOX
            # ======================
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # ======================
            # LABEL SIZE
            # ======================
            (tw, th), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            # ======================
            # LABEL BACKGROUND (Darknet style)
            # ======================
            y1_label = max(y1, th + 10)

            cv2.rectangle(
                img,
                (x1, y1_label - th - 8),
                (x1 + tw + 8, y1_label + baseline - 4),
                color,
                -1
            )

            # ======================
            # LABEL TEXT (black for contrast)
            # ======================
            cv2.putText(
                img,
                text,
                (x1 + 4, y1_label - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                lineType=cv2.LINE_AA
            )

        return img

    # -------------------------
    # Warmup
    # -------------------------
    def warmup(self, image: np.ndarray):
        for _ in range(self.cfg.warmup_runs):
            _ = self.infer(image)

    # -------------------------
    # Benchmark single image
    # -------------------------
    def benchmark_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        h, w = img.shape[:2]

        self.warmup(img)

        times = []
        outputs = None

        for _ in range(25):
            start = time.perf_counter()
            outputs = self.infer(img)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        dets = self.postprocess(outputs, (h, w))
        result = self.draw(img.copy(), dets)

        return result, times, dets

    # -------------------------
    # Full benchmark
    # -------------------------
    def run(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        all_times = []
        total_dets = 0

        for idx, path in enumerate(self.cfg.image_paths):

            result, times, dets = self.benchmark_image(path)

            all_times.extend(times)
            total_dets += len(dets)

            out_path = os.path.join(self.cfg.output_dir, f"result_{idx}.jpg")
            cv2.imwrite(out_path, result)

            print(f"\nImage: {path}")
            print(f"Detections: {len(dets)}")
            print(f"Saved: {out_path}")

        self.print_stats(all_times, total_dets)

    # -------------------------
    # Stats
    # -------------------------
    def print_stats(self, times: List[float], total_dets: int):
        times = np.array(times)

        print("\n================ BENCHMARK ================")
        print(f"Total images: {len(self.cfg.image_paths)}")
        print(f"Total detections: {total_dets}")
        print(f"Avg latency (ms): {times.mean():.3f}")
        print(f"Min latency (ms): {times.min():.3f}")
        print(f"Max latency (ms): {times.max():.3f}")
        print(f"Std dev (ms): {times.std():.3f}")
        print(f"FPS: {1000 / times.mean():.2f}")


# =========================
# MAIN (no argparse)
# =========================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    cfg = Config(
         model_path=str(BASE_DIR / "ckpt" / "yolov26s.onnx"),
        image_paths=[
            str(BASE_DIR / "test_images" / "test_1.jpg"),
            str(BASE_DIR / "test_images" / "test_2.jpg")
        ],
        output_dir=str(BASE_DIR / "output"),
        input_size=640,
        conf_thres=0.4,
        warmup_runs=5,
        class_names=['R', 'L']
    )

    runner = YOLOONNXBenchmark(cfg)
    runner.run()
