import os
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
cv2.setNumThreads(4)

import numpy as np
import threading
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "inswapper_128.onnx")
_DET_PATH  = os.path.join(os.path.expanduser("~"), ".insightface", "models",
                           "buffalo_sc", "det_500m.onnx")

_CPU_PROVIDERS = ["CPUExecutionProvider"]

def _best_providers():
    available = ort.get_available_providers()
    for p in ["CUDAExecutionProvider", "DmlExecutionProvider"]:
        if p in available:
            return [p, "CPUExecutionProvider"]
    return _CPU_PROVIDERS

_SWAP_PROVIDERS = _best_providers()

_PROC_W, _PROC_H = 320, 240


class _Face:
    __slots__ = ("kps",)
    def __init__(self, kps: np.ndarray):
        self.kps = kps


class FaceSwapper:
    def __init__(self):
        self.source_analyzer = None
        self.fast_det        = None
        self.swapper         = None
        self.source_face     = None
        self._lock           = threading.Lock()
        self._tick           = 0
        self._cached_kps     = None
        self.ready           = False
        self.status          = "Carregando modelos..."
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            # Detector leve — CPU é suficiente
            self.status = "Carregando detector (det_500m)..."
            self.fast_det = insightface.model_zoo.get_model(
                _DET_PATH, providers=_CPU_PROVIDERS
            )
            self.fast_det.prepare(ctx_id=0, input_size=(160, 160), det_thresh=0.5)

            # buffalo_l para análise da foto fonte — CPU, roda só uma vez
            self.status = "Carregando buffalo_l..."
            self.source_analyzer = FaceAnalysis(name="buffalo_l", providers=_CPU_PROVIDERS)
            self.source_analyzer.prepare(ctx_id=0, det_size=(320, 320))

            self.status = f"Carregando inswapper ({_SWAP_PROVIDERS[0]})..."
            self.swapper = insightface.model_zoo.get_model(
                MODEL_PATH, providers=_SWAP_PROVIDERS
            )

            self.status = "Warmup GPU (aguarde ~20s)..."
            self._warmup()

            used = self.swapper.session.get_providers()
            if "CUDAExecutionProvider" in used:
                gpu = "NVIDIA GPU (CUDA)"
            elif "DmlExecutionProvider" in used:
                gpu = "Intel/AMD GPU (DirectML)"
            else:
                gpu = "CPU"
            self.ready  = True
            self.status = f"Pronto! [{gpu}] Faca upload de um rosto."
        except Exception as e:
            self.status = f"Erro: {e}"
            import traceback; traceback.print_exc()

    def _warmup(self):
        try:
            ns = self.swapper.input_names
            d0 = np.zeros((1, 3, 128, 128), dtype=np.float32)
            ld = self.swapper.emap.shape[1] if hasattr(self.swapper, "emap") else 512
            d1 = np.zeros((1, ld), dtype=np.float32)
            for _ in range(3):   # 3 warmup passes para compilar os shaders
                self.swapper.session.run(self.swapper.output_names, {ns[0]: d0, ns[1]: d1})
        except Exception as e:
            print(f"[warmup] {e}")

    # ── API pública ────────────────────────────────────────────────────────

    def set_source_face(self, image_bytes: bytes) -> tuple:
        if not self.ready:
            return False, "Modelos ainda carregando."
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "Imagem invalida."
        faces = self.source_analyzer.get(img)
        if not faces:
            return False, "Nenhum rosto detectado."
        best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        if getattr(best, "normed_embedding", None) is None:
            return False, "Nao foi possivel extrair embedding."
        with self._lock:
            self.source_face = best
            self._cached_kps = None
        return True, "Rosto carregado!"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            if not self.ready or self.source_face is None:
                return frame
            source = self.source_face

        small = cv2.resize(frame, (_PROC_W, _PROC_H))
        self._tick += 1

        if self._tick % 6 == 1 or self._cached_kps is None:
            try:
                _, kps_all = self.fast_det.detect(small, max_num=1)
                self._cached_kps = kps_all[0] if (kps_all is not None and len(kps_all)) else None
            except Exception:
                self._cached_kps = None

        if self._cached_kps is None:
            return frame

        try:
            target       = _Face(self._cached_kps)
            swapped_small = self.swapper.get(small.copy(), target, source, paste_back=True)
            return cv2.resize(swapped_small, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"[swap error] {e}")
            return frame

    def clear_source(self):
        with self._lock:
            self.source_face = None
            self._cached_kps = None

    def get_status(self) -> str:
        return self.status
