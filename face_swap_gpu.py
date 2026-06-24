"""
Versao GPU do FaceSwapper — usada no Google Colab com CUDA.
Processa em resolucao completa, roda swap em todo frame (sem tracking).
"""
import os
import cv2
import numpy as np
import threading
import insightface
from insightface.app import FaceAnalysis

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "inswapper_128.onnx")

_SWAP_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
_CPU_PROVIDERS  = ["CPUExecutionProvider"]


class _Face:
    __slots__ = ("kps",)
    def __init__(self, kps):
        self.kps = kps


class FaceSwapperGPU:
    def __init__(self):
        self.source_face = None
        self._lock       = threading.Lock()
        self.ready       = False
        self.status      = "Carregando modelos..."
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.status = "Carregando detector (CPU)..."
            self.analyzer = FaceAnalysis(
                name="buffalo_l", providers=_CPU_PROVIDERS
            )
            self.analyzer.prepare(ctx_id=0, det_size=(640, 640))

            self.status = "Carregando detector rapido (CPU)..."
            self.fast_det = insightface.model_zoo.get_model(
                os.path.join(os.path.expanduser("~"), ".insightface",
                             "models", "buffalo_sc", "det_500m.onnx"),
                providers=_CPU_PROVIDERS
            )
            self.fast_det.prepare(ctx_id=0, input_size=(320, 320), det_thresh=0.5)

            self.status = "Carregando inswapper na GPU (CUDA)..."
            self.swapper = insightface.model_zoo.get_model(
                MODEL_PATH, providers=_SWAP_PROVIDERS
            )

            self.status = "Warmup CUDA..."
            self._warmup()

            used  = self.swapper.session.get_providers()
            accel = "CUDA GPU" if "CUDAExecutionProvider" in used else "CPU"
            self.ready  = True
            self.status = f"Pronto! [{accel}] Faca upload de um rosto."
        except Exception as e:
            self.status = f"Erro: {e}"
            import traceback; traceback.print_exc()

    def _warmup(self):
        try:
            ns = self.swapper.input_names
            d0 = np.zeros((1, 3, 128, 128), dtype=np.float32)
            ld = self.swapper.emap.shape[1] if hasattr(self.swapper, "emap") else 512
            d1 = np.zeros((1, ld), dtype=np.float32)
            for _ in range(5):
                self.swapper.session.run(self.swapper.output_names, {ns[0]: d0, ns[1]: d1})
        except Exception as e:
            print(f"[warmup] {e}")

    def set_source_face(self, image_bytes):
        if not self.ready:
            return False, "Modelos ainda carregando."
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "Imagem invalida."
        faces = self.analyzer.get(img)
        if not faces:
            return False, "Nenhum rosto detectado na imagem."
        best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        if getattr(best, "normed_embedding", None) is None:
            return False, "Nao foi possivel extrair embedding."
        with self._lock:
            self.source_face = best
        return True, "Rosto carregado!"

    def clear_source(self):
        with self._lock:
            self.source_face = None

    def get_status(self):
        return self.status

    def process_frame(self, frame):
        """Roda o swap completo em cada frame. ~30ms na GPU T4."""
        with self._lock:
            if not self.ready or self.source_face is None:
                return frame
            source = self.source_face

        try:
            _, kps_all = self.fast_det.detect(frame, max_num=1)
            if kps_all is None or not len(kps_all):
                return frame
        except Exception:
            return frame

        target = _Face(kps_all[0])
        try:
            return self.swapper.get(frame, target, source, paste_back=True)
        except Exception as e:
            print(f"[swap] {e}")
            return frame
