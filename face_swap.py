import os
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
cv2.setNumThreads(4)

import numpy as np
import threading
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "inswapper_128.onnx")
_DET_PATH  = os.path.join(os.path.expanduser("~"), ".insightface", "models",
                           "buffalo_sc", "det_500m.onnx")

_PROC_W, _PROC_H = 320, 240


class _Face:
    __slots__ = ("kps",)
    def __init__(self, kps):
        self.kps = kps


class FaceSwapper:
    def __init__(self):
        self.source_analyzer = None
        self.fast_det        = None
        self.swapper         = None
        self.source_face     = None

        # Ultimo resultado neural completo (frame com swap aplicado)
        self._last_result = None
        self._last_kps    = None
        self._result_lock = threading.Lock()

        self._det_lock = threading.Lock()

        self.ready  = False
        self.status = "Carregando modelos..."
        threading.Thread(target=self._load_models, daemon=True).start()

    # ── Inicialização ──────────────────────────────────────────────────────

    def _load_models(self):
        try:
            self.status = "Carregando det_500m..."
            self.fast_det = insightface.model_zoo.get_model(
                _DET_PATH, providers=["CPUExecutionProvider"]
            )
            self.fast_det.prepare(ctx_id=0, input_size=(160, 160), det_thresh=0.5)

            self.status = "Carregando buffalo_l..."
            self.source_analyzer = FaceAnalysis(
                name="buffalo_l", providers=["CPUExecutionProvider"]
            )
            self.source_analyzer.prepare(ctx_id=0, det_size=(320, 320))

            self.status = "Carregando inswapper (DirectML)..."
            self.swapper = insightface.model_zoo.get_model(
                MODEL_PATH,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"]
            )

            self.status = "Warmup GPU (aguarde ~20s)..."
            self._warmup()

            used  = self.swapper.session.get_providers()
            accel = "Intel GPU (DirectML)" if "DmlExecutionProvider" in used else "CPU"
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
            for _ in range(3):
                self.swapper.session.run(self.swapper.output_names, {ns[0]: d0, ns[1]: d1})
        except Exception as e:
            print(f"[warmup] {e}")

    # ── API pública ────────────────────────────────────────────────────────

    def set_source_face(self, image_bytes):
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
        with self._result_lock:
            self.source_face = best
            self._last_result = None
            self._last_kps    = None
        return True, "Rosto carregado!"

    def clear_source(self):
        with self._result_lock:
            self.source_face  = None
            self._last_result = None
            self._last_kps    = None

    def get_status(self):
        return self.status

    # ── process_frame: roda o swap completo (~840ms) ───────────────────────
    # Chamado pelo swap thread. Retorna frame com swap aplicado pelo insightface.

    def process_frame(self, frame):
        with self._result_lock:
            if not self.ready or self.source_face is None:
                return frame
            source = self.source_face

        small = cv2.resize(frame, (_PROC_W, _PROC_H))

        try:
            with self._det_lock:
                _, kps_all = self.fast_det.detect(small, max_num=1)
            if kps_all is None or not len(kps_all):
                # Sem rosto: retorna ultimo resultado se disponivel
                with self._result_lock:
                    return self._last_result if self._last_result is not None else frame
        except Exception:
            return frame

        kps    = kps_all[0]
        target = _Face(kps)

        try:
            # paste_back=True: insightface aplica o blend correto internamente
            result = self.swapper.get(small, target, source, paste_back=True)
        except Exception as e:
            print(f"[swap] {e}")
            return frame

        result_full = cv2.resize(result, (frame.shape[1], frame.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        with self._result_lock:
            self._last_result = result_full
            self._last_kps    = kps

        return result_full

    # ── composite_fast: reloca o ultimo swap na nova posicao (~35ms) ───────
    # Chamado pelo capture thread para manter o display fluido entre swaps.

    def composite_fast(self, frame):
        with self._result_lock:
            if not self.ready or self.source_face is None:
                return frame
            last_result = self._last_result
            last_kps    = self._last_kps

        if last_result is None:
            return frame

        small = cv2.resize(frame, (_PROC_W, _PROC_H))

        # Detecta posicao atual do rosto
        try:
            with self._det_lock:
                _, kps_all = self.fast_det.detect(small, max_num=1)
            if kps_all is None or not len(kps_all):
                return last_result   # sem rosto: mostra ultimo swap
        except Exception:
            return last_result

        new_kps = kps_all[0]

        # Estima transformacao entre posicao antiga e nova do rosto
        try:
            M, inliers = cv2.estimateAffinePartial2D(
                last_kps, new_kps, method=cv2.LMEDS
            )
            if M is None or (inliers is not None and inliers.sum() < 3):
                return last_result
        except Exception:
            return last_result

        # Aplica transformacao no ultimo resultado
        h_full, w_full = frame.shape[:2]
        last_small = cv2.resize(last_result, (_PROC_W, _PROC_H))
        warped     = cv2.warpAffine(last_small, M, (_PROC_W, _PROC_H),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        # Mascara suave ao redor do rosto para nao mostrar artefatos do warp
        cx  = float(new_kps[:, 0].mean())
        cy  = float(new_kps[:, 1].mean())
        # Escala da face baseada na distancia entre olhos (kps 0 e 1)
        eye_dist = float(np.linalg.norm(new_kps[0] - new_kps[1]))
        rx = max(int(eye_dist * 1.6), 20)
        ry = max(int(eye_dist * 2.0), 24)

        face_mask = np.zeros((_PROC_H, _PROC_W), dtype=np.float32)
        cv2.ellipse(face_mask, (int(cx), int(cy)), (rx, ry), 0, 0, 360, 1.0, -1)
        face_mask = cv2.GaussianBlur(face_mask, (31, 31), 10)
        m3 = face_mask[:, :, np.newaxis]

        blended = (warped.astype(np.float32) * m3
                   + small.astype(np.float32) * (1.0 - m3)).astype(np.uint8)

        return cv2.resize(blended, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
