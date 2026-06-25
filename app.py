import cv2
import threading
import queue
import time
from flask import Flask, Response, render_template, request, jsonify

# Baixa o modelo automaticamente se nao existir
import baixar_modelo
baixar_modelo.main(silent=True)

from face_swap import FaceSwapper

app = Flask(__name__)
swapper = FaceSwapper()

# ── Estado compartilhado ────────────────────────────────────────────────────
_camera = None
_camera_lock = threading.Lock()
_streaming = False

# Fila de frames brutos (maxsize=1 → sempre processa o mais recente)
_raw_q: queue.Queue = queue.Queue(maxsize=1)

# Frame mais recente para exibição (processado ou bruto como fallback)
_display_frame = None
_display_lock = threading.Lock()

# Métricas
_fps_swap = 0.0
_fps_display = 0.0


# ── Thread de captura ────────────────────────────────────────────────────────
def _capture_loop():
    global _streaming
    while _streaming:
        with _camera_lock:
            cam = _camera
        if cam is None:
            break
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.02)
            continue

        # Reduz para processamento (~2x mais rápido que 640x480)
        small = cv2.resize(frame, (480, 360))

        # Descarta frame antigo se a thread de swap ainda não consumiu
        try:
            _raw_q.put_nowait(small)
        except queue.Full:
            pass

        # Fallback: enquanto não tem resultado do swap, mostra câmera ao vivo
        with _display_lock:
            global _display_frame
            if _display_frame is None:
                _display_frame = small

        time.sleep(0.01)


# ── Thread de swap (roda em background, sem bloquear câmera) ─────────────────
def _swap_loop():
    global _fps_swap
    t0 = time.time()
    count = 0

    while _streaming:
        try:
            frame = _raw_q.get(timeout=0.5)
        except queue.Empty:
            continue

        processed = swapper.process_frame(frame)

        with _display_lock:
            global _display_frame
            _display_frame = processed

        count += 1
        elapsed = time.time() - t0
        if elapsed >= 2.0:
            _fps_swap = round(count / elapsed, 1)
            count = 0
            t0 = time.time()


# ── Gerador MJPEG ────────────────────────────────────────────────────────────
def _gen_mjpeg():
    global _fps_display
    t0 = time.time()
    count = 0

    while True:
        with _display_lock:
            frame = _display_frame

        if frame is None:
            time.sleep(0.04)
            continue

        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )

        count += 1
        elapsed = time.time() - t0
        if elapsed >= 2.0:
            _fps_display = round(count / elapsed, 1)
            count = 0
            t0 = time.time()

        time.sleep(0.033)  # ~30 FPS no display (independente do swap)


# ── Rotas ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({
        "status": swapper.get_status(),
        "ready": swapper.ready,
        "fps_swap": _fps_swap,
        "fps_display": _fps_display,
    })


@app.route("/upload_face", methods=["POST"])
def upload_face():
    if "face" not in request.files:
        return jsonify({"ok": False, "msg": "Nenhum arquivo enviado."}), 400
    data = request.files["face"].read()
    ok, msg = swapper.set_source_face(data)
    return jsonify({"ok": ok, "msg": msg}), (200 if ok else 400)


@app.route("/clear_face", methods=["POST"])
def clear_face():
    swapper.clear_source()
    return jsonify({"ok": True})


@app.route("/start_camera", methods=["POST"])
def start_camera():
    global _camera, _streaming, _display_frame

    with _camera_lock:
        if _camera is None:
            _camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            _camera.set(cv2.CAP_PROP_FPS, 30)

        if not _camera.isOpened():
            _camera = None
            return jsonify({"ok": False, "msg": "Webcam nao encontrada."}), 500

    _display_frame = None
    _streaming = True
    threading.Thread(target=_capture_loop, daemon=True).start()
    threading.Thread(target=_swap_loop, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global _camera, _streaming, _display_frame
    _streaming = False
    time.sleep(0.3)
    with _camera_lock:
        if _camera:
            _camera.release()
            _camera = None
    _display_frame = None
    return jsonify({"ok": True})


@app.route("/video_feed")
def video_feed():
    return Response(
        _gen_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    print("\nFace Swap Live iniciando...")
    print("   Abra: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
