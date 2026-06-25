import os
import base64
import threading
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from face_swap_gpu import FaceSwapperGPU

app      = Flask(__name__)
app.config["SECRET_KEY"] = "faceswaplive2"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=2 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)
swapper     = FaceSwapperGPU()
_busy       = False
_busy_lock  = threading.Lock()


@app.route("/")
def index():
    return render_template("index_colab.html")


@app.route("/status")
def status():
    return jsonify({"status": swapper.get_status(), "ready": swapper.ready})


@socketio.on("frame")
def handle_frame(data):
    global _busy
    # Descarta frame se ainda processando o anterior — evita fila que atrasa tudo
    with _busy_lock:
        if _busy:
            return
        _busy = True

    try:
        img_bytes = base64.b64decode(data.split(",")[1])
        frame     = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        result = swapper.process_frame(frame)

        _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 75])
        emit("result", "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
    except Exception as e:
        print(f"[frame] {e}")
    finally:
        with _busy_lock:
            _busy = False


@socketio.on("upload_face")
def handle_upload(data):
    try:
        img_bytes = base64.b64decode(data.split(",")[1])
        ok, msg   = swapper.set_source_face(img_bytes)
        emit("upload_result", {"ok": ok, "msg": msg})
    except Exception as e:
        emit("upload_result", {"ok": False, "msg": str(e)})


@socketio.on("clear_face")
def handle_clear():
    swapper.clear_source()


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 use_reloader=False, allow_unsafe_werkzeug=True)
