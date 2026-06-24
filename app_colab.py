"""
Servidor Flask para rodar no Google Colab com GPU.
Usa Socket.IO + eventlet para receber frames do browser e devolver o resultado.
"""
import os
import base64
import cv2
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from face_swap_gpu import FaceSwapperGPU

app      = Flask(__name__)
app.config["SECRET_KEY"] = "faceswaplive2"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    max_http_buffer_size=5 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)
swapper  = FaceSwapperGPU()


@app.route("/")
def index():
    return render_template("index_colab.html")


@app.route("/status")
def status():
    return jsonify({"status": swapper.get_status(), "ready": swapper.ready})


@socketio.on("frame")
def handle_frame(data):
    """Recebe frame JPEG em base64, processa, devolve resultado."""
    try:
        img_bytes = base64.b64decode(data.split(",")[1])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        result = swapper.process_frame(frame)

        _, buf      = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 82])
        result_b64  = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        emit("result", result_b64)
    except Exception as e:
        print(f"[frame] {e}")


@socketio.on("upload_face")
def handle_upload(data):
    """Recebe foto do rosto fonte em base64."""
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
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
