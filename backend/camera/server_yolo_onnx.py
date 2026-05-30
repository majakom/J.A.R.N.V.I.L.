from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from picamera2 import Picamera2
import onnxruntime as ort

app = Flask(__name__)

# =========================
# CAMERA (KEEP WORKING PIPELINE)
# =========================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration())
picam2.start()

# =========================
# ONNX MODEL
# =========================
session = ort.InferenceSession(
    "best.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
IMG_SIZE = 640

# =========================
# HTML
# =========================
HTML = """
<html>
<head>
    <title>YOLO ONNX</title>
</head>
<body style="margin:0;background:black;text-align:center;color:white;">
<h2 style="color:lime;">YOLO ONNX Pi Zero 2 W</h2>
<img src="/video" style="width:800px;max-width:100%;">
</body>
</html>
"""

# =========================
# PREPROCESS (MATCHES REAL YOLO TRAINING)
# =========================
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # IMPORTANT FIX:
    # YOLO .pt → OpenCV uses BGR internally
    # so we KEEP BGR (DO NOT convert to RGB)
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)

    return img

# =========================
# INFERENCE
# =========================
def run_inference(frame):
    h, w = frame.shape[:2]

    inp = preprocess(frame)
    outputs = session.run(None, {input_name: inp})[0]
    print(outputs[0].shape)
    print(outputs[0][0][:10])

    for det in outputs[0]:
        if len(det) < 6:
            continue

        x1, y1, x2, y2, score, class_id = det[:6]

        if score < 0.2:
            continue

        scale_x = w / IMG_SIZE
        scale_y = h / IMG_SIZE

        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(frame,
                    f"{int(class_id)} {score:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    1)

    return frame

# =========================
# STREAM (USE YOUR WORKING COLOR PIPELINE)
# =========================
def generate():
    frame_counter = 0
    last_frame = None

    while True:
        frame = picam2.capture_array()

        # FIX COLORS FIRST (THIS IS YOUR WORKING BASE)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # run YOLO every 5 frames
        if frame_counter % 5 == 0:
            last_frame = run_inference(frame.copy())

        frame_counter += 1

        if last_frame is None:
            last_frame = frame

        out = last_frame

        _, buffer = cv2.imencode(
            ".jpg",
            out,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video")
def video():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# =========================
# START
# =========================
app.run(host="0.0.0.0", port=5000, threaded=True)