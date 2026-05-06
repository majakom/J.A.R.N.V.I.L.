from flask import Flask, Response, render_template_string
import cv2
from picamera2 import Picamera2

app = Flask(__name__)
picam2 = Picamera2()

picam2.configure(picam2.create_video_configuration())
picam2.start()  

HTML = """
<html>
<head>
    <title>Camera</title>
    <style>
        body {
            margin: 0;
            background: black;
            color: white;
            font-family: Arial;
            text-align: center;
        }

        h1 {
            margin: 10px;
            color: lime;
        }

        img {
            width: 800px;
        }
    </style>
</head>
<body>
    <h1>Raspberry Pi Camera Feed</h1>
    <img src="/video">
</body>
</html>
"""

def generate():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5000)