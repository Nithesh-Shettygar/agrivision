from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64
import urllib.request
from threading import Thread, Lock
import time

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load YOLO models
fruit_detection_model = YOLO(r"C:\Users\hp\Downloads\Fruit-Ripeness-and-Disease-Detection-main\Fruit-Ripeness-and-Disease-Detection-main\weights_3\best.pt")
banana_disease_detection_model = YOLO(r"C:\Users\hp\Downloads\Fruit-Ripeness-and-Disease-Detection-main\Fruit-Ripeness-and-Disease-Detection-main\train\weights\best.pt")
mango_disease_detection_model = YOLO(r"C:/Users/hp/Downloads/Fruit-Ripeness-and-Disease-Detection-main/Fruit-Ripeness-and-Disease-Detection-main/train/weights/best.pt")
pomogranate_disease_detection_model = YOLO(r"C:\Users\hp\Downloads\Fruit-Ripeness-and-Disease-Detection-main\Fruit-Ripeness-and-Disease-Detection-main\train4\weights\best.pt")

class IPCamera:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.stream = None
        
    def start(self):
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self):
        while self.running:
            try:
                self.stream = urllib.request.urlopen(self.url)
                bytes_data = bytes()
                while self.running:
                    bytes_data += self.stream.read(1024)
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.frame = frame
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.5)
                
    def read(self):
        with self.lock:
            return self.frame if self.frame is not None else np.zeros((480, 640, 3), np.uint8)
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        if self.stream:
            self.stream.close()

# Initialize camera
camera = IPCamera("http://100.99.245.168:8080/video")
camera.start()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fruit_detection')
def fruit_detection():
    return render_template('fruit_detection.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/water')
def water():
    return render_template('water.html')

def generate_frames():
    while True:
        try:
            frame = camera.read()
            
            # Process frame with YOLO
            results = fruit_detection_model(frame, verbose=False)
            for result in results:
                frame = result.plot()
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        except Exception as e:
            print(f"Frame generation error: {e}")
            time.sleep(0.1)
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    try:
        image_data = request.json['image_data'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        results = fruit_detection_model(image)
        detected_objects = []
        for result in results:
            boxes = result.boxes.xywh.cpu()
            clss = result.boxes.cls.cpu().tolist()
            names = result.names
            confs = result.boxes.conf.float().cpu().tolist()

            for box, cls, conf in zip(boxes, clss, confs):
                detected_objects.append({
                    'class': names[cls], 
                    'bbox': box.tolist(), 
                    'confidence': conf
                })

        return jsonify(detected_objects)
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify([])

# ... (keep the rest of your routes the same) ...

if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        camera.stop()