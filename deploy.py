import cv2
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, Response, render_template
from flask_socketio import SocketIO
from PIL import Image
import sys
import os
from model import TemporalLocalizationModel  # Your model definition

app = Flask(__name__)
socketio = SocketIO(app)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
# Feature Extraction Setup using ResNet18
#########################################
# Load a pre-trained ResNet18 and remove its final FC layer.
resnet18 = models.resnet18(pretrained=True)
modules = list(resnet18.children())[:-1]  # Remove the classification layer.
feature_extractor = nn.Sequential(*modules)
feature_extractor.eval()
feature_extractor.to(device)

# Adapter to map ResNet18's 512-dim output to 768-dim.
adapter = nn.Linear(512, 768)
adapter.eval()
adapter.to(device)

# Define preprocessing transforms for ResNet input.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

def extract_features_from_frame(frame):
    """
    Extract a 768-dimensional feature vector from a frame using ResNet18.
    """
    # Convert BGR (OpenCV) to RGB and then to PIL image.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    input_tensor = preprocess(pil_img)  # [3, 224, 224]
    input_batch = input_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]
    with torch.no_grad():
        features = feature_extractor(input_batch)  # [1, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [1, 512]
        features = adapter(features)  # [1, 768]
    return features.cpu().numpy()[0]  # Return as a 1D numpy array (768,)

#########################################
# Load the Temporal Localization Model
#########################################
num_classes = 21  # Adjust to your configuration (background + actions)
model = TemporalLocalizationModel(feature_dim=768, hidden_dim=256, num_classes=num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

#########################################
# Streaming Video Feed Endpoint (MJPEG)
#########################################
def generate_video():
    cap = cv2.VideoCapture(0)  # Change to video file path if needed.
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

#########################################
# Inference Thread (Background Task)
#########################################
def inference_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera for inference")
        return
    window_size = 16
    window_stride = 8
    fps = 25.0
    frames_buffer = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Optionally, you can preprocess frame (resize, etc.) here.
        features = extract_features_from_frame(frame)
        frames_buffer.append(features)
        if len(frames_buffer) >= window_size:
            window_features = np.stack(frames_buffer[:window_size], axis=0)  # [window_size, 768]
            input_tensor = torch.tensor(window_features).unsqueeze(0).to(device)  # [1, window_size, 768]
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = nn.functional.softmax(outputs, dim=1)
                conf, pred_label = torch.max(probs, dim=1)
            prediction = int(pred_label.cpu().numpy()[0])
            confidence = float(conf.cpu().numpy()[0])
            timestamp = time.time()
            # Emit inference result via SocketIO
            socketio.emit('inference_result', {
                'timestamp': timestamp,
                'prediction': prediction,
                'confidence': confidence
            })
            frames_buffer = frames_buffer[window_stride:]
        socketio.sleep(0.01)
    cap.release()

# Start inference thread.
thread = threading.Thread(target=inference_thread)
thread.daemon = True
thread.start()

#########################################
# Run the Flask Server
#########################################
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
