from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from flask_cors import CORS
import gdown
url = "https://drive.google.com/file/d/1tFrMafXv94Ya7rOf_d96-tcwdNE3y12Y/view?usp=drive_link"
output = "mask_detector_model.h5"
gdown.download(url, output, quiet=False)

app = Flask(__name__)
CORS(app)  # Allows frontend to connect

# Load face detection model
haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if required files exist
if os.path.exists("With_mask.npy") and os.path.exists("Without_mask.npy"):
    # Load trained mask detection data
    with_mask = np.load("With_mask.npy")
    without_mask = np.load("Without_mask.npy")

    # Prepare dataset
    X = np.r_[with_mask.reshape(len(with_mask), -1), without_mask.reshape(len(without_mask), -1)]
    labels = np.zeros(X.shape[0])
    labels[len(with_mask):] = 1.0  # 1 = No Mask, 0 = Mask

    # Train SVM Model
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    svm = SVC()
    svm.fit(X_pca, labels)
else:
    print("Error: Required .npy files not found. Please run collect_data.py and train_model.py first.")
    svm = None
    pca = None

@app.route("/detect", methods=["POST"])
def detect():
    if svm is None or pca is None:
        return jsonify({"error": "Model not loaded. Ensure you have trained the model."}), 500
    
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image provided."}), 400
    
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    faces = haar_data.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    results = []

    for x, y, w, h in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (50, 50)).reshape(1, -1)
        face_pca = pca.transform(face)
        prediction = int(svm.predict(face_pca)[0])
        results.append({"x": x, "y": y, "w": w, "h": h, "label": "No Mask" if prediction else "Mask"})

    return jsonify({"faces": results})
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Face Mask Detection API is running!"})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
