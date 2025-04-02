from flask import Flask, jsonify, send_from_directory, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import numpy as np
import os
from PIL import Image

app = Flask(__name__, static_folder="cifar-10-upscaled")
CORS(app, resources={r"/*": {"origins": "*"}})

# Load models
models = {
    'cnn': load_model('models/cnn_model.h5'),
    'resnet': load_model('models/resnet_model.h5'),
    'vgg': load_model('models/vgg_model.h5')
}

# Class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Corrected image folder path
IMAGE_DIR = os.path.join(os.getcwd(), "cifar-10-upscaled")  # Đảm bảo thư mục này tồn tại trong working directory

# Get all categories and images
@app.route('/categories', methods=['GET'])
def get_categories():
    categories = {}
    if not os.path.exists(IMAGE_DIR):
        return jsonify({"error": "Image directory not found"}), 500
    for category in os.listdir(IMAGE_DIR):
        category_path = os.path.join(IMAGE_DIR, category)
        if os.path.isdir(category_path):
            categories[category] = os.listdir(category_path)[:6]  # Giới hạn 6 ảnh/mục
    return jsonify(categories)

# Serve image files (có caching)
@app.route('/image/<category>/<filename>', methods=['GET'])
def get_image(category, filename):
    image_path = os.path.join(IMAGE_DIR, category, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404
    response = send_file(image_path, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response

# Prepare image for model
def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict endpoint (dựa trên URL)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_url = data.get("image_url")
    model_type = data.get("model_type", "cnn")
    
    if not image_url or model_type not in models:
        return jsonify({"error": "Invalid request"}), 400
    
    # Extract relative path from URL
    image_path = image_url.split("/image/")[-1]
    full_path = os.path.join(IMAGE_DIR, image_path.replace("/", os.sep))  # Đảm bảo đúng đường dẫn

    if not os.path.exists(full_path):
        return jsonify({"error": "Image not found"}), 404
    
    img_array = prepare_image(full_path)
    prediction = models[model_type].predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return jsonify({
        "class_name": class_names[predicted_class],
        "confidence": float(prediction[0][predicted_class])
    })

# Upload endpoint: Nhận file upload và dự đoán
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    # Lấy model type từ form-data (mặc định là 'cnn')
    model_type = request.form.get("model_type", "cnn")
    if model_type not in models:
        return jsonify({"error": "Invalid model type"}), 400

    # Tạo thư mục tạm nếu chưa tồn tại
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        img_array = prepare_image(temp_path)
        prediction = models[model_type].predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = {
            "class_name": class_names[predicted_class],
            "confidence": float(prediction[0][predicted_class])
        }
    except Exception as e:
        result = {"error": str(e)}
    finally:
        # Xóa file tạm nếu muốn
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
