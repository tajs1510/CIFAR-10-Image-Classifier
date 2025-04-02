import os
import cv2
import numpy as np
from tqdm import tqdm

# Thư mục chứa dữ liệu CIFAR-10 gốc
data_path = "data"

# Thư mục để lưu ảnh sau khi làm sạch
cleaned_path = "cleaned_cifar10"
os.makedirs(cleaned_path, exist_ok=True)

# Duyệt qua từng thư mục con (các lớp ảnh)
for class_name in os.listdir(data_path):
    class_path = os.path.join(data_path, class_name)
    cleaned_class_path = os.path.join(cleaned_path, class_name)

    # Tạo thư mục mới cho lớp nếu chưa tồn tại
    os.makedirs(cleaned_class_path, exist_ok=True)

    if os.path.isdir(class_path):
        for img_name in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            img_path = os.path.join(class_path, img_name)
            new_img_path = os.path.join(cleaned_class_path, img_name)

            try:
                # Đọc ảnh
                img = cv2.imread(img_path)

                # Kiểm tra ảnh lỗi
                if img is None:
                    print(f"Removing corrupted image: {img_path}")
                    continue

                # Chuyển ảnh về RGB nếu cần
                if len(img.shape) == 2 or img.shape[2] != 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # Resize ảnh về 32x32
                img = cv2.resize(img, (32, 32))

                # Chuẩn hóa giá trị pixel về 0-1
                img = img.astype(np.float32) / 255.0

                # Lưu ảnh vào thư mục mới
                cv2.imwrite(new_img_path, (img * 255).astype(np.uint8))

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
