import os
import cv2
from tqdm import tqdm

def upscale_images(input_folder, output_folder):  # Tăng từ 32x32 lên 128x128
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for class_name in os.listdir(input_folder):
        class_input_path = os.path.join(input_folder, class_name)
        class_output_path = os.path.join(output_folder, class_name)
        
        if os.path.isdir(class_input_path):
            if not os.path.exists(class_output_path):
                os.makedirs(class_output_path)
            
            for img_name in tqdm(os.listdir(class_input_path), desc=f"Processing {class_name}"):
                img_path = os.path.join(class_input_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    print(f"Original size: {img.shape}")
                    upscaled_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
                    print(f"New size: {upscaled_img.shape}")
                    cv2.imwrite(os.path.join(class_output_path, img_name), upscaled_img)
                
if __name__ == "__main__":
    input_dir = "cleaned_cifar10"  # Thay bằng thư mục chứa ảnh CIFAR-10
    output_dir = "cifar-10-upscaled"  # Thư mục đầu ra
    upscale_images(input_dir, output_dir)  # Tăng từ 32x32 lên 128x128
    print("Tăng độ phân giải hoàn tất!")