import os
from ultralytics import YOLO
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# Cấu hình và khởi tạo mô hình OCR
config = Cfg.load_config_from_name('vgg_transformer')
#config['weights'] = 'vgg_transformerocr.pth'  # Hoặc giữ nguyên nếu đã có
config['device'] = 'cpu'  # Sử dụng CPU, có thể thay đổi thành 'cuda' nếu có GPU
detector = Predictor(config)

# Đường dẫn thư mục ảnh và thư mục lưu ảnh crop
input_folder = 'images\\crop'
output_file = 'images\\crop'

# Duyệt qua tất cả file trong thư mục
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        results = model.predict(source=img_path)

        # Đọc ảnh bằng OpenCV
        img_cv2 = cv2.imread(img_path)

        for result_idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None and boxes.xyxy is not None:
                for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)

                    # Crop ảnh
                    cropped = img_cv2[y1:y2, x1:x2]

                    # Tạo tên file crop
                    base_name = os.path.splitext(filename)[0]
                    crop_filename = f"{base_name}_crop_{i+1}.jpg"
                    crop_path = os.path.join(output_folder, crop_filename)

                    # Ghi ảnh
                    cv2.imwrite(crop_path, cropped)
                    print(f"[+] Saved: {crop_path}")
