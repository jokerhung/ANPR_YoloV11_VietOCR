import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
from PIL import Image

# Cấu hình model VietOCR
config = Cfg.load_config_from_name('vgg_transformer')
#config['weights'] = 'weights/transformerocr.pth'  # Đường dẫn nếu bạn đã tải model thủ công
config['device'] = 'cpu'  # Tự động dùng GPU nếu có

# Khởi tạo predictor
detector = Predictor(config)

# Thư mục chứa ảnh
input_folder = 'images\\test2'  # Thay đổi đường dẫn tới thư mục chứa ảnh của bạn

# File output
output_file = 'images\\test2\\label.txt'

with open(output_file, 'w', encoding='utf-8') as out:
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            cropped_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_rgb)
            
            try:
                text = detector.predict(pil_image)
            except Exception as e:
                text = f""

            out.write(f"{filename}\t{text}\n")
            print(f"Processed: {filename} - Detected text: {text}")

print(f"OCR complete. Results saved to: {output_file}")
