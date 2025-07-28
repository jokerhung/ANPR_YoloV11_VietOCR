import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
from PIL import Image
from paddleocr import PaddleOCR

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    return image

def preprocess_image(image):
    """Preprocess the image for OCR."""
    # Convert the image to RGB format
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def extract_text(ocr_model, image):
    """Extract text from the image using the provided OCR model."""
    result = ocr_model.ocr(image, cls=True)
    text = ''
    for line in result:
        for word_info in line:
            text += word_info[1][0] + ' '
    return text.strip()

# Initialize PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Set language as needed

# Thư mục chứa ảnh
input_folder = 'images\\crop'  # Thay đổi đường dẫn tới thư mục chứa ảnh của bạn

# File output
output_file = 'images\\crop\\label.txt'

# Xóa file output nếu đã tồn tại
if os.path.exists(output_file):
    os.remove(output_file)

with open(output_file, 'w', encoding='utf-8') as out:
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            result = ocr.predict(input=image_path)
            # Visualize the results and save the JSON results
            text_result = ''
            for res in result:
                rec_texts = res.get('rec_texts')
                combined_text = ''.join(rec_texts)
                text_result += combined_text  # Combine with text_result
            
            print(f"Processed: {filename} - Detected text: {text_result}")
            out.write(f"{filename}\t{text_result}\n")

print(f"OCR complete. Results saved to: {output_file}")
