from ultralytics import YOLO
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

# Cấu hình và khởi tạo mô hình OCR
config = Cfg.load_config_from_name('vgg_transformer')
#config['weights'] = 'vgg_transformerocr.pth'  # Hoặc giữ nguyên nếu đã có
config['device'] = 'cpu'  # Sử dụng CPU, có thể thay đổi thành 'cuda' nếu có GPU
detector = Predictor(config)

# Load the YOLO model and perform inference on the specified image
img_path = 'images\\20250716\\In_3416214B88B4390004543732_Manual_20250716103939795_1.png'
#img_path = 'images\\test\\image001.jpg'
model = YOLO('model\\license-plate-finetune-v1x.pt')
results = model.predict(source=img_path)

# Đọc ảnh bằng OpenCV
img_cv2 = cv2.imread(img_path)

# Truy cập bounding boxes
for result in results:
    boxes = result.boxes
    # Nếu có box, tiến hành crop và hiển thị
    if boxes is not None and boxes.xyxy is not None:
        for i, box in enumerate(boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

            # Crop ảnh theo bounding box
            cropped = img_cv2[y1:y2, x1:x2]
            cv2.imwrite(f'crop_{i+1}.jpg', cropped)
            #cv2.imshow('cropped', cropped)

            # Chuyển OpenCV BGR -> RGB, rồi sang PIL
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cropped_rgb)
            #cv2.imwrite(f'pil_image_{i+1}.jpg', pil_image)
            #cv2.imshow('pil_image', pil_image)

            # Nhận dạng chữ
            text = detector.predict(pil_image)
            print(f"Biển số {i+1}: {text}")

            # Vẽ hình chữ nhật lên ảnh gốc
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            cv2.putText(img_cv2, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Hiển thị ảnh gốc sau khi vẽ box
cv2.imshow('Final Result', img_cv2)

# Chờ người dùng nhấn phím để đóng cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()