# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en")

# Run OCR inference on a sample image 
result = ocr.predict(
    input="D:\\jkhung\\work\\vetc\\github\\ANPR_YoloV11_VietOCR\\images\\test2\\In_1965155_Manual_20250716084600263_1_crop_1.jpg")

# Visualize the results and save the JSON results
for res in result:
    rec_texts = res.get('rec_texts')
    combined_text = ''.join(rec_texts)
    print("Recognized texts:", combined_text)