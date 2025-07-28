import os
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import gc
import time
import paddle
import psutil

# Set PaddleOCR to use mobile models for lower resource usage
#paddle.set_flags({
#    "FLAGS_fraction_of_cpu_memory_to_use": 0.5,  # Limit to 50% of CPU memory
#    "FLAGS_use_pinned_memory": False,           # Disable pinned memory for a lower CPU load
#})

# Limit CPU usage to 50% of available cores
process = psutil.Process(os.getpid())
total_cores = psutil.cpu_count(logical=True)
cores_to_use = max(1, int((total_cores * 0.5) / 100))
process.cpu_affinity(list(range(cores_to_use)))

# Initialize PaddleOCR model with optimized settings
ocr = PaddleOCR(
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_detection_model_name="PP-OCRv5_mobile_det",
    #text_recognition_model_name="PP-OCRv5_server_rec",
    #text_detection_model_name="PP-OCRv5_server_det",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    cpu_threads=4,  # Increased from 2 - adjust based on your CPU cores
    enable_mkldnn=True,  # Enable Intel MKL-DNN for CPU optimization
    lang='en',
)

paddle.set_flags({
    "FLAGS_fraction_of_cpu_memory_to_use": 0.5,  # Limit to 50% of CPU memory
    "FLAGS_use_pinned_memory": False,           # Disable pinned memory for a lower CPU load
})

# Input/Output paths
input_folder = 'images\\crop2'
output_file = 'images\\crop2\\label.txt'

def preprocess_image(image_path, max_size=640):
    """
    Preprocess image to reduce computational load
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Get original dimensions
        height, width = img.shape[:2]
        
        # Resize if image is too large
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def process_images_batch():
    """
    Process images with memory management and CPU optimization
    """
    # Remove existing output file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    total_files = len(image_files)
    print(f"Found {total_files} images to process")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for idx, filename in enumerate(image_files, 1):
            try:
                #print(f"Processing {idx}/{total_files}: {filename}")
                
                image_path = os.path.join(input_folder, filename)
                
                # Preprocess image to reduce size if needed
                processed_img = preprocess_image(image_path)
                if processed_img is None:
                    print(f"Skipping {filename} - could not load image")
                    continue
                
                # Run OCR with timeout protection
                start_time = time.time()
                result = ocr.predict(processed_img)  # Use ocr.ocr() instead of predict()
                processing_time = (time.time() - start_time)*1000  # Convert to milliseconds
                
                # Extract text from results
                text_result = ''
                for res in result:
                    rec_texts = res.get('rec_texts')
                    combined_text = ''.join(rec_texts)
                    text_result += combined_text  # Combine with text_result
                
                text_result = text_result.strip()
                
                print(f"Completed {idx}/{total_files}: {filename} - Time: {processing_time:.2f}ms - Text: {text_result[:50]}")
                out.write(f"{filename}\t{text_result}\n")
                
                # Memory cleanup every 10 images
                if idx % 10 == 0:
                    gc.collect()
                    time.sleep(0.1)  # Small pause to prevent CPU overload
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                out.write(f"{filename}\tERROR: {str(e)}\n")
                continue

def main():
    """
    Main function with proper error handling
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    
    print("Starting OCR processing with optimized settings...")
    start_time = time.time()
    
    try:
        process_images_batch()
        
        total_time = time.time() - start_time
        print(f"\nOCR complete! Total time: {total_time:.2f}s")
        print(f"Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Final cleanup
        gc.collect()

if __name__ == "__main__":
    main()