# FastANPR - License Plate Detection using open-image-models

This module provides a complete ANPR pipeline using the [open-image-models](https://github.com/ankandrew/open-image-models) library for detection and [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) for OCR:

1. **detect_plate_from_image()** - Detect license plates from image files
2. **crop_image()** - Crop detected license plate regions 
3. **scan_folder_and_process()** - Process all images in a folder
4. **ocr_cropped_images()** - Perform OCR on cropped license plate images ✨ **NEW**
5. **ocr_single_image()** - Perform OCR on a single cropped image ✨ **NEW**
6. **create_label_file_from_cropped_folder()** - Create label file from cropped images ✨ **NEW**
7. **create_label_file_from_ocr_results()** - Create label file from OCR results ✨ **NEW**

## Features

- 🚀 **Fast Detection**: Uses pre-trained YOLOv9 models via ONNX for fast inference
- 📷 **Multiple Models**: Choose from different model sizes (256px to 608px) based on speed vs accuracy needs
- 📁 **Batch Processing**: Process entire folders of images automatically
- 💾 **Auto-cropping**: Automatically crop detected license plates and save them
- 🔤 **OCR Integration**: Built-in OCR using fast-plate-ocr for text recognition ✨ **NEW**
- ⚡ **Lightweight OCR**: Fast CCT-based models for license plate text recognition
- 📄 **Label File Generation**: Auto-create training labels in filename\ttext format ✨ **NEW**
- 🧹 **Text Cleaning**: Automatic removal of brackets and underscores from OCR results ✨ **NEW**
- 📊 **Statistics**: Get detailed processing statistics and results including OCR success rates
- 🎯 **Confidence Filtering**: Filter detections by confidence threshold
- 🔄 **Complete Pipeline**: Detection → Cropping → OCR → Label Generation in a single workflow

## Installation

Make sure you have the required dependencies installed:

```bash
pip install open-image-models[onnx] fast-plate-ocr[onnx] opencv-python numpy
```

Or if you have GPU support:
```bash
pip install open-image-models[onnx-gpu] fast-plate-ocr[onnx-gpu] opencv-python numpy
```

For other hardware acceleration options:
```bash
# Intel CPU optimization
pip install fast-plate-ocr[onnx-openvino]

# Windows DirectML support  
pip install fast-plate-ocr[onnx-directml]

# Qualcomm chipsets
pip install fast-plate-ocr[onnx-qnn]
```

## Available Models

### Detection Models

| Model | Image Size | Speed | Accuracy | Use Case |
|-------|------------|--------|----------|----------|
| yolo-v9-t-256-license-plate-end2end | 256px | ⚡⚡⚡ Fastest | ⭐⭐⭐ Good | Real-time processing |
| yolo-v9-t-384-license-plate-end2end | 384px | ⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | **Recommended** |
| yolo-v9-t-416-license-plate-end2end | 416px | ⚡⚡ Fast | ⭐⭐⭐⭐ Very Good | Balanced |
| yolo-v9-t-512-license-plate-end2end | 512px | ⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | High accuracy |
| yolo-v9-t-640-license-plate-end2end | 640px | ⚡ Medium | ⭐⭐⭐⭐⭐ Excellent | High accuracy |
| yolo-v9-s-608-license-plate-end2end | 608px | 🐌 Slower | ⭐⭐⭐⭐⭐ Best | Maximum accuracy |

### OCR Models ✨ **NEW**

| Model | Size | Speed | Accuracy | Latency (CPU) | Use Case |
|-------|------|--------|----------|---------------|----------|
| cct-xs-v1-global-model | XS | ⚡⚡⚡ Fastest | ⭐⭐⭐⭐ Very Good | ~0.3ms | **Recommended** |
| cct-s-v1-global-model | S | ⚡⚡ Fast | ⭐⭐⭐⭐⭐ Excellent | ~0.6ms | High accuracy |

## Quick Start

### 1. Complete ANPR Pipeline (Detection + Cropping + OCR)

```python
from ocr_folders_fastanpr import FastANPRProcessor

# Initialize with both detection and OCR models
processor = FastANPRProcessor(
    detection_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# Complete pipeline in one go
image_path = "images/car_with_plate.jpg"
detections = processor.detect_plate_from_image(image_path)
if detections:
    cropped_paths = processor.crop_image(image_path, detections)
    ocr_results = processor.ocr_cropped_images(cropped_paths)
    
    for result in ocr_results:
        print(f"License Plate: '{result['text']}' (confidence: {result['confidence']:.3f})")
```

### 2. Step-by-Step Processing

```python
from ocr_folders_fastanpr import detect_plate_from_image, crop_image, ocr_cropped_images

# Step 1: Detect license plates
image_path = "images/car_with_plate.jpg"
detections = detect_plate_from_image(image_path, confidence_threshold=0.5)

# Step 2: Crop detected plates
if detections:
    cropped_paths = crop_image(image_path, detections, "output/cropped")
    print(f"Cropped {len(cropped_paths)} license plates")
    
    # Step 3: Perform OCR
    ocr_results = ocr_cropped_images(cropped_paths)
    for result in ocr_results:
        print(f"OCR: '{result['text']}'")
```

### 3. Folder Processing with Complete Pipeline

```python
from ocr_folders_fastanpr import scan_folder_and_process

# Process all images in a folder with detection, cropping, and OCR
results = scan_folder_and_process(
    input_folder="images",
    output_folder="output",
    detection_model="yolo-v9-t-384-license-plate-end2end",
    confidence_threshold=0.5
)

# Print results summary
print(f"Processed {results['processed_images']} images")
print(f"Found {results['total_detections']} license plates")
print(f"OCR results: {results['total_ocr_results']}")
print(f"Successful OCR: {results['successful_ocr']}")

# Access individual results
for result in results['results']:
    if result['status'] == 'success':
        print(f"\nImage: {result['image_path']}")
        for ocr in result['ocr_results']:
            print(f"  License Plate: '{ocr['text']}' (confidence: {ocr['confidence']:.3f})")
```

### 4. Class-based Usage with OCR

```python
from ocr_folders_fastanpr import FastANPRProcessor

# Initialize processor with both models
processor = FastANPRProcessor(
    detection_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# Process single image
detections = processor.detect_plate_from_image("image.jpg")
cropped_paths = processor.crop_image("image.jpg", detections)
ocr_results = processor.ocr_cropped_images(cropped_paths)

# Process folder
results = processor.scan_folder_and_process("images", "output")
```

### 5. Label File Creation ✨ **NEW**

```python
from ocr_folders_fastanpr import create_label_file_from_cropped_folder

# Method 1: Create labels from cropped folder (performs OCR)
create_label_file_from_cropped_folder(
    cropped_folder="output/cropped",
    output_file="labels.txt",
    ocr_model="cct-xs-v1-global-model"
)

# Method 2: Create labels from existing OCR results (no re-processing)
from ocr_folders_fastanpr import ocr_cropped_images, create_label_file_from_ocr_results

cropped_files = ["crop1.jpg", "crop2.jpg"]
ocr_results = ocr_cropped_images(cropped_files)
create_label_file_from_ocr_results(ocr_results, "labels.txt")

# Note: When using scan_folder_and_process(), label.txt is automatically created!
```

## Function Details

### detect_plate_from_image()

```python
def detect_plate_from_image(image_path: str, 
                           detection_model: str = "yolo-v9-t-384-license-plate-end2end",
                           confidence_threshold: float = 0.5) -> List[dict]:
```

**Parameters:**
- `image_path`: Path to the image file
- `detection_model`: Model name (see available models above)
- `confidence_threshold`: Minimum confidence score (0.0-1.0)

**Returns:**
- List of dictionaries with `bbox` and `confidence` keys

**Example:**
```python
detections = detect_plate_from_image("car.jpg", confidence_threshold=0.7)
# Returns: [{'bbox': [x1, y1, x2, y2], 'confidence': 0.85}, ...]
```

### crop_image()

```python
def crop_image(image_path: str, 
               detections: List[dict], 
               output_dir: str = "output/cropped") -> List[str]:
```

**Parameters:**
- `image_path`: Path to the original image
- `detections`: Detection results from `detect_plate_from_image()`
- `output_dir`: Directory to save cropped images

**Returns:**
- List of paths to saved cropped images

### scan_folder_and_process()

```python
def scan_folder_and_process(input_folder: str,
                           output_folder: str = "output",
                           detection_model: str = "yolo-v9-t-384-license-plate-end2end",
                           confidence_threshold: float = 0.5) -> dict:
```

**Parameters:**
- `input_folder`: Folder containing images to process
- `output_folder`: Output folder for results
- `detection_model`: Model name for detection
- `confidence_threshold`: Minimum confidence score

**Returns:**
- Dictionary with processing statistics and results (now includes OCR statistics)

### ocr_cropped_images() ✨ **NEW**

```python
def ocr_cropped_images(cropped_image_paths: List[str], 
                      ocr_model: str = "cct-xs-v1-global-model") -> List[dict]:
```

**Parameters:**
- `cropped_image_paths`: List of paths to cropped license plate images
- `ocr_model`: Model name for OCR recognition

**Returns:**
- List of OCR result dictionaries with `image_path`, `text`, and `confidence` keys

**Example:**
```python
ocr_results = ocr_cropped_images(["crop1.jpg", "crop2.jpg"])
# Returns: [{'image_path': 'crop1.jpg', 'text': 'ABC123', 'confidence': 0.95}, ...]
```

### ocr_single_image() ✨ **NEW**

```python
def ocr_single_image(image_path: str, 
                    ocr_model: str = "cct-xs-v1-global-model") -> dict:
```

**Parameters:**
- `image_path`: Path to the cropped license plate image
- `ocr_model`: Model name for OCR recognition

**Returns:**
- OCR result dictionary with `image_path`, `text`, and `confidence` keys

**Example:**
```python
result = ocr_single_image("cropped_plate.jpg")
# Returns: {'image_path': 'cropped_plate.jpg', 'text': 'ABC123', 'confidence': 0.95}
```

### create_label_file_from_cropped_folder() ✨ **NEW**

```python
def create_label_file_from_cropped_folder(cropped_folder: str, 
                                         output_file: str = "output/results/label.txt",
                                         ocr_model: str = "cct-xs-v1-global-model") -> None:
```

**Parameters:**
- `cropped_folder`: Path to folder containing cropped license plate images
- `output_file`: Path where to save the label.txt file
- `ocr_model`: Model name for OCR recognition

**Description:**
Creates a label file from a folder of cropped images by performing OCR and saving results in tab-separated format.

**Example:**
```python
create_label_file_from_cropped_folder("output/cropped", "labels.txt")
```

### create_label_file_from_ocr_results() ✨ **NEW**

```python
def create_label_file_from_ocr_results(ocr_results: List[dict], 
                                      output_file: str = "output/results/label.txt") -> None:
```

**Parameters:**
- `ocr_results`: List of OCR result dictionaries from ocr_cropped_images()
- `output_file`: Path where to save the label.txt file

**Description:**
Creates a label file from existing OCR results without re-processing images.

**Example:**
```python
ocr_results = ocr_cropped_images(["crop1.jpg", "crop2.jpg"])
create_label_file_from_ocr_results(ocr_results, "labels.txt")
```

## Output Structure

When using `scan_folder_and_process()`, the output folder will contain:

```
output/
├── cropped/                    # Cropped license plate images
│   ├── image1_cropped_1.jpg
│   ├── image1_cropped_2.jpg
│   └── ...
└── results/
    ├── processing_results.json # Detailed results, statistics, and OCR text
    └── label.txt              # Tab-separated filename-to-text mappings ✨ NEW
```

### Label File Format ✨ **NEW**

The `label.txt` file contains tab-separated mappings of cropped image filenames to their OCR text:

```
image1_cropped_1.jpg	ABC123
image1_cropped_2.jpg	XYZ789
image2_cropped_1.jpg	DEF456
car3_cropped_1.jpg	GHI789
```

**Format:** `filename\tocr_text`

#### Text Cleaning ✨ **NEW**
OCR text is automatically cleaned before writing to the label file:
- **Square brackets** `[` `]` are removed
- **Underscores** `_` are removed  
- **Extra spaces** are normalized

**Example:**
- Raw OCR: `[ABC_123]` → Cleaned: `ABC123`
- Raw OCR: `XY_Z_789` → Cleaned: `XYZ789`

This format is perfect for:
- **Training datasets**: Use as ground truth labels for ML models
- **Data validation**: Manually verify OCR accuracy  
- **Export/Import**: Easy integration with other tools
- **Analysis**: Quick overview of all recognized license plates

**Enhanced processing_results.json structure:**
```json
{
  "total_images": 10,
  "processed_images": 8,
  "total_detections": 12,
  "cropped_images": 12,
  "total_ocr_results": 12,
  "successful_ocr": 10,
  "failed_images": [],
  "results": [
    {
      "image_path": "images/car1.jpg",
      "detections": [...],
      "cropped_paths": ["output/cropped/car1_cropped_1.jpg"],
      "ocr_results": [
        {
          "image_path": "output/cropped/car1_cropped_1.jpg",
          "text": "ABC123",
          "confidence": 0.95
        }
      ],
      "status": "success"
    }
  ]
}
```

## Testing

Run the test script to verify everything works:

```bash
python test_fastanpr.py
```

This will test:
- Single image processing with OCR
- Folder processing with complete pipeline
- Different detection model comparisons
- OCR functionality with different models ✨ **NEW**
- Label file creation and format validation ✨ **NEW**

## Tips for Best Results

1. **Model Selection:**
   - Use `yolo-v9-t-256-license-plate-end2end` for real-time applications
   - Use `yolo-v9-t-384-license-plate-end2end` for balanced speed/accuracy (recommended)
   - Use `yolo-v9-s-608-license-plate-end2end` for maximum accuracy

2. **Confidence Threshold:**
   - Start with `0.5` for general use
   - Lower to `0.3` if you're missing detections
   - Raise to `0.7+` if you're getting false positives

3. **Image Quality:**
   - Better lighting improves detection accuracy
   - Higher resolution images work better
   - Avoid extremely blurry or distorted images

## Integration with OCR

The cropped license plate images can be easily integrated with OCR systems:

```python
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2

# Setup OCR
config = Cfg.load_config_from_name('vgg_transformer')
config['device'] = 'cpu'
ocr_detector = Predictor(config)

# Process images with FastANPR + OCR
detections = detect_plate_from_image("car.jpg")
cropped_paths = crop_image("car.jpg", detections)

for crop_path in cropped_paths:
    # Load cropped image
    img = cv2.imread(crop_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    
    # OCR recognition
    text = ocr_detector.predict(pil_image)
    print(f"License plate text: {text}")
```

## Error Handling

The functions include comprehensive error handling:
- Invalid image paths
- Corrupted image files
- Missing directories
- Model loading errors
- Processing failures

All errors are logged with descriptive messages.

## Performance

### Detection Performance
Approximate processing times on CPU (Intel i7):
- **yolo-v9-t-256**: ~50ms per image
- **yolo-v9-t-384**: ~80ms per image  
- **yolo-v9-t-512**: ~120ms per image
- **yolo-v9-s-608**: ~200ms per image

### OCR Performance ✨ **NEW**
Approximate processing times per cropped license plate:
- **cct-xs-v1-global-model**: ~0.3ms per plate (CPU)
- **cct-s-v1-global-model**: ~0.6ms per plate (CPU)

### Complete Pipeline Performance
For a typical image with 1-2 license plates:
- **Detection + Cropping + OCR**: ~85-125ms (CPU)
- **GPU acceleration**: 3-5x faster for detection, 2-3x for OCR

**Note**: GPU processing is significantly faster. Install GPU-enabled ONNX runtime for best performance. 