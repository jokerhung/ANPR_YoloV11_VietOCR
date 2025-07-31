# FastANPR Web API

A RESTful web API for license plate recognition using FastANPR (open-image-models + fast-plate-ocr).

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Start the Server
```bash
python ocr_server_fastanpr.py
```

The server will start on `http://localhost:8087`

### 3. Test the API
```bash
# Test with image file
curl "http://localhost:8087/reg?file=images/car.jpg&confidence=0.5"

# Or use the test client
python test_api_client.py images/car.jpg
```

## 📋 API Endpoints

### 🔤 OCR Recognition
**Endpoint:** `GET /reg`

**Parameters:**
- `file` (required): Path to image file
- `confidence` (optional): Detection confidence threshold (0.0-1.0, default: 0.5)

**Example:**
```bash
curl "http://localhost:8087/reg?file=images/car.jpg&confidence=0.7"
```

**Response:**
```json
{
  "success": true,
  "file_path": "images/car.jpg",
  "timestamp": "2025-01-31T10:30:45.123456",
  "statistics": {
    "total_detections": 2,
    "successful_ocr": 2,
    "confidence_threshold": 0.7
  },
  "results": [
    {
      "detection_id": 1,
      "bbox": [100, 200, 300, 250],
      "detection_confidence": 0.95,
      "ocr_text": "ABC123",
      "ocr_confidence": 0.89
    },
    {
      "detection_id": 2,
      "bbox": [400, 180, 600, 230],
      "detection_confidence": 0.87,
      "ocr_text": "XYZ789",
      "ocr_confidence": 0.92
    }
  ]
}
```

### ❤️ Health Check
**Endpoint:** `GET /health`

**Example:**
```bash
curl "http://localhost:8087/health"
```

**Response:**
```json
{
  "status": "healthy",
  "service": "FastANPR OCR API",
  "timestamp": "2025-01-31T10:30:45.123456",
  "processor_initialized": true
}
```

### ℹ️ API Information
**Endpoint:** `GET /`

**Example:**
```bash
curl "http://localhost:8087/"
```

## 🔧 Configuration

### Server Settings
The server runs on:
- **Host:** `0.0.0.0` (accessible from all interfaces)
- **Port:** `8087` (configurable via PORT variable)
- **Debug:** `False` (production mode)

### OCR Text Cleaning ✨ **NEW**
OCR results are automatically cleaned before being returned:
- **Square brackets** `[` `]` are removed
- **Underscores** `_` are removed  
- **Extra spaces** are normalized

**Examples:**
- Raw OCR: `[ABC_123]` → API Response: `ABC123`
- Raw OCR: `XY_Z_789` → API Response: `XYZ789`

### Model Configuration
Default models used:
- **Detection:** `yolo-v9-t-384-license-plate-end2end` (balanced speed/accuracy)
- **OCR:** `cct-xs-v1-global-model` (fastest)

To change models, edit the `init_processor()` function in `ocr_server_fastanpr.py`:

```python
def init_processor():
    processor = FastANPRProcessor(
        detection_model="yolo-v9-s-608-license-plate-end2end",  # Higher accuracy
        ocr_model="cct-s-v1-global-model"  # Higher accuracy
    )
```

## 📊 Performance

### Lazy Loading
- Models are loaded only on the first request
- Subsequent requests are much faster
- Initial request may take 10-15 seconds for model loading

### Processing Speed
- **Detection:** ~80ms per image (CPU)
- **OCR:** ~0.3ms per license plate (CPU)
- **Total:** ~100-200ms per image with 1-2 plates

### Memory Usage
- **RAM:** ~2-3GB with models loaded
- **Temporary files:** Automatically cleaned up after processing

## 🛠️ Testing

### Test Client
Use the provided test client:
```bash
# Test with automatic image discovery
python test_api_client.py

# Test with specific image
python test_api_client.py path/to/your/image.jpg
```

### Manual Testing
```bash
# Health check
curl "http://localhost:8087/health"

# API info
curl "http://localhost:8087/"

# OCR processing
curl "http://localhost:8087/reg?file=images/test.jpg&confidence=0.5"
```

### Python Integration
```python
import requests

# Simple OCR request
response = requests.get(
    "http://localhost:8087/reg",
    params={"file": "images/car.jpg", "confidence": 0.5}
)

if response.status_code == 200:
    result = response.json()
    if result['success']:
        for detection in result['results']:
            print(f"License Plate: {detection['ocr_text']}")
```

## 🚨 Error Handling

### Common Error Responses

**Missing file parameter:**
```json
{
  "success": false,
  "error": "Missing required parameter: file",
  "usage": "GET /reg?file=path/to/image.jpg&confidence=0.5"
}
```

**File not found:**
```json
{
  "success": false,
  "error": "File not found: /path/to/image.jpg",
  "timestamp": "2025-01-31T10:30:45.123456"
}
```

**Processing error:**
```json
{
  "success": false,
  "error": "Error details here",
  "timestamp": "2025-01-31T10:30:45.123456"
}
```

### No Detections
When no license plates are detected:
```json
{
  "success": true,
  "file_path": "images/empty.jpg",
  "detections": [],
  "ocr_results": [],
  "message": "No license plates detected",
  "timestamp": "2025-01-31T10:30:45.123456"
}
```

## 🔒 Security Considerations

### File Access
- API only accepts file paths, not file uploads
- Ensure proper file system permissions
- Consider restricting access to specific directories

### Production Deployment
For production use:
1. **Use a WSGI server** (Gunicorn, uWSGI)
2. **Add authentication** (API keys, JWT)
3. **Implement rate limiting**
4. **Use HTTPS**
5. **Add input validation**

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8087 ocr_server_fastanpr:app
```

## 📈 Monitoring

### Logs
The server prints processing logs:
```
Processing OCR request for file: images/car.jpg
OCR processing complete: 2 detections, 2 successful OCR
```

### Health Monitoring
Use the `/health` endpoint for monitoring:
```bash
# Check if service is healthy
curl -f "http://localhost:8087/health" || echo "Service is down"
```

## 🔧 Troubleshooting

### Common Issues

**Models not loading:**
- Check internet connection for model downloads
- Ensure sufficient RAM (2-3GB free)
- Check disk space for model cache

**Connection refused:**
- Verify server is running: `python ocr_server_fastanpr.py`
- Check port 8087 is not used by other services
- Verify firewall settings

**Slow performance:**
- Use GPU-enabled ONNX runtime for faster processing
- Consider using smaller/faster models
- Ensure adequate system resources

**File not found errors:**
- Use absolute paths for better reliability
- Check file permissions
- Verify image file formats (.jpg, .png, etc.)

## 📝 Examples

### Batch Processing
```python
import requests
import os

def process_folder(folder_path, api_url="http://localhost:8087"):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            
            response = requests.get(
                f"{api_url}/reg",
                params={"file": file_path, "confidence": 0.5}
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'filename': filename,
                    'result': result
                })
    
    return results

# Process all images in a folder
results = process_folder("images/")
for item in results:
    print(f"File: {item['filename']}")
    for detection in item['result'].get('results', []):
        print(f"  License Plate: {detection['ocr_text']}")
```

### Integration with Database
```python
import requests
import sqlite3

def save_ocr_results(image_path, db_path="results.db"):
    # Get OCR results
    response = requests.get(
        "http://localhost:8087/reg",
        params={"file": image_path}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        # Save to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        for detection in result.get('results', []):
            cursor.execute("""
                INSERT INTO license_plates 
                (image_path, ocr_text, confidence, bbox, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                image_path,
                detection['ocr_text'],
                detection['ocr_confidence'],
                str(detection['bbox']),
                result['timestamp']
            ))
        
        conn.commit()
        conn.close()
```

## 📄 License

MIT License - See the main project LICENSE file for details.