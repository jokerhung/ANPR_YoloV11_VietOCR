import cv2
import os
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from fast_alpr import ALPR
from flask import Flask, request, jsonify
import json
import traceback
import time
import logging
import logging.handlers
from datetime import datetime
import psutil
import requests
import tempfile
import urllib.parse
from urllib.parse import urlparse
from mylicence import get_machine_key
from mylicence import verify_lic

class FastANPRProcessor:
    """Fast ANPR processor using FastALPR library"""
    
    def __init__(self, 
                 detection_model: str = "yolo-v9-s-608-license-plate-end2end",
                 ocr_model: str = "cct-s-v1-global-model"):
        """
        Initialize the FastANPR processor
        
        Args:
            detection_model: Model name for license plate detection
                Available models:
                - yolo-v9-s-608-license-plate-end2end (highest accuracy)
                - yolo-v9-t-640-license-plate-end2end
                - yolo-v9-t-512-license-plate-end2end
                - yolo-v9-t-416-license-plate-end2end
                - yolo-v9-t-384-license-plate-end2end (balanced)
                - yolo-v9-t-256-license-plate-end2end (fastest)
            ocr_model: Model name for OCR text recognition
                Available models:
                - cct-xs-v1-global-model (fastest, recommended)
                - cct-s-v1-global-model (higher accuracy)
        """
        if logger:
            logger.info(f"Initializing FastALPR with detection model: {detection_model} and OCR model: {ocr_model}")
        else:
            print(f"Initializing FastALPR with detection model: {detection_model} and OCR model: {ocr_model}")
        
        self.alpr = ALPR(
            detector_model=detection_model,
            ocr_model=ocr_model
        )
        
        if logger:
            logger.info("FastALPR initialized successfully!")
        else:
            print("FastALPR initialized successfully!")



    def predict_alpr(self, image_path: str, confidence_threshold: float = 0.5) -> List[dict]:
        """
        Perform complete ALPR (detection + OCR) on an image file using FastALPR
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            List of ALPR result dictionaries containing bounding boxes, confidence scores, and OCR text
            Format: [{'bbox': [x1, y1, x2, y2], 'detection_confidence': float, 'ocr_text': str, 'ocr_confidence': float}, ...]
        """
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                return []
            
            # Load image using OpenCV to verify it's valid
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image: {image_path}")
                return []
            
            print(f"Processing ALPR for image: {image_path}")
            print(f"Image shape: {image.shape}")
            
            # Perform license plate detection and OCR using FastALPR
            alpr_results = self.alpr.predict(image_path)
            #print(f"ALPR results: {alpr_results}")
            
            # Process results and filter by confidence
            processed_results = []
            for i, result in enumerate(alpr_results):
                # Extract bounding box coordinates and detection confidence
                # Handle both flat structure (result.bounding_box) and nested structure (result.detection.bounding_box)
                if hasattr(result, 'detection') and result.detection:
                    # Nested structure: ALPRResult with detection property
                    bbox = result.detection.bounding_box
                    detection_confidence = result.detection.confidence
                else:
                    # Flat structure: direct properties on result
                    bbox = result.bounding_box
                    detection_confidence = result.confidence
                
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                
                if detection_confidence >= confidence_threshold:
                    # Extract OCR results if available
                    ocr_text = ""
                    ocr_confidence = 0.0
                    if hasattr(result, 'ocr') and result.ocr:
                        ocr_text = self._clean_ocr_text(result.ocr.text if hasattr(result.ocr, 'text') else str(result.ocr))
                        ocr_confidence = result.ocr.confidence if hasattr(result.ocr, 'confidence') else 1.0
                    
                    processed_results.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'detection_confidence': float(detection_confidence),
                        'ocr_text': ocr_text,
                        'ocr_confidence': float(ocr_confidence)
                    })
                    print(f"ALPR result {i+1}: bbox=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], detection_conf={detection_confidence:.3f}, text='{ocr_text}', ocr_conf={ocr_confidence:.3f}")
            
            print(f"Found {len(processed_results)} license plates above confidence threshold {confidence_threshold}")
            return processed_results
            
        except Exception as e:
            print(f"Error performing ALPR on image {image_path}: {str(e)}")
            return []

    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean OCR text by removing unwanted characters
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text with square brackets and underscores removed
        """
        if not text:
            return text
            
        # Remove square brackets and underscores
        cleaned = text.replace('[', '').replace(']', '').replace('_', '').replace('\'', '')
        
        # Remove extra spaces and strip
        cleaned = ' '.join(cleaned.split())
        
        return cleaned

# Flask Web API
app = Flask(__name__)

# Global processor instance
processor = None

# Global logger instance
logger = None

def download_image_from_url(url, timeout=30, max_size=50*1024*1024):
    """
    Download an image from URL and save to temporary file
    
    Args:
        url: HTTP/HTTPS URL to the image
        timeout: Request timeout in seconds
        max_size: Maximum file size in bytes (50MB default)
        
    Returns:
        tuple: (temp_file_path, success, error_message)
    """
    try:
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or parsed_url.scheme not in ['http', 'https']:
            return None, False, "Invalid URL: Must be http or https"
        
        if logger:
            logger.info(f"Downloading image from URL: {url}")
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download with streaming and size check
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
            return None, False, f"Invalid content type: {content_type}. Expected image."
        
        # Check content length if available
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size:
            return None, False, f"File too large: {content_length} bytes (max: {max_size})"
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        downloaded_size = 0
        
        # Download in chunks
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                downloaded_size += len(chunk)
                if downloaded_size > max_size:
                    temp_file.close()
                    os.unlink(temp_file.name)
                    return None, False, f"File too large: {downloaded_size} bytes (max: {max_size})"
                temp_file.write(chunk)
        
        temp_file.close()
        
        # Verify it's a valid image by trying to load it
        test_image = cv2.imread(temp_file.name)
        if test_image is None:
            os.unlink(temp_file.name)
            return None, False, "Downloaded file is not a valid image"
        
        if logger:
            logger.info(f"Successfully downloaded image: {downloaded_size} bytes -> {temp_file.name}")
        
        return temp_file.name, True, None
        
    except requests.exceptions.Timeout:
        return None, False, f"Request timeout after {timeout} seconds"
    except requests.exceptions.ConnectionError:
        return None, False, "Connection error: Unable to reach the URL"
    except requests.exceptions.HTTPError as e:
        return None, False, f"HTTP error: {e.response.status_code} - {e.response.reason}"
    except requests.exceptions.RequestException as e:
        return None, False, f"Request error: {str(e)}"
    except Exception as e:
        return None, False, f"Unexpected error downloading image: {str(e)}"

@app.route('/reg', methods=['GET'])
def recognize_license_plate():
    """
    OCR endpoint: http://localhost:8086/reg?file=file_path
    
    Query Parameters:
        file (str): Path to the image file to process
        confidence (float, optional): Confidence threshold (default: 0.5)
        
    Returns:
        JSON response with OCR results
    """
    # Record start time for processing duration calculation
    start_time = time.time()
    
    try:
        # Get file parameter
        file_path = request.args.get('file')
        confidence_threshold = float(request.args.get('confidence', 0.5))
        
        if not file_path:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: file',
                'usage': 'GET /reg?file=path/to/image.jpg&confidence=0.5'
            }), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            processing_duration = round((time.time() - start_time) * 1000, 1)
            return jsonify({
                'success': False,
                'message': f'File not found: {file_path}',
                'processing_time': processing_duration,
            }), 404
        
        # Use global processor (initialized at startup)
        global processor
        if processor is None:
            return jsonify({
                'success': False,
                'message': 'Processor not initialized',
                'processing_time': round((time.time() - start_time) * 1000, 1),
            }), 500
        
        # Process the image
        if logger:
            logger.info(f"Processing OCR request for file: {file_path}")
        else:
            print(f"Processing OCR request for file: {file_path}")
        
        # Perform complete ALPR (detection + OCR) using FastALPR
        alpr_results = processor.predict_alpr(file_path, confidence_threshold)
        
        if not alpr_results:
            processing_duration = round((time.time() - start_time) * 1000, 1)
            return jsonify({
                'success': True,
                'file_path': file_path,
                'detections': [],
                'ocr_results': [],
                'message': 'No license plates detected',
                'processing_time': processing_duration,
            })
        
        # Prepare response from unified ALPR results
        results = []
        for i, alpr_result in enumerate(alpr_results):
            result = {
                'detection_id': i + 1,
                'bbox': alpr_result['bbox'],
                'detection_confidence': alpr_result['detection_confidence'],
                'ocr_text': alpr_result['ocr_text'],
                'ocr_confidence': alpr_result['ocr_confidence']
            }
            results.append(result)
        
        # Count successful OCR results
        successful_ocr = len([r for r in results if r['ocr_text'].strip()])
        
        # Calculate processing duration
        processing_duration = round((time.time() - start_time) * 1000, 1)
        
        response = {
            'success': True,
            'message': 'SUCCESS',
            'file_path': file_path,
            'processing_time': processing_duration,
            'results': results
        }
        
        if logger:
            logger.info(f"ALPR processing complete: {len(alpr_results)} detections, {successful_ocr} successful OCR, processing time: {processing_duration}ms, results: {results}")
        else:
            print(f"ALPR processing complete: {len(alpr_results)} detections, {successful_ocr} successful OCR")
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        if logger:
            logger.error(f"Error processing OCR request: {error_msg}")
            logger.error(traceback.format_exc())
        else:
            print(f"Error processing OCR request: {error_msg}")
            print(traceback.format_exc())
        
        processing_duration = round((time.time() - start_time) * 1000, 1)
        return jsonify({
            'success': False,
            'message': error_msg,
            'processing_time': processing_duration,
        }), 500

@app.route('/regURL', methods=['GET'])
def recognize_license_plate_url():
    """
    OCR endpoint for images from URLs: http://localhost:8087/regURL?url=http_url
    
    Query Parameters:
        url (str): HTTP/HTTPS URL to the image file to process
        confidence (float, optional): Confidence threshold (default: 0.5)
        timeout (int, optional): Download timeout in seconds (default: 30)
        
    Returns:
        JSON response with OCR results
    """
    # Record start time for processing duration calculation
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Get URL parameter
        image_url = request.args.get('url')
        confidence_threshold = float(request.args.get('confidence', 0.5))
        timeout = int(request.args.get('timeout', 30))
        
        if not image_url:
            return jsonify({
                'success': False,
                'message': 'Missing required parameter: url',
                'usage': 'GET /regURL?url=http://example.com/image.jpg&confidence=0.5&timeout=30'
            }), 400
        
        # Use global processor (initialized at startup)
        global processor
        if processor is None:
            return jsonify({
                'success': False,
                'message': 'Processor not initialized',
                'processing_time': round((time.time() - start_time) * 1000, 1),
            }), 500
        
        # Download image from URL
        temp_file_path, download_success, error_message = download_image_from_url(image_url, timeout)
        
        if not download_success:
            processing_duration = round((time.time() - start_time) * 1000, 1)
            return jsonify({
                'success': False,
                'message': f'Failed to download image: {error_message}',
                'url': image_url,
                'processing_time': processing_duration,
            }), 400
        
        # Process the downloaded image
        if logger:
            logger.info(f"Processing OCR request for URL: {image_url}")
        else:
            print(f"Processing OCR request for URL: {image_url}")
        
        # Perform complete ALPR (detection + OCR) using FastALPR
        alpr_results = processor.predict_alpr(temp_file_path, confidence_threshold)
        
        if not alpr_results:
            processing_duration = round((time.time() - start_time) * 1000, 1)
            return jsonify({
                'success': True,
                'url': image_url,
                'detections': [],
                'ocr_results': [],
                'message': 'No license plates detected',
                'processing_time': processing_duration,
            })
        
        # Prepare response from unified ALPR results
        results = []
        for i, alpr_result in enumerate(alpr_results):
            result = {
                'detection_id': i + 1,
                'bbox': alpr_result['bbox'],
                'detection_confidence': alpr_result['detection_confidence'],
                'ocr_text': alpr_result['ocr_text'],
                'ocr_confidence': alpr_result['ocr_confidence']
            }
            results.append(result)
        
        # Count successful OCR results
        successful_ocr = len([r for r in results if r['ocr_text'].strip()])
        
        # Calculate processing duration
        processing_duration = round((time.time() - start_time) * 1000, 1)
        
        response = {
            'success': True,
            'message': 'SUCCESS',
            'url': image_url,
            'processing_time': processing_duration,
            'results': results
        }
        
        if logger:
            logger.info(f"ALPR processing complete for URL: {image_url} - {len(alpr_results)} detections, {successful_ocr} successful OCR, processing time: {processing_duration}ms, results: {results}")
        else:
            print(f"ALPR processing complete: {len(alpr_results)} detections, {successful_ocr} successful OCR")
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        if logger:
            logger.error(f"Error processing OCR request for URL {image_url}: {error_msg}")
            logger.error(traceback.format_exc())
        else:
            print(f"Error processing OCR request: {error_msg}")
            print(traceback.format_exc())
        
        processing_duration = round((time.time() - start_time) * 1000, 1)
        return jsonify({
            'success': False,
            'message': error_msg,
            'url': image_url if 'image_url' in locals() else 'unknown',
            'processing_time': processing_duration,
        }), 500
    
    finally:
        # Clean up downloaded temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                if logger:
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not delete temporary file {temp_file_path}: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'FastANPR OCR API',
        'timestamp': datetime.now().isoformat(),
        'processor_initialized': processor is not None
    })

@app.route('/', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'service': 'FastANPR OCR API',
        'version': '1.0.0',
        'endpoints': {
            'ocr': {
                'url': '/reg',
                'method': 'GET',
                'parameters': {
                    'file': 'Path to image file (required)',
                    'confidence': 'Detection confidence threshold 0.0-1.0 (optional, default: 0.5)'
                },
                'example': '/reg?file=images/car.jpg&confidence=0.7',
                'description': 'OCR processing for local image files'
            },
            'ocr_url': {
                'url': '/regURL',
                'method': 'GET',
                'parameters': {
                    'url': 'HTTP/HTTPS URL to image file (required)',
                    'confidence': 'Detection confidence threshold 0.0-1.0 (optional, default: 0.5)',
                    'timeout': 'Download timeout in seconds (optional, default: 30)'
                },
                'example': '/regURL?url=https://example.com/car.jpg&confidence=0.7&timeout=60',
                'description': 'OCR processing for images from URLs'
            },
            'health': {
                'url': '/health',
                'method': 'GET',
                'description': 'Health check endpoint'
            }
        },
        'timestamp': datetime.now().isoformat()
    })

def setup_logging(log_file="fastanpr_server.log", log_level=logging.INFO):
    """Setup logging configuration"""
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "logs"
        if log_dir and log_dir != "":
            os.makedirs(log_dir, exist_ok=True)
        else:
            log_file = os.path.join("logs", os.path.basename(log_file))
            os.makedirs("logs", exist_ok=True)
        
        # Configure logging format
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Setup daily rotating file handler (rotate at midnight, keep 30 days)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='D', interval=1, backupCount=30, encoding='utf-8'
        )
        # Add date suffix to rotated files
        file_handler.suffix = '%Y-%m-%d'
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format,
            handlers=[file_handler, console_handler]
        )
        
        # Create logger for this module
        logger = logging.getLogger('FastANPR')
        logger.info(f"Logging initialized - Daily rotating log file: {log_file} (keeps {file_handler.backupCount} days)")
        return logger
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=log_level)
        logger = logging.getLogger('FastANPR')
        logger.warning("Using basic logging configuration due to setup error")
        return logger

def load_config(config_file="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        if logger:
            logger.info(f"Configuration loaded from {config_file}")
        else:
            print(f"Configuration loaded from {config_file}")
        return config
    except FileNotFoundError:
        msg = f"Warning: {config_file} not found, using default configuration"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return {
            "server": {"port": 8087, "host": "0.0.0.0"},
            "models": {
                "detection_model": "yolo-v9-t-384-license-plate-end2end",
                "ocr_model": "cct-s-v1-global-model"
            }
        }
    except json.JSONDecodeError as e:
        msg1 = f"Error parsing {config_file}: {e}"
        msg2 = "Using default configuration"
        if logger:
            logger.error(msg1)
            logger.info(msg2)
        else:
            print(msg1)
            print(msg2)
        return {
            "server": {"port": 8087, "host": "0.0.0.0"},
            "models": {
                "detection_model": "yolo-v9-t-384-license-plate-end2end",
                "ocr_model": "cct-s-v1-global-model"
            }
        }
    except Exception as e:
        msg1 = f"Unexpected error loading {config_file}: {e}"
        msg2 = "Using default configuration"
        if logger:
            logger.error(msg1)
            logger.info(msg2)
        else:
            print(msg1)
            print(msg2)
        return {
            "server": {"port": 8087, "host": "0.0.0.0"},
            "models": {
                "detection_model": "yolo-v9-t-384-license-plate-end2end",
                "ocr_model": "cct-s-v1-global-model"
            }
        }

# Example usage
if __name__ == "__main__":
    
    # Setup logging with daily rotation
    today = datetime.now().strftime('%Y-%m-%d')
    logger = setup_logging(f"logs/fastanpr_server_{today}.log")
    
    # Load configuration
    config = load_config()

    # load licence key
    # INSERT_YOUR_CODE
    # Load license key from file 'lic.key'
    LICENCE_FILE = "licence.key"
    
    try:
        # get machine key
        uid = get_machine_key()
        # Write uid to machine.key only if it does not exist
        if not os.path.exists("machine.key"):
            with open("machine.key", "w") as f:
                f.write(uid)
        # read licence key
        with open(LICENCE_FILE, "r") as f:
            license_b32 = f.read().strip()
        # You may want to set your public key here
        PUBLIC_KEY_B32 = config["licence"]["public_key"]
        if not PUBLIC_KEY_B32:
            logger.error("Public key for license verification not set. Set FASTANPR_PUBLIC_KEY environment variable.")
            raise Exception("Public key for license verification not set.")
        verify_lic(license_b32, PUBLIC_KEY_B32, uid)
        logger.info("License verification successful.")
    except Exception as e:
        logger.error(f"License verification failed: {e}")
        print(f"License verification failed: {e}")
        exit(1)
    
    def init_processor():
        """Initialize the FastANPR processor"""
        global processor
        if processor is None:
            detection_model = config["models"]["detection_model"]
            ocr_model = config["models"]["ocr_model"]
            logger.info(f"Initializing FastANPR processor with models: {detection_model}, {ocr_model}")
            processor = FastANPRProcessor(
                detection_model=detection_model,
                ocr_model=ocr_model
            )
            logger.info("FastANPR processor initialized successfully!")
        return processor
    
    # Limit CPU usage to 50% of available cores
    # process = psutil.Process(os.getpid())
    # total_cores = psutil.cpu_count(logical=True)
    # cores_to_use = max(1, int((total_cores * 0.8) / 100))
    # process.cpu_affinity(list(range(cores_to_use)))

    # Configuration from JSON
    PORT = config["server"]["port"]
    HOST = config["server"]["host"]
    
    logger.info("=" * 60)
    logger.info("FastANPR OCR Web API Server")
    logger.info("=" * 60)
    logger.info(f"Server: http://{HOST}:{PORT}")
    logger.info(f"Detection Model: {config['models']['detection_model']}")
    logger.info(f"OCR Model: {config['models']['ocr_model']}")
    logger.info("")
    logger.info("Available endpoints:")
    logger.info(f"  • OCR (File): GET http://localhost:{PORT}/reg?file=path/to/image.jpg")
    logger.info(f"  • OCR (URL):  GET http://localhost:{PORT}/regURL?url=http://example.com/image.jpg")
    logger.info(f"  • Health:     GET http://localhost:{PORT}/health")
    logger.info(f"  • Info:       GET http://localhost:{PORT}/")
    logger.info("")
    logger.info("Example usage:")
    logger.info(f"  curl 'http://localhost:{PORT}/reg?file=images/car.jpg&confidence=0.5'")
    logger.info(f"  curl 'http://localhost:{PORT}/regURL?url=https://example.com/car.jpg&confidence=0.5&timeout=30'")
    logger.info(f"  curl 'http://localhost:{PORT}/regURL?url=https://i.imgur.com/license_plate.jpg'")
    logger.info(f"  curl 'http://localhost:{PORT}/regURL?url=https://via.placeholder.com/300x150.jpg'")
    logger.info("")
    logger.info("Note: regURL endpoint downloads images from HTTP/HTTPS URLs (max 50MB, 30s timeout)")
    logger.info("      Supported formats: JPG, PNG, BMP, TIFF - requires valid image content-type")
    logger.info("")
    
    # Initialize processor at startup
    init_processor()
    logger.info("Note: Processor initialized at startup")
    logger.info("=" * 60)
    
    # Create temp directory for cropped images
    os.makedirs("temp_crops", exist_ok=True)
    logger.info("Created temp_crops directory for temporary files")
    
    # Run the Flask development server
    app.run(host=HOST, port=PORT, debug=False)
