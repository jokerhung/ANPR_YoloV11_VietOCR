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

    def detect_plate_from_image(self, image_path: str, confidence_threshold: float = 0.5) -> List[dict]:
        """
        Detect license plates from an image file using FastALPR
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            List of detection dictionaries containing bounding boxes and confidence scores
            Format: [{'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
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
            
            print(f"Processing image: {image_path}")
            print(f"Image shape: {image.shape}")
            
            # Perform license plate detection and OCR using FastALPR
            alpr_results = self.alpr.predict(image_path)
            
            # Process results and filter by confidence
            processed_detections = []
            for i, result in enumerate(alpr_results):
                # Extract bounding box coordinates and confidence
                bbox = result.bounding_box
                x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                confidence = result.confidence
                
                if confidence >= confidence_threshold:
                    processed_detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence)
                    })
                    print(f"Detected plate: bbox=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}], confidence={confidence:.3f}")
            
            print(f"Found {len(processed_detections)} license plates above confidence threshold {confidence_threshold}")
            return processed_detections
            
        except Exception as e:
            print(f"Error detecting plates from image {image_path}: {str(e)}")
            return []

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

    def crop_image(self, image_path: str, detections: List[dict], output_dir: str = "output/cropped") -> List[str]:
        """
        Crop license plate regions from the image based on detection results
        
        Args:
            image_path: Path to the original image
            detections: List of detection dictionaries from detect_plate_from_image()
            output_dir: Directory to save cropped images
            
        Returns:
            List of paths to saved cropped images
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Load the original image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image: {image_path}")
                return []
            
            cropped_paths = []
            base_filename = Path(image_path).stem
            
            print(f"Cropping {len(detections)} detected plates from {image_path}")
            
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                confidence = detection['confidence']
                x1, y1, x2, y2 = bbox
                
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                # Crop the image
                cropped = image[y1:y2, x1:x2]
                
                if cropped.size > 0:
                    # Generate output filename
                    crop_filename = f"{base_filename}_cropped_{i+1}.jpg"
                    crop_path = os.path.join(output_dir, crop_filename)
                    
                    # Save cropped image
                    cv2.imwrite(crop_path, cropped)
                    cropped_paths.append(crop_path)
                    
                    print(f"Saved cropped plate {i+1}: {crop_path} (size: {cropped.shape})")
                else:
                    print(f"Warning: Empty crop for detection {i+1}")
            
            return cropped_paths
            
        except Exception as e:
            print(f"Error cropping image {image_path}: {str(e)}")
            return []

    def ocr_cropped_images(self, cropped_image_paths: List[str]) -> List[dict]:
        """
        Perform OCR on cropped license plate images using FastALPR
        
        Args:
            cropped_image_paths: List of paths to cropped license plate images
            
        Returns:
            List of OCR result dictionaries containing image path and recognized text
            Format: [{'image_path': str, 'text': str, 'confidence': float}, ...]
        """
        try:
            ocr_results = []
            
            print(f"Performing OCR on {len(cropped_image_paths)} cropped images")
            
            for i, crop_path in enumerate(cropped_image_paths, 1):
                try:
                    if not os.path.exists(crop_path):
                        print(f"Warning: Cropped image not found: {crop_path}")
                        continue
                    
                    print(f"OCR processing {i}/{len(cropped_image_paths)}: {os.path.basename(crop_path)}")
                    
                    # Perform OCR using FastALPR
                    alpr_results = self.alpr.predict(crop_path)
                    
                    # Extract the best OCR result (typically the first/highest confidence)
                    text = ""
                    confidence = 0.0
                    
                    if alpr_results:
                        # Use the first result (highest confidence)
                        result = alpr_results[0]
                        if hasattr(result, 'ocr') and result.ocr:
                            text = self._clean_ocr_text(result.ocr.text if hasattr(result.ocr, 'text') else str(result.ocr))
                            confidence = result.ocr.confidence if hasattr(result.ocr, 'confidence') else 1.0
                    
                    result_dict = {
                        'image_path': crop_path,
                        'text': text,
                        'confidence': confidence
                    }
                    
                    ocr_results.append(result_dict)
                    print(f"OCR result: '{text}' (confidence: {confidence:.3f})")
                    
                except Exception as e:
                    print(f"Error performing OCR on {crop_path}: {str(e)}")
                    ocr_results.append({
                        'image_path': crop_path,
                        'text': '',
                        'confidence': 0.0,
                        'error': str(e)
                    })
            
            print(f"OCR completed for {len(ocr_results)} images")
            return ocr_results
            
        except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            return []

    def ocr_single_image(self, image_path: str) -> dict:
        """
        Perform OCR on a single cropped license plate image
        
        Args:
            image_path: Path to the cropped license plate image
            
        Returns:
            OCR result dictionary containing text and confidence
            Format: {'image_path': str, 'text': str, 'confidence': float}
        """
        try:
            if not os.path.exists(image_path):
                return {
                    'image_path': image_path,
                    'text': '',
                    'confidence': 0.0,
                    'error': 'Image file not found'
                }
            
            print(f"Performing OCR on: {image_path}")
            
            # Perform OCR using FastALPR
            alpr_results = self.alpr.predict(image_path)
            
            # Extract the best OCR result (typically the first/highest confidence)
            text = ""
            confidence = 0.0
            
            if alpr_results:
                # Use the first result (highest confidence)
                result = alpr_results[0]
                if hasattr(result, 'ocr') and result.ocr:
                    text = self._clean_ocr_text(result.ocr.text if hasattr(result.ocr, 'text') else str(result.ocr))
                    confidence = result.ocr.confidence if hasattr(result.ocr, 'confidence') else 1.0
            
            result_dict = {
                'image_path': image_path,
                'text': text,
                'confidence': confidence
            }
            
            print(f"OCR result: '{text}' (confidence: {confidence:.3f})")
            return result_dict
            
        except Exception as e:
            print(f"Error performing OCR on {image_path}: {str(e)}")
            return {
                'image_path': image_path,
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }

    def scan_folder_and_process(self, 
                               input_folder: str, 
                               output_folder: str = "output",
                               confidence_threshold: float = 0.5,
                               supported_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')) -> dict:
        """
        Scan all image files in a folder and process them for license plate detection and cropping
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Path to output folder for results
            confidence_threshold: Minimum confidence score for detections
            supported_extensions: Tuple of supported image file extensions
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            # Check if input folder exists
            if not os.path.exists(input_folder):
                print(f"Error: Input folder not found: {input_folder}")
                return {}
            
            # Create output folders
            cropped_dir = os.path.join(output_folder, "cropped")
            results_dir = os.path.join(output_folder, "results")
            os.makedirs(cropped_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            
            # Find all image files
            image_files = []
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(supported_extensions):
                        image_files.append(os.path.join(root, file))
            
            print(f"Found {len(image_files)} image files in {input_folder}")
            
            # Processing statistics
            stats = {
                'total_images': len(image_files),
                'processed_images': 0,
                'total_detections': 0,
                'cropped_images': 0,
                'total_ocr_results': 0,
                'successful_ocr': 0,
                'failed_images': [],
                'results': []
            }
            
            # Process each image
            for i, image_path in enumerate(image_files, 1):
                print(f"\n--- Processing image {i}/{len(image_files)}: {os.path.basename(image_path)} ---")
                
                try:
                    # Step 1: Detect plates
                    detections = self.detect_plate_from_image(image_path, confidence_threshold)
                    
                    if detections:
                        # Step 2: Crop detected plates
                        cropped_paths = self.crop_image(image_path, detections, cropped_dir)
                        
                        # Step 3: Perform OCR on cropped images
                        ocr_results = []
                        if cropped_paths:
                            ocr_results = self.ocr_cropped_images(cropped_paths)
                        
                        # Update statistics
                        stats['total_detections'] += len(detections)
                        stats['cropped_images'] += len(cropped_paths)
                        stats['total_ocr_results'] += len(ocr_results)
                        stats['successful_ocr'] += len([r for r in ocr_results if r.get('text', '').strip()])
                        
                        # Store result
                        result = {
                            'image_path': image_path,
                            'detections': detections,
                            'cropped_paths': cropped_paths,
                            'ocr_results': ocr_results,
                            'status': 'success'
                        }
                    else:
                        print(f"No license plates detected in {os.path.basename(image_path)}")
                        result = {
                            'image_path': image_path,
                            'detections': [],
                            'cropped_paths': [],
                            'ocr_results': [],
                            'status': 'no_detections'
                        }
                    
                    stats['results'].append(result)
                    stats['processed_images'] += 1
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    stats['failed_images'].append(image_path)
                    stats['results'].append({
                        'image_path': image_path,
                        'detections': [],
                        'cropped_paths': [],
                        'ocr_results': [],
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Print final statistics
            print(f"\n=== Processing Complete ===")
            print(f"Total images: {stats['total_images']}")
            print(f"Successfully processed: {stats['processed_images']}")
            print(f"Failed images: {len(stats['failed_images'])}")
            print(f"Total detections: {stats['total_detections']}")
            print(f"Cropped images: {stats['cropped_images']}")
            print(f"OCR results: {stats['total_ocr_results']}")
            print(f"Successful OCR: {stats['successful_ocr']}")
            print(f"Output folder: {output_folder}")
            
            # Save results summary
            import json
            results_file = os.path.join(results_dir, "processing_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {results_file}")
            
            # Create label.txt file with cropped_filename\tocr_text format
            label_file = os.path.join(results_dir, "label.txt")
            self._create_label_file(stats['results'], label_file)
            print(f"Label file saved to: {label_file}")
            
            return stats
            
        except Exception as e:
            print(f"Error scanning folder {input_folder}: {str(e)}")
            return {}

    def _create_label_file(self, results: List[dict], label_file_path: str) -> None:
        """
        Create a label.txt file with format: cropped_filename\tocr_text
        
        Args:
            results: List of processing results from scan_folder_and_process
            label_file_path: Path where to save the label.txt file
        """
        try:
            with open(label_file_path, 'w', encoding='utf-8') as f:
                label_count = 0
                
                for result in results:
                    if result.get('status') == 'success' and result.get('ocr_results'):
                        for ocr_result in result['ocr_results']:
                            # Get cropped filename (just the filename, not full path)
                            cropped_path = ocr_result.get('image_path', '')
                            if cropped_path:
                                cropped_filename = os.path.basename(cropped_path)
                                ocr_text = ocr_result.get('text', '').strip()
                                
                                # Clean OCR text: remove square brackets and underscores
                                cleaned_text = self._clean_ocr_text(ocr_text)
                                
                                # Only write if we have both filename and text
                                if cropped_filename and cleaned_text:
                                    f.write(f"{cropped_filename}\t{cleaned_text}\n")
                                    label_count += 1
                                elif cropped_filename:
                                    # Write even if OCR text is empty (for completeness)
                                    f.write(f"{cropped_filename}\t\n")
                                    label_count += 1
                
                print(f"Created label file with {label_count} entries")
                
        except Exception as e:
            print(f"Error creating label file {label_file_path}: {str(e)}")

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

# Standalone utility function for cleaning OCR text
def _clean_ocr_text(text: str) -> str:
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

# Convenience functions for direct usage
def detect_plate_from_image(image_path: str, detection_model: str = "yolo-v9-t-384-license-plate-end2end", confidence_threshold: float = 0.5) -> List[dict]:
    """
    Convenience function to detect license plates from a single image
    
    Args:
        image_path: Path to the image file
        detection_model: Model name for detection
        confidence_threshold: Minimum confidence score
        
    Returns:
        List of detection dictionaries
    """
    processor = FastANPRProcessor(detection_model)
    return processor.detect_plate_from_image(image_path, confidence_threshold)


def crop_image(image_path: str, detections: List[dict], output_dir: str = "output/cropped") -> List[str]:
    """
    Convenience function to crop license plate regions from an image
    
    Args:
        image_path: Path to the original image
        detections: List of detection dictionaries
        output_dir: Directory to save cropped images
        
    Returns:
        List of paths to saved cropped images
    """
    processor = FastANPRProcessor()
    return processor.crop_image(image_path, detections, output_dir)


def scan_folder_and_process(input_folder: str, 
                           output_folder: str = "output",
                           detection_model: str = "yolo-v9-t-384-license-plate-end2end",
                           confidence_threshold: float = 0.5) -> dict:
    """
    Convenience function to process all images in a folder
    
    Args:
        input_folder: Path to folder containing images
        output_folder: Path to output folder
        detection_model: Model name for detection
        confidence_threshold: Minimum confidence score
        
    Returns:
        Dictionary with processing results
    """
    processor = FastANPRProcessor(detection_model)
    return processor.scan_folder_and_process(input_folder, output_folder, confidence_threshold)


def ocr_cropped_images(cropped_image_paths: List[str], ocr_model: str = "cct-xs-v1-global-model") -> List[dict]:
    """
    Convenience function to perform OCR on cropped license plate images
    
    Args:
        cropped_image_paths: List of paths to cropped license plate images
        ocr_model: Model name for OCR recognition
        
    Returns:
        List of OCR result dictionaries
    """
    processor = FastANPRProcessor(ocr_model=ocr_model)
    return processor.ocr_cropped_images(cropped_image_paths)


def ocr_single_image(image_path: str, ocr_model: str = "cct-xs-v1-global-model") -> dict:
    """
    Convenience function to perform OCR on a single cropped license plate image
    
    Args:
        image_path: Path to the cropped license plate image
        ocr_model: Model name for OCR recognition
        
    Returns:
        OCR result dictionary
    """
    processor = FastANPRProcessor(ocr_model=ocr_model)
    return processor.ocr_single_image(image_path)


def create_label_file_from_cropped_folder(cropped_folder: str, 
                                         output_file: str = "output/results/label.txt",
                                         ocr_model: str = "cct-xs-v1-global-model") -> None:
    """
    Create a label.txt file from a folder of cropped license plate images
    
    Args:
        cropped_folder: Path to folder containing cropped license plate images
        output_file: Path where to save the label.txt file
        ocr_model: Model name for OCR recognition
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Find all image files in cropped folder
        image_files = []
        if os.path.exists(cropped_folder):
            for file in os.listdir(cropped_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                    image_files.append(os.path.join(cropped_folder, file))
        
        if not image_files:
            print(f"No image files found in {cropped_folder}")
            return
        
        print(f"Found {len(image_files)} cropped images, performing OCR...")
        
        # Perform OCR on all images
        ocr_results = ocr_cropped_images(image_files, ocr_model)
        
        # Create label file
        with open(output_file, 'w', encoding='utf-8') as f:
            label_count = 0
            
            for result in ocr_results:
                cropped_path = result.get('image_path', '')
                if cropped_path:
                    cropped_filename = os.path.basename(cropped_path)
                    ocr_text = result.get('text', '').strip()
                    
                    # Clean OCR text: remove square brackets and underscores
                    cleaned_text = _clean_ocr_text(ocr_text)
                    
                    # Write filename and OCR text separated by tab
                    f.write(f"{cropped_filename}\t{cleaned_text}\n")
                    label_count += 1
        
        print(f"Created label file '{output_file}' with {label_count} entries")
        
    except Exception as e:
        print(f"Error creating label file from cropped folder: {str(e)}")


def create_label_file_from_ocr_results(ocr_results: List[dict], output_file: str = "output/results/label.txt") -> None:
    """
    Create a label.txt file from existing OCR results
    
    Args:
        ocr_results: List of OCR result dictionaries
        output_file: Path where to save the label.txt file
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            label_count = 0
            
            for result in ocr_results:
                cropped_path = result.get('image_path', '')
                if cropped_path:
                    cropped_filename = os.path.basename(cropped_path)
                    ocr_text = result.get('text', '').strip()
                    
                    # Clean OCR text: remove square brackets and underscores
                    cleaned_text = _clean_ocr_text(ocr_text)
                    
                    # Write filename and OCR text separated by tab
                    f.write(f"{cropped_filename}\t{cleaned_text}\n")
                    label_count += 1
        
        print(f"Created label file '{output_file}' with {label_count} entries")
        
    except Exception as e:
        print(f"Error creating label file from OCR results: {str(e)}")


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
