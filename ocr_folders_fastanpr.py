import cv2
import os
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from open_image_models import LicensePlateDetector
from fast_plate_ocr import LicensePlateRecognizer

class FastANPRProcessor:
    """Fast ANPR processor using open-image-models library"""
    
    def __init__(self, 
                 detection_model: str = "yolo-v9-t-384-license-plate-end2end",
                 ocr_model: str = "cct-xs-v1-global-model"):
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
        print(f"Initializing FastANPR with detection model: {detection_model}")
        self.lp_detector = LicensePlateDetector(detection_model=detection_model)
        
        print(f"Initializing OCR with model: {ocr_model}")
        self.ocr_recognizer = LicensePlateRecognizer(ocr_model)
        print("FastANPR initialized successfully!")

    def detect_plate_from_image(self, image_path: str, confidence_threshold: float = 0.5) -> List[dict]:
        """
        Detect license plates from an image file using open-image-models
        
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
            
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image: {image_path}")
                return []
            
            print(f"Processing image: {image_path}")
            print(f"Image shape: {image.shape}")
            
            # Perform license plate detection
            detections = self.lp_detector.predict(image)
            
            # Process detections and filter by confidence
            processed_detections = []
            for detection in detections:
                # Extract bounding box coordinates and confidence
                x1, y1, x2, y2 = detection.bounding_box
                confidence = detection.confidence
                
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
        Perform OCR on cropped license plate images using fast-plate-ocr
        
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
                    
                    # Perform OCR using fast-plate-ocr
                    ocr_result = self.ocr_recognizer.run(crop_path)
                    
                    # Extract text and confidence if available
                    if isinstance(ocr_result, dict):
                        text = ocr_result.get('text', '')
                        confidence = ocr_result.get('confidence', 0.0)
                    elif isinstance(ocr_result, str):
                        text = ocr_result
                        confidence = 1.0  # Default confidence if not provided
                    else:
                        text = str(ocr_result)
                        confidence = 1.0
                    
                    result = {
                        'image_path': crop_path,
                        'text': text,
                        'confidence': confidence
                    }
                    
                    ocr_results.append(result)
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
            
            # Perform OCR using fast-plate-ocr
            ocr_result = self.ocr_recognizer.run(image_path)
            
            # Extract text and confidence if available
            if isinstance(ocr_result, dict):
                text = ocr_result.get('text', '')
                confidence = ocr_result.get('confidence', 0.0)
            elif isinstance(ocr_result, str):
                text = ocr_result
                confidence = 1.0  # Default confidence if not provided
            else:
                text = str(ocr_result)
                confidence = 1.0
            
            result = {
                'image_path': image_path,
                'text': text,
                'confidence': confidence
            }
            
            print(f"OCR result: '{text}' (confidence: {confidence:.3f})")
            return result
            
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


# Helper function for cleaning OCR text (used by standalone functions)
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
    cleaned = text.replace('[', '').replace(']', '').replace('_', '')
    
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


# Example usage
if __name__ == "__main__":
    # Initialize processor with both detection and OCR models
    processor = FastANPRProcessor(
        detection_model="yolo-v9-s-608-license-plate-end2end",
        ocr_model="cct-s-v1-global-model"
    )
    
    # Example 1: Process a single image with OCR
    # print("=== Example 1: Single Image Processing with OCR ===")
    # image_path = "images/test/image001.jpg"  # Update with your image path
    # if os.path.exists(image_path):
    #     # Step 1: Detect plates
    #     detections = processor.detect_plate_from_image(image_path)
    #     if detections:
    #         # Step 2: Crop plates
    #         cropped_paths = processor.crop_image(image_path, detections)
    #         print(f"Cropped images saved: {cropped_paths}")
            
    #         # Step 3: Perform OCR
    #         if cropped_paths:
    #             ocr_results = processor.ocr_cropped_images(cropped_paths)
    #             for result in ocr_results:
    #                 print(f"OCR: '{result['text']}' from {result['image_path']}")
    
    # Example 2: Process entire folder with full pipeline (detection + cropping + OCR)
    print("\n=== Example 2: Complete Folder Processing with OCR ===")
    input_folder = "images\\test"  # Update with your folder path
    if os.path.exists(input_folder):
        results = processor.scan_folder_and_process(input_folder, "output")
        print(f"Processing complete. Check 'output' folder for results.")
        
        # Print summary of OCR results
        # for result in results.get('results', []):
        #     if result['status'] == 'success' and result['ocr_results']:
        #         print(f"\nImage: {os.path.basename(result['image_path'])}")
        #         for ocr in result['ocr_results']:
        #             text = ocr.get('text', 'N/A')
        #             conf = ocr.get('confidence', 0.0)
        #             print(f"  License Plate: '{text}' (confidence: {conf:.3f})")
        
        # Note: label.txt file is automatically created in output/results/
    
    # Example 3: OCR on existing cropped images
    # print("\n=== Example 3: OCR on Existing Cropped Images ===")
    # cropped_folder = "output/cropped"
    # if os.path.exists(cropped_folder):
    #     cropped_files = [os.path.join(cropped_folder, f) for f in os.listdir(cropped_folder) 
    #                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    #     if cropped_files:
    #         ocr_results = ocr_cropped_images(cropped_files[:3])  # Test first 3 images
    #         for result in ocr_results:
    #             print(f"OCR Result: '{result['text']}' from {os.path.basename(result['image_path'])}")
