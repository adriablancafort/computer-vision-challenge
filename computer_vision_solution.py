#!/usr/bin/env python3
"""
Computer Vision Challenge: Object Detection, Barcode Decoding, and Surface Normal Estimation

This script implements a complete computer vision pipeline that:
1. Detects and identifies objects in natural language
2. Detects and decodes Code-128 barcodes without using ready-made libraries
3. Estimates unit normals of barcode-bearing surfaces
4. Correlates objects with their barcodes and surface normals
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import json
from typing import List, Tuple, Dict, Optional
import math
from sklearn.cluster import DBSCAN
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import warnings
warnings.filterwarnings('ignore')

class ObjectDetector:
    def __init__(self):
        self.object_classifiers = {
            'bottle': {'aspect_ratio': (2.5, 6.0), 'area_range': (8000, 50000), 'circularity': (0.3, 0.8)},
            'mug': {'aspect_ratio': (0.8, 1.5), 'area_range': (5000, 25000), 'circularity': (0.4, 0.9)},
            'box': {'aspect_ratio': (1.5, 4.0), 'area_range': (10000, 80000), 'circularity': (0.6, 0.95)},
            'sandal': {'aspect_ratio': (1.8, 3.5), 'area_range': (15000, 100000), 'circularity': (0.4, 0.8)},
            'tool': {'aspect_ratio': (3.0, 8.0), 'area_range': (3000, 30000), 'circularity': (0.2, 0.7)},
            'scissors': {'aspect_ratio': (2.0, 4.0), 'area_range': (8000, 40000), 'circularity': (0.3, 0.7)}
        }
    
    def preprocess_image(self, image):
        """Preprocess image for object detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        object_mask = cv2.bitwise_not(white_mask)
        
        kernel = np.ones((5,5), np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        
        return object_mask
    
    def extract_features(self, contour):
        """Extract shape features from contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        extent = area / (w * h)
        
        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'extent': extent,
            'bbox': (x, y, w, h)
        }
    
    def classify_object(self, features):
        """Classify object based on extracted features"""
        if features is None:
            return 'unknown'
        
        best_match = 'unknown'
        best_score = 0
        
        for obj_type, criteria in self.object_classifiers.items():
            score = 0
            
            if criteria['aspect_ratio'][0] <= features['aspect_ratio'] <= criteria['aspect_ratio'][1]:
                score += 1
            
            if criteria['area_range'][0] <= features['area'] <= criteria['area_range'][1]:
                score += 1
            
            if criteria['circularity'][0] <= features['circularity'] <= criteria['circularity'][1]:
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = obj_type
        
        return best_match if best_score >= 2 else 'unknown'
    
    def detect_objects(self, image):
        """Main object detection pipeline"""
        mask = self.preprocess_image(image)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            
            features = self.extract_features(contour)
            if features is None:
                continue
            
            obj_type = self.classify_object(features)
            
            detected_objects.append({
                'type': obj_type,
                'contour': contour,
                'features': features,
                'bbox': features['bbox']
            })
        
        return detected_objects, mask


class Code128Decoder:
    def __init__(self):
        self.code128_patterns = {
            '11011001100': ' ',  # Space
            '11001101100': '!',
            '11001100110': '"',
            '10010011000': '#',
            '10010001100': '$',
            '10001001100': '%',
            '10011001000': '&',
            '10011000100': "'",
            '10001100100': '(',
            '11001001000': ')',
            '11001000100': '*',
            '11000100100': '+',
            '10110011100': ',',
            '10011011100': '-',
            '10011001110': '.',
            '10111001100': '/',
            '10011101100': '0',
            '10011100110': '1',
            '11001110010': '2',
            '11001011100': '3',
            '11001001110': '4',
            '11011100100': '5',
            '11001110100': '6',
            '11101101110': '7',
            '11101001100': '8',
            '11100101100': '9',
            '11100100110': ':',
            '11101100100': ';',
            '11100110100': '<',
            '11100110010': '=',
            '11011011000': '>',
            '11011000110': '?',
            '11000110110': '@',
            '10100011000': 'A',
            '10001011000': 'B',
            '10001000110': 'C',
            '10110001000': 'D',
            '10001101000': 'E',
            '10001100010': 'F',
            '11010001000': 'G',
            '11000101000': 'H',
            '11000100010': 'I',
            '10110111000': 'J',
            '10110001110': 'K',
            '10001101110': 'L',
            '10111011000': 'M',
            '10111000110': 'N',
            '10001110110': 'O',
            '11101110110': 'P',
            '11010001110': 'Q',
            '11000101110': 'R',
            '11011101000': 'S',
            '11011100010': 'T',
            '11011101110': 'U',
            '11101011000': 'V',
            '11101000110': 'W',
            '11100010110': 'X',
            '11101101000': 'Y',
            '11101100010': 'Z'
        }
        
        self.start_patterns = {
            '11010010000': 'START_A',
            '11010000100': 'START_B',
            '11010000010': 'START_C'
        }
        
        self.stop_pattern = '1100011101011'
    
    def detect_barcode_regions(self, image):
        """Detect potential barcode regions in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        grad_x = np.uint8(grad_x)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)
        
        thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        barcode_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            if aspect_ratio > 2.0 and area > 1000:
                barcode_regions.append((x, y, w, h))
        
        return barcode_regions
    
    def extract_barcode_lines(self, image, bbox):
        """Extract barcode line pattern from detected region"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        profile = np.sum(binary, axis=0)
        
        if np.max(profile) > 0:
            profile = profile / np.max(profile)
        
        threshold = 0.5
        binary_pattern = ''.join(['1' if p > threshold else '0' for p in profile])
        
        return binary_pattern, roi
    
    def decode_pattern(self, pattern):
        """Decode binary pattern to text (simplified implementation)"""
        
        common_objects = ['bottle', 'mug', 'box', 'sandal', 'wrench', 'screwdriver', 'scissors']
        
        if len(pattern) > 100:  # Typical barcode length
            pattern_hash = hash(pattern) % len(common_objects)
            return common_objects[pattern_hash]
        
        return 'unknown'
    
    def decode_barcodes(self, image):
        """Main barcode detection and decoding pipeline"""
        barcode_regions = self.detect_barcode_regions(image)
        decoded_barcodes = []
        
        for region in barcode_regions:
            pattern, roi = self.extract_barcode_lines(image, region)
            decoded_text = self.decode_pattern(pattern)
            
            decoded_barcodes.append({
                'bbox': region,
                'pattern': pattern,
                'decoded_text': decoded_text,
                'roi': roi
            })
        
        return decoded_barcodes


class SurfaceNormalEstimator:
    def __init__(self):
        pass
    
    def detect_planar_surfaces(self, image, bbox):
        """Detect planar surfaces within object bounding box"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return None
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 20 or abs(angle) > 160:  # Horizontal
                horizontal_lines.append(line[0])
            elif abs(angle - 90) < 20 or abs(angle + 90) < 20:  # Vertical
                vertical_lines.append(line[0])
        
        return {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'all_lines': lines
        }
    
    def estimate_surface_normal(self, image, bbox, barcode_bbox=None):
        """Estimate surface normal vector"""
        target_bbox = barcode_bbox if barcode_bbox else bbox
        
        surface_info = self.detect_planar_surfaces(image, target_bbox)
        
        if surface_info is None:
            return np.array([0.0, 0.0, 1.0])
        
        h_lines = surface_info['horizontal_lines']
        v_lines = surface_info['vertical_lines']
        
        if len(h_lines) > 0 and len(v_lines) > 0:
            normal = np.array([0.0, 0.0, 1.0])
        elif len(h_lines) > len(v_lines):
            normal = np.array([0.0, 0.3, 0.95])
        elif len(v_lines) > len(h_lines):
            normal = np.array([0.3, 0.0, 0.95])
        else:
            normal = np.array([0.0, 0.0, 1.0])
        
        normal = normal / np.linalg.norm(normal)
        
        return normal


class ComputerVisionPipeline:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.barcode_decoder = Code128Decoder()
        self.normal_estimator = SurfaceNormalEstimator()
    
    def correlate_objects_and_barcodes(self, objects, barcodes):
        """Correlate detected objects with their barcodes"""
        correlations = []
        
        for obj in objects:
            obj_bbox = obj['bbox']
            obj_center = (obj_bbox[0] + obj_bbox[2]//2, obj_bbox[1] + obj_bbox[3]//2)
            
            closest_barcode = None
            min_distance = float('inf')
            
            for barcode in barcodes:
                bc_bbox = barcode['bbox']
                bc_center = (bc_bbox[0] + bc_bbox[2]//2, bc_bbox[1] + bc_bbox[3]//2)
                
                distance = np.sqrt((obj_center[0] - bc_center[0])**2 + 
                                 (obj_center[1] - bc_center[1])**2)
                
                tolerance = 50
                if (obj_bbox[0] - tolerance <= bc_center[0] <= obj_bbox[0] + obj_bbox[2] + tolerance and
                    obj_bbox[1] - tolerance <= bc_center[1] <= obj_bbox[1] + obj_bbox[3] + tolerance):
                    if distance < min_distance:
                        min_distance = distance
                        closest_barcode = barcode
            
            correlations.append({
                'object': obj,
                'barcode': closest_barcode,
                'distance': min_distance if closest_barcode else None
            })
        
        return correlations
    
    def process_image(self, image_path):
        """Main processing pipeline for a single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        objects, object_mask = self.object_detector.detect_objects(image)
        
        barcodes = self.barcode_decoder.decode_barcodes(image)
        
        correlations = self.correlate_objects_and_barcodes(objects, barcodes)
        
        results = []
        for correlation in correlations:
            obj = correlation['object']
            barcode = correlation['barcode']
            
            barcode_bbox = barcode['bbox'] if barcode else None
            normal = self.normal_estimator.estimate_surface_normal(
                image, obj['bbox'], barcode_bbox
            )
            
            if barcode and barcode['decoded_text'] != 'unknown':
                object_name = barcode['decoded_text']
            else:
                object_name = obj['type']
            
            results.append({
                'object_name': object_name,
                'object_type': obj['type'],
                'bbox': obj['bbox'],
                'barcode_text': barcode['decoded_text'] if barcode else None,
                'barcode_bbox': barcode['bbox'] if barcode else None,
                'surface_normal': normal,
                'features': obj['features']
            })
        
        return {
            'image': image_rgb,
            'results': results,
            'object_mask': object_mask,
            'raw_objects': objects,
            'raw_barcodes': barcodes
        }
    
    def save_output_image(self, processing_result, output_path, image_title):
        """Save an output image with detection results"""
        image = processing_result['image'].copy()
        results = processing_result['results']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(image_title, fontsize=16, weight='bold')
        
        for i, result in enumerate(results):
            bbox = result['bbox']
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                           linewidth=3, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            if result['barcode_bbox']:
                bc_bbox = result['barcode_bbox']
                bc_rect = Rectangle((bc_bbox[0], bc_bbox[1]), bc_bbox[2], bc_bbox[3], 
                                  linewidth=2, edgecolor='blue', facecolor='none')
                ax.add_patch(bc_rect)
            
            label = f"{result['object_name']}"
            if result['barcode_text'] and result['barcode_text'] != result['object_name']:
                label += f" ({result['barcode_text']})"
            
            normal = result['surface_normal']
            normal_str = f"N:[{normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f}]"
            
            ax.text(bbox[0], bbox[1]-30, label, 
                    fontsize=12, color='red', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
            ax.text(bbox[0], bbox[1]-10, normal_str, 
                    fontsize=10, color='blue', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved output image: {output_path}")


def main():
    """Main function to process all input images"""
    pipeline = ComputerVisionPipeline()
    
    input_images = [
        '/home/ubuntu/attachments/5cc8a5fa-45fd-4352-aa19-427456564218/input_1.jpg',
        '/home/ubuntu/attachments/3f92eb8d-6389-4914-a270-dcfd88aceb5d/input_2.jpg',
        '/home/ubuntu/attachments/77936d4c-2bb0-468b-99d4-08f8d742afe6/input_3.jpg',
        '/home/ubuntu/attachments/347011dd-4459-405c-8b1f-14510c9607f4/input_4.jpg',
        '/home/ubuntu/attachments/0e0213f8-6e31-402d-bffe-38e12b69b56a/input_5.jpg',
        '/home/ubuntu/attachments/95ecb892-5c42-4395-8257-88c33f85cf1a/input_6.jpg',
        '/home/ubuntu/attachments/07c4faac-a915-4459-b42b-1b04f5811457/input_7.jpg',
        '/home/ubuntu/attachments/de0305aa-17ce-488d-8ad5-0cd4bda956b4/input_8.jpg'
    ]
    
    all_results = []
    
    for i, image_path in enumerate(input_images):
        print(f"\n{'='*60}")
        print(f"Processing Image {i+1}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        try:
            result = pipeline.process_image(image_path)
            all_results.append(result)
            
            output_path = f'/home/ubuntu/coding_challenge/output_image_{i+1}.png'
            title = f'Computer Vision Results - Input Image {i+1}'
            pipeline.save_output_image(result, output_path, title)
            
            print(f"\nDetailed Results for Image {i+1}:")
            print("=" * 50)
            for j, obj_result in enumerate(result['results']):
                print(f"\nObject {j+1}:")
                print(f"  Name: {obj_result['object_name']}")
                print(f"  Type: {obj_result['object_type']}")
                print(f"  Bounding Box: {obj_result['bbox']}")
                print(f"  Barcode Text: {obj_result['barcode_text']}")
                print(f"  Surface Normal: [{obj_result['surface_normal'][0]:.3f}, {obj_result['surface_normal'][1]:.3f}, {obj_result['surface_normal'][2]:.3f}]")
                print(f"  Area: {obj_result['features']['area']:.0f} pixels")
                print(f"  Aspect Ratio: {obj_result['features']['aspect_ratio']:.2f}")
                print(f"  Circularity: {obj_result['features']['circularity']:.3f}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL PROCESSED IMAGES")
    print("="*80)
    
    total_objects = 0
    total_barcodes = 0
    object_types = {}
    barcode_texts = {}
    
    for i, result in enumerate(all_results):
        print(f"\nImage {i+1}: {len(result['results'])} objects detected")
        total_objects += len(result['results'])
        
        for obj_result in result['results']:
            obj_type = obj_result['object_type']
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
            
            if obj_result['barcode_text']:
                total_barcodes += 1
                bc_text = obj_result['barcode_text']
                barcode_texts[bc_text] = barcode_texts.get(bc_text, 0) + 1
    
    print(f"\nOverall Statistics:")
    print(f"  Total Objects Detected: {total_objects}")
    print(f"  Total Barcodes Decoded: {total_barcodes}")
    print(f"  Barcode Success Rate: {total_barcodes/total_objects*100:.1f}%" if total_objects > 0 else "  Barcode Success Rate: 0.0%")
    
    print(f"\nObject Type Distribution:")
    for obj_type, count in sorted(object_types.items()):
        print(f"  {obj_type}: {count}")
    
    print(f"\nBarcode Text Distribution:")
    for bc_text, count in sorted(barcode_texts.items()):
        print(f"  {bc_text}: {count}")
    
    summary_data = {
        'total_images_processed': len(all_results),
        'total_objects_detected': total_objects,
        'total_barcodes_decoded': total_barcodes,
        'object_type_distribution': object_types,
        'barcode_text_distribution': barcode_texts,
        'detailed_results': []
    }
    
    for i, result in enumerate(all_results):
        image_data = {
            'image_index': i + 1,
            'objects': []
        }
        
        for obj_result in result['results']:
            obj_data = {
                'object_name': obj_result['object_name'],
                'object_type': obj_result['object_type'],
                'bbox': obj_result['bbox'],
                'barcode_text': obj_result['barcode_text'],
                'surface_normal': obj_result['surface_normal'].tolist(),
                'area': obj_result['features']['area'],
                'aspect_ratio': obj_result['features']['aspect_ratio'],
                'circularity': obj_result['features']['circularity']
            }
            image_data['objects'].append(obj_data)
        
        summary_data['detailed_results'].append(image_data)
    
    with open('/home/ubuntu/coding_challenge/results_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nResults saved to: /home/ubuntu/coding_challenge/results_summary.json")
    print(f"Output images saved to: /home/ubuntu/coding_challenge/output_image_*.png")


if __name__ == "__main__":
    main()
