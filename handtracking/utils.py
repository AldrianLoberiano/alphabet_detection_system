"""
Utility Functions for ASL Detection System
Helper functions for visualization, data processing, and analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

class LandmarkVisualizer:
    """Visualize hand landmarks and connections"""
    
    @staticmethod
    def draw_hand_skeleton(img, hand_landmarks, mpHands, mpDraw, color=(0, 255, 0)):
        """Draw hand landmarks with custom styling"""
        mpDraw.draw_landmarks(
            img, 
            hand_landmarks, 
            mpHands.HAND_CONNECTIONS,
            mpDraw.DrawingSpec(color=color, thickness=2, circle_radius=2),
            mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2)
        )
        return img
    
    @staticmethod
    def draw_finger_tips(img, hand_landmarks, finger_tip_ids=[4, 8, 12, 16, 20]):
        """Highlight finger tips"""
        h, w, _ = img.shape
        for id in finger_tip_ids:
            lm = hand_landmarks.landmark[id]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
        return img
    
    @staticmethod
    def draw_bounding_box(img, hand_landmarks):
        """Draw bounding box around hand"""
        h, w, _ = img.shape
        
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
        y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
        
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
        
        return img


class DataAnalyzer:
    """Analyze collected data and model performance"""
    
    @staticmethod
    def analyze_dataset(data_path):
        """Analyze dataset statistics"""
        import pickle
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        X = data['X']
        y = data['y']
        
        print("\n" + "="*60)
        print("Dataset Analysis")
        print("="*60)
        print(f"\nTotal samples: {len(X)}")
        print(f"Feature dimensions: {X.shape[1]}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass Distribution:")
        for letter, count in sorted(zip(unique, counts)):
            percentage = (count / len(y)) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {letter}: {count:3d} samples {bar} {percentage:.1f}%")
        
        # Feature statistics
        print(f"\nFeature Statistics:")
        print(f"  Mean: {np.mean(X):.4f}")
        print(f"  Std: {np.std(X):.4f}")
        print(f"  Min: {np.min(X):.4f}")
        print(f"  Max: {np.max(X):.4f}")
        
        # Check for missing values
        nan_count = np.isnan(X).sum()
        print(f"\nMissing values: {nan_count}")
        
        print("="*60 + "\n")
        
        return {
            'total_samples': len(X),
            'features': X.shape[1],
            'classes': len(np.unique(y)),
            'class_distribution': dict(zip(unique, counts))
        }
    
    @staticmethod
    def plot_class_distribution(data_path, save_path='results/plots/class_distribution.png'):
        """Plot class distribution"""
        import pickle
        
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        y = data['y']
        unique, counts = np.unique(y, return_counts=True)
        
        plt.figure(figsize=(14, 6))
        plt.bar(unique, counts, color='steelblue', edgecolor='black')
        plt.xlabel('ASL Letter', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (letter, count) in enumerate(zip(unique, counts)):
            plt.text(i, count + 1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"Class distribution plot saved: {save_path}")


class PerformanceMonitor:
    """Monitor and log system performance"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_start = datetime.now()
        self.predictions = []
        
    def log_prediction(self, letter, confidence, actual=None):
        """Log a prediction"""
        self.predictions.append({
            'timestamp': datetime.now().isoformat(),
            'letter': letter,
            'confidence': confidence,
            'actual': actual
        })
    
    def save_session_log(self):
        """Save session log to file"""
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f'session_{timestamp}.json')
        
        session_data = {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'total_predictions': len(self.predictions),
            'predictions': self.predictions
        }
        
        with open(log_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session log saved: {log_file}")
    
    def get_session_stats(self):
        """Get session statistics"""
        if not self.predictions:
            return None
        
        confidences = [p['confidence'] for p in self.predictions]
        letters = [p['letter'] for p in self.predictions]
        
        unique_letters = list(set(letters))
        
        stats = {
            'total_predictions': len(self.predictions),
            'unique_letters': len(unique_letters),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'most_common_letter': max(set(letters), key=letters.count)
        }
        
        return stats


class UIHelper:
    """Helper functions for UI elements"""
    
    @staticmethod
    def draw_info_panel(img, title, items, position=(10, 30), bg_color=(50, 50, 50)):
        """Draw an information panel"""
        h, w, _ = img.shape
        
        # Calculate panel size
        panel_height = 30 + len(items) * 30
        panel_width = 400
        
        x, y = position
        
        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), bg_color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Draw title
        cv2.putText(img, title, (x + 10, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw items
        y_offset = y + 55
        for item in items:
            cv2.putText(img, item, (x + 20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += 30
        
        return img
    
    @staticmethod
    def draw_progress_bar(img, progress, position=(10, 100), width=300, height=20, 
                         color=(0, 255, 0), bg_color=(100, 100, 100)):
        """Draw a progress bar"""
        x, y = position
        
        # Background
        cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        # Progress
        progress_width = int(width * progress)
        cv2.rectangle(img, (x, y), (x + progress_width, y + height), color, -1)
        
        # Percentage text
        percentage = int(progress * 100)
        text = f"{percentage}%"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img
    
    @staticmethod
    def draw_countdown(img, countdown, color=(0, 0, 255)):
        """Draw countdown overlay"""
        h, w, _ = img.shape
        
        if countdown > 0:
            # Semi-transparent overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            
            # Countdown text
            text = str(countdown)
            font_scale = 10
            thickness = 20
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return img


def create_demo_gif(video_path, output_path='demo.gif', fps=10, duration=5):
    """Create demo GIF from video"""
    try:
        from PIL import Image
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = int(duration * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = max(1, total_frames // frame_count)
        
        count = 0
        while cap.isOpened() and len(frames) < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                frames.append(Image.fromarray(frame))
            
            count += 1
        
        cap.release()
        
        if frames:
            frames[0].save(output_path, save_all=True, append_images=frames[1:], 
                          duration=1000//fps, loop=0)
            print(f"Demo GIF created: {output_path}")
    except Exception as e:
        print(f"Error creating GIF: {e}")


def check_system_requirements():
    """Check if all required packages are installed"""
    requirements = [
        'cv2', 'mediapipe', 'numpy', 'sklearn', 
        'matplotlib', 'seaborn', 'pyttsx3', 'pickle'
    ]
    
    missing = []
    
    for package in requirements:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("\nMissing required packages:")
        for package in missing:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("  pip install -r requirements.txt")
        return False
    
    print("All required packages are installed ✓")
    return True


if __name__ == "__main__":
    # Test functions
    print("Utility Functions Module")
    print("This module provides helper functions for the ASL Detection System")
    
    # Check system requirements
    check_system_requirements()
