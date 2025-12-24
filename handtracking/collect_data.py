"""
Data Collection Module for ASL Alphabet Detection
Collects hand landmark data for training the ML model
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from datetime import datetime
import time

class ASLDataCollector:
    def __init__(self, data_dir='data/raw'):
        """Initialize data collection system"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # MediaPipe setup
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # ASL Alphabet letters
        self.letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.current_letter_idx = 0
        self.current_letter = self.letters[self.current_letter_idx]
        
        # Data storage
        self.collected_data = {letter: [] for letter in self.letters}
        self.samples_per_letter = 100
        self.current_samples = 0
        
        # Collection state
        self.collecting = False
        self.countdown = 0
        self.countdown_start = 0
        
    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        # Normalize landmarks relative to wrist
        landmarks = np.array(landmarks).reshape(-1, 3)
        wrist = landmarks[0]
        landmarks = landmarks - wrist
        
        # Scale normalization
        hand_size = np.max(np.linalg.norm(landmarks, axis=1))
        if hand_size > 0:
            landmarks = landmarks / hand_size
        
        return landmarks.flatten()
    
    def draw_ui(self, img):
        """Draw collection interface"""
        h, w, _ = img.shape
        
        # Background overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Current letter (large display)
        cv2.putText(img, f'Letter: {self.current_letter}', (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        
        # Progress
        progress_text = f'Samples: {self.current_samples}/{self.samples_per_letter}'
        cv2.putText(img, progress_text, (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Progress bar
        bar_length = 400
        progress = int((self.current_samples / self.samples_per_letter) * bar_length)
        cv2.rectangle(img, (20, 140), (20 + bar_length, 170), (100, 100, 100), 2)
        cv2.rectangle(img, (20, 140), (20 + progress, 170), (0, 255, 0), -1)
        
        # Letter progress display
        letter_progress = f'Letter {self.current_letter_idx + 1}/{len(self.letters)}'
        cv2.putText(img, letter_progress, (w - 200, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Countdown display
        if self.countdown > 0:
            countdown_text = f'Starting in: {self.countdown}'
            cv2.putText(img, countdown_text, (w//2 - 150, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        
        # Instructions
        instructions = [
            "Position your hand to show the letter",
            "Press 'S' to start collecting samples",
            "Press 'N' to skip to next letter",
            "Press 'P' to go to previous letter",
            "Press 'R' to reset current letter data",
            "Press 'Q' to quit and save"
        ]
        
        y_offset = h - 200
        for instruction in instructions:
            cv2.putText(img, instruction, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30
        
        # Collection status
        if self.collecting:
            status = "COLLECTING..."
            color = (0, 255, 0)
        else:
            status = "READY - Press 'S' to start"
            color = (0, 165, 255)
        
        cv2.putText(img, status, (w//2 - 200, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return img
    
    def next_letter(self):
        """Move to next letter"""
        if self.current_letter_idx < len(self.letters) - 1:
            self.current_letter_idx += 1
            self.current_letter = self.letters[self.current_letter_idx]
            self.current_samples = len(self.collected_data[self.current_letter])
            self.collecting = False
            print(f"\nMoved to letter: {self.current_letter}")
    
    def previous_letter(self):
        """Move to previous letter"""
        if self.current_letter_idx > 0:
            self.current_letter_idx -= 1
            self.current_letter = self.letters[self.current_letter_idx]
            self.current_samples = len(self.collected_data[self.current_letter])
            self.collecting = False
            print(f"\nMoved to letter: {self.current_letter}")
    
    def reset_current_letter(self):
        """Reset data for current letter"""
        self.collected_data[self.current_letter] = []
        self.current_samples = 0
        self.collecting = False
        print(f"\nReset data for letter: {self.current_letter}")
    
    def save_data(self):
        """Save collected data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f'asl_data_{timestamp}.pkl')
        
        # Prepare data
        X = []
        y = []
        
        for letter, samples in self.collected_data.items():
            for sample in samples:
                X.append(sample)
                y.append(letter)
        
        if len(X) > 0:
            data = {
                'X': np.array(X),
                'y': np.array(y),
                'letters': self.letters,
                'timestamp': timestamp
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"\n{'='*60}")
            print(f"Data saved successfully: {filename}")
            print(f"Total samples collected: {len(X)}")
            print(f"Letters with data: {len([l for l, s in self.collected_data.items() if len(s) > 0])}")
            print(f"{'='*60}")
            return True
        else:
            print("No data to save!")
            return False
    
    def run(self):
        """Main data collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*60)
        print("ASL Alphabet Data Collection")
        print("="*60)
        print(f"\nCollecting {self.samples_per_letter} samples per letter")
        print("Position your hand to show each letter clearly")
        print("\nControls:")
        print("  S - Start collecting samples")
        print("  N - Next letter")
        print("  P - Previous letter")
        print("  R - Reset current letter data")
        print("  Q - Quit and save data")
        print("\n" + "="*60 + "\n")
        
        while True:
            success, img = cap.read()
            if not success or img is None:
                print("Error: Could not read frame")
                break
            
            # Flip for mirror effect
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(imgRGB)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mpDraw.draw_landmarks(
                        img, hand_landmarks, 
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )
                    
                    # Collect data if in collection mode
                    if self.collecting and self.countdown == 0:
                        if self.current_samples < self.samples_per_letter:
                            landmarks = self.extract_landmarks(hand_landmarks)
                            self.collected_data[self.current_letter].append(landmarks)
                            self.current_samples += 1
                            
                            # Auto-advance to next letter when done
                            if self.current_samples >= self.samples_per_letter:
                                self.collecting = False
                                print(f"Completed collection for letter: {self.current_letter}")
                                
                                if self.current_letter_idx < len(self.letters) - 1:
                                    time.sleep(0.5)
                                    self.next_letter()
                                else:
                                    print("\nAll letters completed!")
            
            # Handle countdown
            if self.countdown > 0:
                elapsed = time.time() - self.countdown_start
                if elapsed >= 1.0:
                    self.countdown -= 1
                    self.countdown_start = time.time()
            
            # Draw UI
            img = self.draw_ui(img)
            
            # Display
            cv2.imshow("ASL Data Collection", img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if not self.collecting and self.current_samples < self.samples_per_letter:
                    self.countdown = 3
                    self.countdown_start = time.time()
                    self.collecting = True
                    print(f"Starting collection for letter: {self.current_letter}")
            elif key == ord('n'):
                self.next_letter()
            elif key == ord('p'):
                self.previous_letter()
            elif key == ord('r'):
                self.reset_current_letter()
        
        # Cleanup and save
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Save collected data
        self.save_data()
        print("\nData Collection Closed")


if __name__ == "__main__":
    collector = ASLDataCollector()
    collector.run()
