"""
ASL Alphabet Detection System
Real-time Sign Language Recognition using MediaPipe and Machine Learning
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3
from collections import deque
import os

class ASLDetector:
    def __init__(self, model_path='model/asl_model.pkl'):
        """Initialize the ASL Detection System"""
        # MediaPipe setup
        self.mpHands = mp.solutions.hands
        try:
            self.hands = self.mpHands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        except Exception as e:
            print(f"Error initializing MediaPipe Hands: {e}")
            raise
        
        self.mpDraw = mp.solutions.drawing_utils
        
        # Load trained model
        self.model = None
        self.label_encoder = None
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("Warning: Model not found. Please train the model first.")
        
        # Text-to-speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        except Exception as e:
            print(f"Warning: TTS initialization failed: {e}")
            self.tts_engine = None
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=10)
        
        # Word formation
        self.detected_letters = []
        self.last_letter = None
        self.letter_stable_count = 0
        self.stability_threshold = 15
        
        # UI settings
        self.colors = {
            'primary': (0, 255, 0),
            'secondary': (255, 0, 255),
            'text': (255, 255, 255),
            'background': (50, 50, 50),
            'warning': (0, 165, 255)
        }
        
        # Performance tracking
        self.pTime = 0
        
    def load_model(self, model_path):
        """Load the trained ML model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract and normalize hand landmarks"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        # Normalize landmarks relative to wrist (landmark 0)
        landmarks = np.array(landmarks).reshape(-1, 3)
        wrist = landmarks[0]
        landmarks = landmarks - wrist
        
        # Calculate hand size for scale normalization
        hand_size = np.max(np.linalg.norm(landmarks, axis=1))
        if hand_size > 0:
            landmarks = landmarks / hand_size
        
        return landmarks.flatten()
    
    def predict_letter(self, landmarks):
        """Predict ASL letter from landmarks"""
        if self.model is None:
            return None, 0.0
        
        try:
            # Reshape for prediction
            landmarks = landmarks.reshape(1, -1)
            
            # Get prediction and confidence
            prediction = self.model.predict(landmarks)
            
            # Get confidence (probability)
            if hasattr(self.model, 'predict_proba'):
                confidence = np.max(self.model.predict_proba(landmarks))
            else:
                confidence = 1.0
            
            # Decode label
            letter = self.label_encoder.inverse_transform(prediction)[0]
            
            return letter, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0
    
    def smooth_prediction(self, letter):
        """Smooth predictions using a buffer"""
        if letter:
            self.prediction_buffer.append(letter)
            
        if len(self.prediction_buffer) >= 5:
            # Get most common letter in buffer
            unique, counts = np.unique(list(self.prediction_buffer), return_counts=True)
            most_common_idx = np.argmax(counts)
            return unique[most_common_idx]
        
        return letter
    
    def update_word_formation(self, letter, confidence):
        """Update detected word with stable letters"""
        if letter and confidence > 0.75:
            if letter == self.last_letter:
                self.letter_stable_count += 1
                
                if self.letter_stable_count == self.stability_threshold:
                    self.detected_letters.append(letter)
                    # Speak the letter
                    self.speak_letter(letter)
            else:
                self.last_letter = letter
                self.letter_stable_count = 0
    
    def speak_letter(self, letter):
        """Speak detected letter using TTS"""
        if self.tts_engine is None:
            return
        try:
            if not self.tts_engine._inLoop:
                self.tts_engine.say(letter)
                self.tts_engine.runAndWait()
        except Exception as e:
            pass
    
    def speak_word(self):
        """Speak the complete word"""
        if self.tts_engine is None:
            return
        if self.detected_letters:
            word = ''.join(self.detected_letters)
            try:
                self.tts_engine.say(word)
                self.tts_engine.runAndWait()
            except Exception as e:
                pass
    
    def draw_ui(self, img, letter, confidence, fps, hand_detected=False):
        """Draw enhanced UI elements"""
        h, w, _ = img.shape
        
        # Semi-transparent overlay for top panel
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # FPS counter
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['primary'], 2)
        
        # Check if model is loaded
        if self.model is None:
            cv2.putText(img, 'NO MODEL - Train model first!', (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(img, '1. Run: collect_data.py', (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, '2. Run: train_model.py', (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif not hand_detected:
            # Show "waiting for hand" message
            cv2.putText(img, 'Show your hand...', (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['warning'], 3)
        elif letter and confidence > 0.6:
            # Detected letter - MUCH LARGER display
            color = self.colors['primary'] if confidence > 0.75 else self.colors['warning']
            
            # Large letter display in center-left
            cv2.putText(img, letter, (50, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 8)
            
            # Label above
            cv2.putText(img, 'Detected:', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
            
            # Confidence bar
            bar_width = int(200 * confidence)
            cv2.rectangle(img, (200, 100), (400, 130), (100, 100, 100), 2)
            cv2.rectangle(img, (200, 100), (200 + bar_width, 130), color, -1)
            cv2.putText(img, f'{int(confidence * 100)}%', (410, 125), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        else:
            # Hand detected but no confident prediction
            cv2.putText(img, 'Hold gesture steady...', (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
        
        # Word formation display
        if self.detected_letters:
            word = ''.join(self.detected_letters)
            cv2.putText(img, f'Word: {word}', (w - 400, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['secondary'], 2)
        
        # Instructions
        instructions = [
            "Press 'SPACE' to add space",
            "Press 'BACKSPACE' to delete letter",
            "Press 'ENTER' to speak word",
            "Press 'C' to clear word",
            "Press 'Q' to quit"
        ]
        
        y_offset = h - 150
        for instruction in instructions:
            cv2.putText(img, instruction, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += 30
        
        return img
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print("\n" + "="*60)
        print("ASL Alphabet Detection System Started")
        print("="*60)
        print("\nControls:")
        print("  SPACE      - Add space to word")
        print("  BACKSPACE  - Delete last letter")
        print("  ENTER      - Speak complete word")
        print("  C          - Clear word")
        print("  Q          - Quit application")
        print("\n" + "="*60 + "\n")
        
        while True:
            success, img = cap.read()
            if not success or img is None:
                print("Error: Could not read frame from camera")
                break
            
            # Validate image
            if img.size == 0:
                continue
            
            # Flip for mirror effect
            img = cv2.flip(img, 1)
            
            # Make image writable and convert to RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB.flags.writeable = False  # Improve performance
            
            # Process hand detection with error handling
            try:
                results = self.hands.process(imgRGB)
            except Exception as e:
                print(f"Warning: Hand detection error: {e}")
                results = None
            
            # Make image writable again
            imgRGB.flags.writeable = True
            
            letter = None
            confidence = 0.0
            
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mpDraw.draw_landmarks(
                        img, hand_landmarks, 
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )
                    
                    # Extract landmarks and predict
                    landmarks = self.extract_landmarks(hand_landmarks)
                    letter, confidence = self.predict_letter(landmarks)
                    
                    # Smooth prediction
                    letter = self.smooth_prediction(letter)
                    
                    # Update word formation
                    self.update_word_formation(letter, confidence)
            
            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - self.pTime) if self.pTime > 0 else 0
            self.pTime = cTime
            
            # Draw UI
            hand_detected = results and results.multi_hand_landmarks is not None
            img = self.draw_ui(img, letter, confidence, fps, hand_detected)
            
            # Display
            cv2.imshow("ASL Alphabet Detection", img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space
                self.detected_letters.append(' ')
            elif key == 8:  # Backspace
                if self.detected_letters:
                    self.detected_letters.pop()
            elif key == 13:  # Enter
                self.speak_word()
            elif key == ord('c'):  # Clear
                self.detected_letters = []
        
        # Cleanup
        try:
            cap.release()
        except:
            pass
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        try:
            if self.hands:
                self.hands.close()
        except:
            pass
        
        print("\nASL Detection System Closed")


if __name__ == "__main__":
    detector = ASLDetector()
    detector.run()