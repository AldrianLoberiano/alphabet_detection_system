"""
Diagnostic Tool for ASL Detection System
Tests camera, MediaPipe, and other components
"""

import sys

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python 3.8+ required")
        return False
    print("  ✓ Python version OK")
    return True

def check_packages():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'pyttsx3': 'pyttsx3'
    }
    
    all_ok = True
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_camera():
    """Test camera access"""
    print("\nChecking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("  ❌ Cannot open camera (device 0)")
            print("  Try: Check if camera is in use by another app")
            print("       Check camera permissions")
            print("       Try different camera index (1, 2, etc.)")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            print("  ❌ Cannot read from camera")
            return False
        
        print(f"  ✓ Camera working (resolution: {frame.shape[1]}x{frame.shape[0]})")
        return True
        
    except Exception as e:
        print(f"  ❌ Camera error: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe Hands"""
    print("\nTesting MediaPipe Hands...")
    try:
        import cv2
        import mediapipe as mp
        
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Create a test image
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_image_rgb.flags.writeable = False
        
        # Try processing
        results = hands.process(test_image_rgb)
        hands.close()
        
        print("  ✓ MediaPipe Hands initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ MediaPipe error: {e}")
        return False

def test_tts():
    """Test Text-to-Speech"""
    print("\nTesting Text-to-Speech...")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        print("  ✓ TTS engine initialized")
        
        # Try to speak (might fail on some systems)
        try:
            engine.say("Test")
            engine.runAndWait()
            print("  ✓ TTS working")
        except:
            print("  ⚠ TTS initialized but speech failed")
            print("    This is OK - TTS will be disabled")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ TTS error: {e}")
        print("    TTS will be disabled - system will still work")
        return True  # Non-critical

def check_model():
    """Check if trained model exists"""
    print("\nChecking trained model...")
    import os
    
    model_path = 'model/asl_model.pkl'
    if os.path.exists(model_path):
        print(f"  ✓ Model found: {model_path}")
        return True
    else:
        print(f"  ⚠ No trained model found")
        print("    Run: python handtracking/collect_data.py")
        print("    Then: python handtracking/train_model.py")
        return False

def check_data():
    """Check if training data exists"""
    print("\nChecking training data...")
    import os
    
    data_dir = 'data/raw'
    if os.path.exists(data_dir):
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        if data_files:
            print(f"  ✓ Found {len(data_files)} data file(s)")
            return True
    
    print("  ⚠ No training data found")
    print("    Run: python handtracking/collect_data.py")
    return False

def test_live_detection():
    """Test live camera with MediaPipe"""
    print("\nTesting live detection (5 seconds)...")
    print("  Position your hand in front of the camera")
    
    try:
        import cv2
        import mediapipe as mp
        import time
        
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        mpDraw = mp.solutions.drawing_utils
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        start_time = time.time()
        hand_detected = False
        frame_count = 0
        
        while time.time() - start_time < 5:
            success, img = cap.read()
            if not success:
                continue
            
            frame_count += 1
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB.flags.writeable = False
            
            try:
                results = hands.process(imgRGB)
                imgRGB.flags.writeable = True
                
                if results and results.multi_hand_landmarks:
                    hand_detected = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    
                    cv2.putText(img, "Hand Detected!", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "No hand detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                
                cv2.putText(img, "Press 'Q' to skip", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                cv2.imshow("Live Detection Test", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"    Error during processing: {e}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        fps = frame_count / 5 if frame_count > 0 else 0
        print(f"  ✓ Processed {frame_count} frames (~{fps:.1f} FPS)")
        
        if hand_detected:
            print("  ✓ Hand detection working")
        else:
            print("  ⚠ No hand detected (this is OK if no hand was shown)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Live detection error: {e}")
        return False

def main():
    print("="*60)
    print("ASL Detection System - Diagnostic Tool")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Packages': check_packages(),
        'Camera': check_camera(),
        'MediaPipe': test_mediapipe(),
        'TTS': test_tts(),
        'Training Data': check_data(),
        'Trained Model': check_model(),
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    critical_passed = all([
        results['Python Version'],
        results['Packages'],
        results['Camera'],
        results['MediaPipe']
    ])
    
    for test, passed in results.items():
        status = "✓" if passed else ("⚠" if test in ['TTS', 'Training Data', 'Trained Model'] else "❌")
        print(f"  {status} {test}")
    
    print("\n" + "="*60)
    
    if critical_passed:
        print("✓ Critical components working!")
        
        if results['Trained Model']:
            print("\nYou can run:")
            print("  python handtracking/handtracking.py")
        else:
            print("\nNext steps:")
            print("  1. python handtracking/collect_data.py")
            print("  2. python handtracking/train_model.py")
            print("  3. python handtracking/handtracking.py")
        
        # Offer live test
        print("\n" + "="*60)
        response = input("\nRun live detection test? (y/n): ")
        if response.lower() == 'y':
            test_live_detection()
    else:
        print("❌ Some critical components failed")
        print("\nPlease fix the issues above before running the system")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
