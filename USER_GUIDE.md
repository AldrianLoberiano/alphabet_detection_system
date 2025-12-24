# ASL Detection System - User Guide

## 📖 Complete Tutorial

### Table of Contents

1. [Installation](#installation)
2. [Data Collection](#data-collection)
3. [Model Training](#model-training)
4. [Real-time Detection](#real-time-detection)
5. [Tips & Best Practices](#tips--best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python quick_start.py
```

This will check if all required packages are installed correctly.

---

## Data Collection

### Overview

The data collection module helps you gather training samples for all 26 ASL letters (A-Z). You'll need approximately 100 samples per letter for good model performance.

### Running Data Collection

```bash
python handtracking/collect_data.py
```

### User Interface

- **Top Left**: Current letter being collected
- **Progress Bar**: Shows samples collected for current letter
- **Top Right**: Overall progress (Letter X/26)
- **Bottom**: Control instructions

### Controls

| Key | Action                                        |
| --- | --------------------------------------------- |
| `S` | Start collecting samples (3-second countdown) |
| `N` | Move to next letter                           |
| `P` | Go to previous letter                         |
| `R` | Reset current letter data                     |
| `Q` | Quit and save collected data                  |

### Collection Process

1. **Start the program**

   ```bash
   python handtracking/collect_data.py
   ```

2. **Position your hand**

   - Make the ASL gesture for letter 'A'
   - Keep your hand clearly visible in the frame
   - Ensure good lighting

3. **Press 'S' to start**

   - A 3-second countdown will appear
   - Get ready with your hand gesture
   - Hold the gesture steady

4. **Collection begins**

   - The system automatically captures 100 samples
   - Keep your hand relatively steady
   - Make slight variations (rotation, position) for better generalization

5. **Automatic advancement**

   - Once 100 samples are collected, it moves to the next letter
   - Repeat the process for all 26 letters

6. **Save data**
   - Press 'Q' when done
   - Data is automatically saved to `data/raw/asl_data_TIMESTAMP.pkl`

### Tips for Good Data Quality

✅ **DO:**

- Use consistent lighting
- Keep background plain and uncluttered
- Make clear, distinct gestures
- Vary hand position slightly (±5-10cm)
- Rotate hand slightly for different angles
- Maintain consistent distance from camera (1-2 feet)

❌ **DON'T:**

- Make blurry or rushed gestures
- Block hand with other objects
- Use poor lighting
- Make gestures too far from camera
- Skip letters (complete all 26)

---

## Model Training

### Overview

The training module trains multiple machine learning models and selects the best performer.

### Running Training

```bash
python handtracking/train_model.py
```

### What Happens During Training

1. **Data Loading**

   - Loads most recent dataset from `data/raw/`
   - Displays dataset statistics
   - Shows class distribution

2. **Model Training**
   - Trains 4 different models:
     - Random Forest Classifier
     - Support Vector Machine (SVM)
     - Neural Network (MLP)
     - Gradient Boosting Classifier
3. **Hyperparameter Tuning**

   - Uses GridSearchCV for optimal parameters
   - 5-fold cross-validation
   - Parallel processing for speed

4. **Model Evaluation**

   - Compares all models
   - Generates confusion matrices
   - Creates performance comparison plots

5. **Model Saving**
   - Saves best performing model to `model/asl_model.pkl`
   - Includes label encoder and scaler

### Output Files

```
model/
  └── asl_model.pkl           # Trained model

results/plots/
  ├── confusion_matrix_*.png  # Confusion matrices
  └── model_comparison_*.png  # Performance comparison
```

### Expected Performance

| Model             | Typical Test Accuracy |
| ----------------- | --------------------- |
| Random Forest     | 95-97%                |
| SVM               | 94-96%                |
| Neural Network    | 93-95%                |
| Gradient Boosting | 94-96%                |

### Training Time

- **Small dataset** (100 samples/letter): 2-5 minutes
- **Large dataset** (200+ samples/letter): 5-15 minutes

_Time varies based on CPU performance and hyperparameter search space_

---

## Real-time Detection

### Overview

The main detection system recognizes ASL letters in real-time using your webcam.

### Running Detection

```bash
python handtracking/handtracking.py
```

### User Interface

#### Top Panel

- **FPS Counter**: Shows frames per second
- **Detected Letter**: Current recognized letter
- **Confidence Bar**: Prediction confidence (0-100%)

#### Right Side

- **Word Display**: Shows formed word from detected letters

#### Bottom

- **Control Instructions**: Keyboard shortcuts

### Controls

| Key         | Action                      |
| ----------- | --------------------------- |
| `Q`         | Quit application            |
| `SPACE`     | Add space to word           |
| `BACKSPACE` | Delete last letter          |
| `ENTER`     | Speak word (Text-to-Speech) |
| `C`         | Clear formed word           |

### How It Works

1. **Hand Detection**

   - MediaPipe detects hand in video frame
   - Extracts 21 landmark points (63 coordinates)
   - Normalizes landmarks for consistency

2. **Prediction**

   - Landmarks fed to trained ML model
   - Model predicts letter and confidence score
   - Prediction smoothed over multiple frames

3. **Word Formation**

   - Letter must be stable for 15 frames (0.5 seconds)
   - Automatically adds to word
   - Prevents duplicate detection

4. **Text-to-Speech**
   - Speaks individual letters as detected
   - Press ENTER to speak complete word

### Features

#### Prediction Smoothing

- Uses buffer of last 10 predictions
- Takes most common letter to reduce jitter
- Improves stability and accuracy

#### Confidence Visualization

- **Green** (>75%): High confidence
- **Orange** (60-75%): Medium confidence
- **No display** (<60%): Low confidence (not shown)

#### Visual Feedback

- Hand landmarks drawn in real-time
- Hand connections shown
- Bounding box around hand (optional)

### Usage Tips

#### For Best Accuracy

1. **Hand Positioning**

   - Keep hand centered in frame
   - Distance: 1-2 feet from camera
   - Show full hand (all fingers visible)

2. **Gesture Clarity**

   - Make clear, distinct gestures
   - Hold steady for 0.5 seconds
   - Avoid rapid movements

3. **Environment**

   - Good lighting (front-lit, not backlit)
   - Plain background
   - Minimal camera movement

4. **Spelling Words**
   - Hold each letter until it appears in word
   - Wait for letter confirmation (spoken)
   - Use BACKSPACE to correct mistakes

---

## Tips & Best Practices

### Data Collection

1. **Quality over Quantity**

   - Better to have 100 good samples than 200 poor ones
   - Ensure each sample is clear and distinct

2. **Variation is Good**

   - Slight position changes
   - Minor rotation differences
   - Different hand sizes (if multiple users)

3. **Consistency Matters**
   - Same lighting conditions
   - Similar background
   - Consistent camera setup

### Training

1. **Multiple Training Runs**

   - Train multiple times
   - Compare results
   - Keep best performing model

2. **Dataset Size**
   - Minimum: 100 samples/letter
   - Recommended: 150-200 samples/letter
   - More data = better generalization

### Real-time Use

1. **Warm-up Period**

   - First few predictions may be unstable
   - System stabilizes after 2-3 seconds

2. **Deliberate Gestures**

   - No need to rush
   - Hold each letter clearly
   - Wait for confirmation

3. **Error Recovery**
   - Use BACKSPACE for mistakes
   - Clear and restart with 'C'
   - Adjust hand position if detection fails

---

## Troubleshooting

### Camera Issues

**Problem**: Camera not detected

```python
# Solution: Try different camera index in config.py
CAMERA_CONFIG['device_id'] = 1  # Try 0, 1, 2, etc.
```

**Problem**: Poor video quality

```python
# Solution: Adjust resolution in config.py
CAMERA_CONFIG['frame_width'] = 640
CAMERA_CONFIG['frame_height'] = 480
```

### Detection Issues

**Problem**: Low accuracy / wrong predictions

**Solutions**:

1. Collect more training data
2. Improve lighting conditions
3. Make clearer gestures
4. Retrain model
5. Adjust confidence threshold:
   ```python
   DETECTION_CONFIG['confidence_threshold'] = 0.7  # Increase for stricter
   ```

**Problem**: Delayed detection

**Solutions**:

1. Reduce stability threshold:
   ```python
   DETECTION_CONFIG['stability_threshold'] = 10  # Default: 15
   ```
2. Reduce prediction buffer:
   ```python
   DETECTION_CONFIG['prediction_buffer_size'] = 5  # Default: 10
   ```

### TTS Issues

**Problem**: Text-to-speech not working

**Solutions**:

```bash
# Reinstall pyttsx3
pip install pyttsx3 --upgrade

# Windows: Install pywin32
pip install pywin32

# Disable TTS in config.py
TTS_CONFIG['enabled'] = False
```

### Performance Issues

**Problem**: Low FPS

**Solutions**:

1. Reduce camera resolution
2. Close other applications
3. Use simpler model (Random Forest instead of Neural Network)
4. Disable unnecessary UI elements

**Problem**: High CPU usage

**Solutions**:

1. Reduce FPS in config
2. Increase frame skip
3. Use lighter model

---

## Advanced Configuration

### Custom Model Training

Edit [train_model.py](handtracking/train_model.py) to customize:

```python
# Add your own model
from sklearn.ensemble import AdaBoostClassifier

def train_custom_model(self):
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(self.X_train, self.y_train)
    return model
```

### Custom Gestures

To add custom gestures beyond A-Z:

1. Modify `ASL_LETTERS` in [config.py](handtracking/config.py)
2. Collect data for new gestures
3. Retrain model

### Integration

To use detection in your own project:

```python
from handtracking.handtracking import ASLDetector

# Initialize detector
detector = ASLDetector(model_path='model/asl_model.pkl')

# Your video capture loop
# ... get frame ...

# Process frame
results = detector.hands.process(imgRGB)
if results.multi_hand_landmarks:
    landmarks = detector.extract_landmarks(results.multi_hand_landmarks[0])
    letter, confidence = detector.predict_letter(landmarks)
    print(f"Detected: {letter} ({confidence:.2f})")
```

---

## Support

- **Documentation**: README.md
- **Code Issues**: Check inline comments
- **GitHub Issues**: Report bugs
- **Configuration**: See config.py for all settings

---

**Happy Signing! 🤟**
