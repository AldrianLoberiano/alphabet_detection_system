# Quick Reference Card - ASL Detection System

## 🚀 Quick Start Commands

```bash
# Setup
pip install -r requirements.txt
python quick_start.py

# Workflow
python handtracking/collect_data.py  # Step 1: Collect data
python handtracking/train_model.py   # Step 2: Train model
python handtracking/handtracking.py  # Step 3: Run detection
```

## ⌨️ Keyboard Shortcuts

### Data Collection (collect_data.py)

| Key | Action                   |
| --- | ------------------------ |
| `S` | Start collecting samples |
| `N` | Next letter              |
| `P` | Previous letter          |
| `R` | Reset current letter     |
| `Q` | Quit and save            |

### Real-time Detection (handtracking.py)

| Key         | Action             |
| ----------- | ------------------ |
| `SPACE`     | Add space to word  |
| `BACKSPACE` | Delete last letter |
| `ENTER`     | Speak word (TTS)   |
| `C`         | Clear word         |
| `Q`         | Quit               |

## 📁 File Structure

```
HandTracking/
├── handtracking/
│   ├── handtracking.py      # Main detection
│   ├── collect_data.py      # Data collection
│   ├── train_model.py       # Model training
│   ├── config.py            # Settings
│   └── utils.py             # Utilities
├── data/raw/                # Training data
├── model/                   # Trained model
├── results/plots/           # Evaluation plots
└── requirements.txt         # Dependencies
```

## 🎯 Configuration (config.py)

### Common Settings

```python
# Camera
CAMERA_CONFIG['device_id'] = 0  # Change camera

# Detection confidence
DETECTION_CONFIG['confidence_threshold'] = 0.6  # 0.0-1.0

# Stability (frames before adding letter)
DETECTION_CONFIG['stability_threshold'] = 15  # Lower = faster

# Text-to-Speech
TTS_CONFIG['enabled'] = True  # Enable/disable TTS
TTS_CONFIG['rate'] = 150      # Speech rate
```

## 🔧 Troubleshooting

### Camera Issues

```python
# Try different camera
CAMERA_CONFIG['device_id'] = 1
```

### Low Accuracy

1. Collect more data (100+ samples/letter)
2. Improve lighting
3. Make clearer gestures
4. Retrain model

### Poor Performance

```python
# Reduce resolution
CAMERA_CONFIG['frame_width'] = 640
CAMERA_CONFIG['frame_height'] = 480
```

### TTS Not Working

```bash
pip install pyttsx3 --upgrade
```

Or disable:

```python
TTS_CONFIG['enabled'] = False
```

## 📊 Expected Results

### Model Accuracy

- Training: 96-99%
- Testing: 93-97%
- Real-world: 90-95%

### Performance

- FPS: 25-30
- Latency: <50ms
- CPU: 30-50%

## 💡 Best Practices

### Data Collection

✅ Good lighting
✅ Plain background
✅ Clear gestures
✅ 100+ samples/letter
✅ Slight variations

### Detection

✅ Hold gesture steady (0.5s)
✅ Center hand in frame
✅ 1-2 feet from camera
✅ All fingers visible

## 🎓 Workflow Tips

### First Time Setup

1. Run `quick_start.py` to verify
2. Start with a few letters (A, B, C)
3. Train and test
4. Collect remaining letters
5. Retrain final model

### Data Collection Strategy

- Day 1: Letters A-M (13 letters)
- Day 2: Letters N-Z (13 letters)
- Each session: ~30-45 minutes
- Take breaks every 5-7 letters

### Training Tips

- Use latest dataset
- Compare multiple models
- Check confusion matrix
- Save best model

## 📞 Quick Help

### File Locations

- **Config**: `handtracking/config.py`
- **Model**: `model/asl_model.pkl`
- **Data**: `data/raw/asl_data_*.pkl`
- **Plots**: `results/plots/`

### Documentation

- **Setup**: `README.md`
- **Tutorial**: `USER_GUIDE.md`
- **Details**: `PROJECT_SUMMARY.md`
- **Changes**: `CHANGELOG.md`

### Python Package Check

```python
python -c "import cv2, mediapipe, sklearn, pyttsx3; print('All packages OK')"
```

## 🔗 Links

- ASL Alphabet: https://www.lifeprint.com/asl101/fingerspelling/abc.htm
- MediaPipe Docs: https://google.github.io/mediapipe/
- scikit-learn: https://scikit-learn.org/

## 📝 Quick Notes

- **Landmark points**: 21 per hand
- **Feature vector**: 63 dimensions (x,y,z)
- **Model types**: 4 (RF, SVM, MLP, GB)
- **Letters**: 26 (A-Z)
- **Default samples**: 100 per letter

---

**Print this card for easy reference!**
