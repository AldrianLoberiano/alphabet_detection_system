# 🤟 ASL Alphabet Detection System

A real-time American Sign Language (ASL) alphabet recognition system using computer vision and machine learning. Detect all 26 ASL letters, form words, and convert them to speech.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **Real-time Detection**: Recognizes all 26 ASL letters (A-Z) from webcam input
- **Word Formation**: Automatically builds words from detected letters
- **Text-to-Speech**: Speaks formed words using integrated TTS
- **Multiple ML Models**: Supports Random Forest, SVM, KNN, and Decision Trees
- **Interactive UI**: Live confidence scores, FPS monitoring, and visual feedback
- **Data Collection Tool**: Easy-to-use interface for collecting training data
- **Model Training Pipeline**: Automated training with hyperparameter tuning and evaluation

## 📋 Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Requirements

- Python 3.8 or higher
- Webcam/Camera
- Windows/Linux/macOS
- 4GB RAM minimum (8GB recommended)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/handtracking.git
cd handtracking
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python quick_start.py
```

## 🚀 Quick Start

### Option 1: Use Pre-trained Model

If you have a pre-trained model file:

```bash
python -m handtracking.handtracking
```

### Option 2: Train Your Own Model

1. **Collect Training Data**:

   ```bash
   python -m handtracking.collect_data
   ```

   - Follow on-screen instructions to collect samples for each letter (A-Z)
   - Recommended: 100+ samples per letter for best accuracy

2. **Train the Model**:

   ```bash
   python -m handtracking.train_model
   ```

   - Compares multiple ML algorithms
   - Saves the best model automatically
   - Generates performance plots and confusion matrices

3. **Run Detection**:
   ```bash
   python -m handtracking.handtracking
   ```

## 📖 Usage

### Hand Tracking System

```bash
python -m handtracking.handtracking
```

**Controls:**

- `SPACE`: Add detected letter to word
- `BACKSPACE`: Remove last letter
- `ENTER`: Speak the formed word
- `C`: Clear current word
- `Q`: Quit application

**Tips:**

- Keep hand clearly visible in the frame
- Use good lighting conditions
- Position hand at a comfortable distance (1-2 feet from camera)
- Hold each letter steady for accurate detection

### Data Collection

```bash
python -m handtracking.collect_data
```

**Process:**

1. System prompts for each letter (A-Z)
2. Show the ASL sign for the current letter
3. Press `SPACE` to capture sample
4. Collect 100+ samples per letter
5. Data saves automatically to `data/raw/`

### Model Training

```bash
python -m handtracking.train_model
```

**Training Options:**

- Automatically tests multiple algorithms
- Performs hyperparameter optimization
- Generates evaluation metrics
- Saves best model to `model/` directory
- Creates visualization plots in `results/plots/`

## 📁 Project Structure

```
HandTracking/
├── handtracking/           # Main package
│   ├── __init__.py        # Package initializer
│   ├── handtracking.py    # Real-time detection system
│   ├── collect_data.py    # Data collection tool
│   ├── train_model.py     # Model training pipeline
│   ├── config.py          # Configuration settings
│   └── utils.py           # Utility functions
├── data/                  # Data directory
│   ├── raw/              # Raw collected samples
│   └── processed/        # Processed training data
├── model/                # Trained models
├── results/              # Training results
│   └── plots/           # Visualization plots
├── logs/                 # Application logs
├── requirements.txt      # Python dependencies
├── quick_start.py        # Setup verification script
├── diagnostics.py        # System diagnostics
├── README.md            # This file
├── USER_GUIDE.md        # Detailed user guide
├── QUICK_REFERENCE.md   # Quick reference guide
├── PROJECT_SUMMARY.md   # Project overview
├── CHANGELOG.md         # Version history
└── LICENSE              # License information
```

## 📚 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)**: Comprehensive tutorial and troubleshooting
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)**: Command reference and shortcuts
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Detailed project overview
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and updates

## 🎯 Configuration

Edit [handtracking/config.py](handtracking/config.py) to customize:

- Camera settings (resolution, FPS)
- Detection parameters (confidence thresholds)
- UI appearance (colors, fonts)
- Text-to-speech settings (voice, rate)
- Model parameters

## 🧪 Testing

Run diagnostics to test your setup:

```bash
python diagnostics.py
```

This checks:

- Camera functionality
- Hand detection
- Model loading
- Dependencies

## 🐛 Troubleshooting

### Camera Not Working

- Check camera permissions
- Ensure no other application is using the camera
- Try changing `CAMERA_INDEX` in config.py

### Low Accuracy

- Collect more training samples (200+ per letter)
- Ensure consistent lighting
- Check hand is fully visible in frame
- Retrain model with better data

### Import Errors

```bash
pip install --upgrade -r requirements.txt
```

### Performance Issues

- Lower `CAMERA_WIDTH` and `CAMERA_HEIGHT` in config.py
- Close other resource-intensive applications
- Use a simpler model (e.g., Decision Tree instead of Random Forest)

For more help, see [USER_GUIDE.md](USER_GUIDE.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** for hand tracking technology
- **OpenCV** for computer vision capabilities
- **scikit-learn** for machine learning algorithms
- The ASL community for inspiration

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

## ⭐ Show Your Support

If you find this project helpful, please give it a star ⭐ on GitHub!

---

**Happy Signing! 🤟**
