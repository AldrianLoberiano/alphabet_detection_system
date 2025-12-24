"""
Quick Start Script for ASL Detection System
This script guides users through the complete setup and usage
"""

import os
import sys

def print_header(text):
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def check_files():
    """Check if all necessary files exist"""
    required_files = [
        'handtracking/handtracking.py',
        'handtracking/collect_data.py',
        'handtracking/train_model.py',
        'handtracking/config.py',
        'handtracking/utils.py',
        'requirements.txt'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        return False
    
    print("✓ All required files present")
    return True

def check_dependencies():
    """Check if dependencies are installed"""
    required_packages = [
        'cv2', 'mediapipe', 'numpy', 'sklearn', 
        'matplotlib', 'seaborn', 'pyttsx3'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("\n❌ Missing required packages:")
        for package in missing:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies installed")
    return True

def check_data():
    """Check if training data exists"""
    data_dir = 'data/raw'
    if os.path.exists(data_dir):
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
        if data_files:
            print(f"✓ Found {len(data_files)} data file(s)")
            return True
    
    print("⚠ No training data found")
    return False

def check_model():
    """Check if trained model exists"""
    model_path = 'model/asl_model.pkl'
    if os.path.exists(model_path):
        print("✓ Trained model found")
        return True
    
    print("⚠ No trained model found")
    return False

def main():
    print_header("ASL Alphabet Detection System - Quick Start")
    
    print("Step 1: Checking System Setup")
    print("-" * 60)
    
    if not check_files():
        print("\n❌ Setup incomplete. Please check your installation.")
        return
    
    if not check_dependencies():
        print("\n❌ Dependencies missing. Please install them first.")
        return
    
    print("\n✅ System setup complete!\n")
    
    print("Step 2: Checking Project Status")
    print("-" * 60)
    
    has_data = check_data()
    has_model = check_model()
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60 + "\n")
    
    if not has_data:
        print("1️⃣  Collect Training Data:")
        print("   python handtracking/collect_data.py")
        print("   • Collect 100 samples per letter (A-Z)")
        print("   • Use good lighting and clear hand visibility")
        print("   • Follow on-screen instructions\n")
    
    if not has_model:
        print("2️⃣  Train the Model:")
        print("   python handtracking/train_model.py")
        print("   • Trains multiple ML models")
        print("   • Evaluates and selects best model")
        print("   • Saves model for real-time detection\n")
    
    print("3️⃣  Run Real-time Detection:")
    print("   python handtracking/handtracking.py")
    print("   • Recognizes ASL letters in real-time")
    print("   • Forms words from detected letters")
    print("   • Text-to-speech output available\n")
    
    if has_data and has_model:
        print("✅ You're all set! Run the detection system now:")
        print("   python handtracking/handtracking.py\n")
    
    print("="*60)
    print("📚 Documentation: See README.md for detailed information")
    print("🐛 Issues: Report bugs on GitHub Issues")
    print("💡 Tips: Read config.py for customization options")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
