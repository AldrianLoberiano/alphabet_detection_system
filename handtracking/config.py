"""
Configuration file for ASL Detection System
Centralized settings and hyperparameters
"""

# Camera Settings
CAMERA_CONFIG = {
    'device_id': 0,
    'frame_width': 1280,
    'frame_height': 720,
    'fps': 30,
    'flip_horizontal': True
}

# MediaPipe Hand Detection Settings
HAND_DETECTION_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.7,
    'model_complexity': 1  # 0 or 1
}

# Data Collection Settings
DATA_COLLECTION_CONFIG = {
    'samples_per_letter': 100,
    'data_dir': 'data/raw',
    'countdown_duration': 3,  # seconds before collection starts
    'auto_advance': True  # Auto-advance to next letter when done
}

# Model Training Settings
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1,  # Use all CPU cores
    'verbose': 1
}

# Model Hyperparameters
MODEL_HYPERPARAMETERS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'poly']
    },
    'neural_network': {
        'hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [1000]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
}

# Real-time Detection Settings
DETECTION_CONFIG = {
    'model_path': 'model/asl_model.pkl',
    'confidence_threshold': 0.6,
    'high_confidence_threshold': 0.75,
    'prediction_buffer_size': 10,
    'stability_threshold': 15,  # frames before adding letter to word
}

# Text-to-Speech Settings
TTS_CONFIG = {
    'enabled': True,
    'rate': 150,  # words per minute
    'volume': 1.0,  # 0.0 to 1.0
    'voice_index': 0  # 0 for default voice
}

# UI Color Scheme
UI_COLORS = {
    'primary': (0, 255, 0),        # Green
    'secondary': (255, 0, 255),     # Magenta
    'text': (255, 255, 255),        # White
    'background': (50, 50, 50),     # Dark Gray
    'warning': (0, 165, 255),       # Orange
    'error': (0, 0, 255),           # Red
    'success': (0, 255, 0)          # Green
}

# ASL Alphabet
ASL_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# File Paths
PATHS = {
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'models': 'model',
    'results': 'results',
    'plots': 'results/plots',
    'logs': 'logs'
}

# Landmark Processing
LANDMARK_CONFIG = {
    'use_normalization': True,
    'use_scale_invariance': True,
    'reference_point': 0,  # Wrist (landmark 0)
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    'show_fps': True,
    'show_confidence': True,
    'show_landmarks': True,
    'save_logs': False
}

# Keyboard Shortcuts
KEYBOARD_SHORTCUTS = {
    'quit': 'q',
    'space': ' ',
    'backspace': 8,
    'enter': 13,
    'clear': 'c',
    'start_collection': 's',
    'next_letter': 'n',
    'previous_letter': 'p',
    'reset': 'r'
}
