"""
ASL Alphabet Detection System
Real-time Sign Language Recognition using MediaPipe and Machine Learning

A comprehensive system for recognizing American Sign Language alphabet letters
using computer vision and machine learning techniques.

Modules:
    handtracking: Main real-time detection system
    collect_data: Interactive data collection tool
    train_model: Model training and evaluation pipeline
    config: Configuration settings
    utils: Utility functions and helpers

Usage:
    from handtracking.handtracking import ASLDetector
    
    detector = ASLDetector()
    detector.run()

Version: 1.0.0
Author: ASL Detection Team
License: MIT
"""

__version__ = '1.0.0'
__author__ = 'ASL Detection Team'
__license__ = 'MIT'

from .handtracking import ASLDetector
from .config import *

__all__ = ['ASLDetector']
