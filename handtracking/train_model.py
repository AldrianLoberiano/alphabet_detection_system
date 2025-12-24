"""
Model Training Module for ASL Alphabet Detection
Trains machine learning models on collected landmark data
"""

import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ASLModelTrainer:
    def __init__(self, data_path=None):
        """Initialize model trainer"""
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.results = {}
        
    def load_data(self, data_path=None):
        """Load training data from pickle file"""
        if data_path:
            self.data_path = data_path
        
        if not os.path.exists(self.data_path):
            print(f"Error: Data file not found at {self.data_path}")
            return False
        
        try:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            
            X = data['X']
            y = data['y']
            
            print(f"\n{'='*60}")
            print("Data Loaded Successfully")
            print(f"{'='*60}")
            print(f"Total samples: {len(X)}")
            print(f"Feature dimensions: {X.shape[1]}")
            print(f"Number of classes: {len(np.unique(y))}")
            print(f"Classes: {sorted(np.unique(y))}")
            
            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            print(f"\nClass distribution:")
            for letter, count in zip(unique, counts):
                print(f"  {letter}: {count} samples")
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            print(f"\nTraining samples: {len(self.X_train)}")
            print(f"Testing samples: {len(self.X_test)}")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def train_random_forest(self):
        """Train Random Forest Classifier"""
        print("Training Random Forest Classifier...")
        
        # Grid search for hyperparameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_rf = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate
        train_score = best_rf.score(self.X_train, self.y_train)
        test_score = best_rf.score(self.X_test, self.y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}\n")
        
        self.models['Random Forest'] = best_rf
        self.results['Random Forest'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': best_rf
        }
        
        return best_rf
    
    def train_svm(self):
        """Train Support Vector Machine"""
        print("Training Support Vector Machine...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly']
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_svm = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        train_score = best_svm.score(self.X_train, self.y_train)
        test_score = best_svm.score(self.X_test, self.y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}\n")
        
        self.models['SVM'] = best_svm
        self.results['SVM'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': best_svm
        }
        
        return best_svm
    
    def train_neural_network(self):
        """Train Neural Network (MLP)"""
        print("Training Neural Network (MLP)...")
        
        param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        mlp = MLPClassifier(max_iter=1000, random_state=42)
        grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_mlp = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        train_score = best_mlp.score(self.X_train, self.y_train)
        test_score = best_mlp.score(self.X_test, self.y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}\n")
        
        self.models['Neural Network'] = best_mlp
        self.results['Neural Network'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': best_mlp
        }
        
        return best_mlp
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting Classifier"""
        print("Training Gradient Boosting Classifier...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_gb = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        train_score = best_gb.score(self.X_train, self.y_train)
        test_score = best_gb.score(self.X_test, self.y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Testing accuracy: {test_score:.4f}\n")
        
        self.models['Gradient Boosting'] = best_gb
        self.results['Gradient Boosting'] = {
            'train_score': train_score,
            'test_score': test_score,
            'model': best_gb
        }
        
        return best_gb
    
    def evaluate_model(self, model, model_name):
        """Detailed evaluation of a model"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}\n")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_dir = 'results/plots'
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{plot_dir}/confusion_matrix_{model_name.replace(' ', '_')}_{timestamp}.png")
        plt.close()
        
        return accuracy
    
    def compare_models(self):
        """Compare all trained models"""
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}\n")
        
        model_names = []
        train_scores = []
        test_scores = []
        
        for name, result in self.results.items():
            model_names.append(name)
            train_scores.append(result['train_score'])
            test_scores.append(result['test_score'])
        
        # Print comparison
        for name, train, test in zip(model_names, train_scores, test_scores):
            print(f"{name:20s} - Train: {train:.4f}  Test: {test:.4f}")
        
        # Find best model
        best_idx = np.argmax(test_scores)
        best_name = model_names[best_idx]
        self.best_model = self.results[best_name]['model']
        
        print(f"\nBest Model: {best_name} (Test Accuracy: {test_scores[best_idx]:.4f})")
        print(f"{'='*60}\n")
        
        # Plot comparison
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, train_scores, width, label='Training')
        ax.bar(x + width/2, test_scores, width, label='Testing')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = 'results/plots'
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{plot_dir}/model_comparison_{timestamp}.png")
        plt.close()
        
        return best_name
    
    def save_model(self, model=None, filename='model/asl_model.pkl'):
        """Save trained model to file"""
        if model is None:
            model = self.best_model
        
        if model is None:
            print("Error: No model to save")
            return False
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        model_data = {
            'model': model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_dim': self.X_train.shape[1]
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n{'='*60}")
        print(f"Model saved successfully: {filename}")
        print(f"{'='*60}\n")
        
        return True
    
    def train_all_models(self):
        """Train all models and compare"""
        print("\n" + "="*60)
        print("Training All Models")
        print("="*60 + "\n")
        
        # Train models
        self.train_random_forest()
        self.train_svm()
        self.train_neural_network()
        self.train_gradient_boosting()
        
        # Compare models
        best_model_name = self.compare_models()
        
        # Evaluate best model in detail
        self.evaluate_model(self.best_model, best_model_name)
        
        return self.best_model


def main():
    """Main training pipeline"""
    # Find latest data file
    data_dir = 'data/raw'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run collect_data.py first to collect training data.")
        return
    
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not data_files:
        print("Error: No data files found")
        print("Please run collect_data.py first to collect training data.")
        return
    
    # Use latest file
    latest_file = sorted(data_files)[-1]
    data_path = os.path.join(data_dir, latest_file)
    
    print(f"\nUsing data file: {data_path}\n")
    
    # Initialize trainer
    trainer = ASLModelTrainer(data_path)
    
    # Load data
    if not trainer.load_data():
        return
    
    # Train all models
    trainer.train_all_models()
    
    # Save best model
    trainer.save_model()
    
    print("\nTraining pipeline completed successfully!")
    print("You can now run handtracking.py to test real-time detection.")


if __name__ == "__main__":
    main()
