import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
import tensorflow as tf
from data_preprocessing import DataPreprocessor
import pandas as pd

class ModelEvaluator:
    def __init__(self, model_path):
        """Initialize evaluator with trained model"""
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = DataPreprocessor('')
        self.emotion_labels = {
    Angry: 'Angry',
    Disgust: 'Disgust',
    Fear: 'Fear',
    Happy: 'Happy',
    Sad: 'Sad',
    Surprise: 'Surprise',
    Neutral: 'Neutral'
}

    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("Starting model evaluation...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        report = classification_report(
            y_true, y_pred,
            target_names=list(self.emotion_labels.values()),
            output_dict=True
        )
        
        # Print detailed metrics
        self._print_detailed_metrics(report)
        
        # Generate visualizations
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_per_class_metrics(precision, recall, f1, support)
        self._plot_prediction_distribution(y_pred)
        
        # Analyze misclassifications
        self._analyze_misclassifications(y_true, y_pred, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report
        }
    
    def _print_detailed_metrics(self, report):
        """Print detailed classification metrics"""
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*50)
        
        for emotion, metrics in report.items():
            if emotion in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            print(f"\n{emotion.upper()}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1-score']:.4f}")
            print(f"  Support:   {metrics['support']}")
        
        print(f"\n{'='*20} SUMMARY {'='*20}")
        print(f"Macro Average F1:    {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Average F1: {report['weighted avg']['f1-score']:.4f}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot detailed confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw confusion matrix
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(self.emotion_labels.values()),
            yticklabels=list(self.emotion_labels.values()),
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix (Raw Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Normalized confusion matrix
        sns.heatmap(
            cm_normalized, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=list(self.emotion_labels.values()),
            yticklabels=list(self.emotion_labels.values()),
            ax=axes[1]
        )
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('detailed_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_per_class_metrics(self, precision, recall, f1, support):
        """Plot per-class performance metrics"""
        emotions = list(self.emotion_labels.values())
        
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        plt.savefig('per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_prediction_distribution(self, y_pred):
        """Plot distribution of predictions"""
        emotions = list(self.emotion_labels.values())
        pred_counts = np.bincount(y_pred)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(emotions, pred_counts, alpha=0.7)
        plt.title('Distribution of Predicted Emotions')
        plt.xlabel('Emotions')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, pred_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_misclassifications(self, y_true, y_pred, y_pred_proba):
        """Analyze common misclassification patterns"""
        print("\n" + "="*50)
        print("MISCLASSIFICATION ANALYSIS")
        print("="*50)
        
        # Find misclassified samples
        misclassified = y_true != y_pred
        misclassified_indices = np.where(misclassified)[0]
        
        print(f"Total misclassified samples: {len(misclassified_indices)}")
        print(f"Misclassification rate: {len(misclassified_indices)/len(y_true)*100:.2f}%")
        
        # Analyze confidence of misclassifications
        misclassified_confidences = np.max(y_pred_proba[misclassified], axis=1)
        correct_confidences = np.max(y_pred_proba[~misclassified], axis=1)
        
        print(f"\nAverage confidence in misclassifications: {np.mean(misclassified_confidences):.4f}")
        print(f"Average confidence in correct predictions: {np.mean(correct_confidences):.4f}")
        
        # Most common misclassification pairs
        misclass_pairs = {}
        for i in misclassified_indices:
            true_emotion = self.emotion_labels[y_true[i]]
            pred_emotion = self.emotion_labels[y_pred[i]]
            pair = f"{true_emotion} -> {pred_emotion}"
            misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 most common misclassification patterns:")
        for i, (pair, count) in enumerate(sorted_pairs[:10]):
            print(f"{i+1:2d}. {pair}: {count} cases")
    
    def evaluate_on_dataset(self, csv_path):
        """Evaluate model on FER-2013 test set"""
        # Load test data
        X, y = self.preprocessor.load_fer2013(csv_path)
        
        # Split data (use same random state for consistency)
        _, _, X_test, _, _, y_test = self.preprocessor.split_data(X, y)
        
        print(f"Evaluating on {len(X_test)} test samples...")
        
        return self.evaluate_model(X_test, y_test)

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Emotion Detection Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data CSV')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model_path)
    results = evaluator.evaluate_on_dataset(args.data_path)
    
    print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
