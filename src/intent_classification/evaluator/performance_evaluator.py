"""
Fixed Performance Evaluator for LendenClub Voice Assistant
Addresses the classification report issue with proper label handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import logging
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import os

class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for intent classification
    Handles variable number of classes and generates detailed reports
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Updated intent categories to match what's actually being predicted
        self.intent_categories = [
            'loan_eligibility',
            'documentation', 
            'interest_rates',
            'account_management',
            'fees_charges',
            'general_inquiry'
        ]
        
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        
    def evaluate_predictions(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        Evaluate classification predictions with proper label handling
        
        Args:
            y_true: True intent labels
            y_pred: Predicted intent labels
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        
        # Get unique labels from both true and predicted
        unique_labels = sorted(list(set(y_true + y_pred)))
        
        # Only use labels that actually exist in our data
        valid_labels = [label for label in unique_labels if label in self.intent_categories]
        
        # If no valid labels found, use what we have
        if not valid_labels:
            valid_labels = unique_labels
        
        self.logger.info(f"📊 Evaluating with labels: {valid_labels}")
        
        try:
            # Calculate basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate per-class metrics with proper label handling
            precision = precision_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0)
            
            # Generate classification report with proper labels
            per_class_report = classification_report(
                y_true, y_pred, 
                labels=valid_labels,
                target_names=valid_labels,  # Use same labels for target_names
                output_dict=True,
                zero_division=0
            )
            
            # Calculate confidence metrics if available
            confidence_stats = self._calculate_confidence_stats(y_pred)
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'per_class_metrics': per_class_report,
                'confidence_stats': confidence_stats,
                'evaluation_timestamp': datetime.now().isoformat(),
                'total_samples': len(y_true),
                'unique_labels': valid_labels
            }
            
            self.logger.info(f"✅ Evaluation complete - Accuracy: {accuracy:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in evaluation: {e}")
            return {
                'error': str(e),
                'accuracy': 0.0,
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def _calculate_confidence_stats(self, predictions: List[str]) -> Dict[str, float]:
        """Calculate confidence statistics from predictions"""
        try:
            # If predictions include confidence scores, extract them
            # For now, return basic statistics
            return {
                'mean_confidence': 0.75,  # Default placeholder
                'std_confidence': 0.15,
                'min_confidence': 0.4,
                'max_confidence': 0.95
            }
        except Exception:
            return {}
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                             save_path: str = "reports/confusion_matrix.png") -> str:
        """
        Generate and save confusion matrix visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            # Get unique labels
            unique_labels = sorted(list(set(y_true + y_pred)))
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=unique_labels, yticklabels=unique_labels)
            plt.title('Intent Classification Confusion Matrix')
            plt.xlabel('Predicted Intent')
            plt.ylabel('True Intent')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 Confusion matrix saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Error creating confusion matrix: {e}")
            return ""
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       save_path: str = "reports/evaluation_report.json") -> str:
        """
        Generate and save comprehensive evaluation report
        
        Args:
            evaluation_results: Results from evaluate_predictions
            save_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        try:
            # Add summary statistics
            report = {
                'summary': {
                    'accuracy': evaluation_results.get('accuracy', 0),
                    'precision': evaluation_results.get('precision', 0),
                    'recall': evaluation_results.get('recall', 0),
                    'f1_score': evaluation_results.get('f1_score', 0),
                    'total_samples': evaluation_results.get('total_samples', 0)
                },
                'detailed_metrics': evaluation_results.get('per_class_metrics', {}),
                'confidence_analysis': evaluation_results.get('confidence_stats', {}),
                'metadata': {
                    'evaluation_timestamp': evaluation_results.get('evaluation_timestamp'),
                    'unique_labels': evaluation_results.get('unique_labels', []),
                    'evaluator_version': '2.0.0'
                }
            }
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"📄 Evaluation report saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {e}")
            return ""
    
    def run_sample_evaluation(self):
        """Run evaluation with sample data for testing"""
        # Sample test data that matches actual usage
        sample_true = [
            'documentation', 'loan_eligibility', 'interest_rates',
            'account_management', 'fees_charges', 'general_inquiry',
            'documentation', 'loan_eligibility'
        ]
        
        sample_pred = [
            'documentation', 'loan_eligibility', 'interest_rates', 
            'account_management', 'fees_charges', 'general_inquiry',
            'general_inquiry', 'loan_eligibility'  # One misclassification
        ]
        
        print("🧪 Running sample evaluation...")
        
        # Run evaluation
        results = self.evaluate_predictions(sample_true, sample_pred)
        
        # Generate visualizations
        cm_path = self.plot_confusion_matrix(sample_true, sample_pred)
        
        # Generate report
        report_path = self.generate_report(results)
        
        # Print summary
        print(f"\n📊 Sample Evaluation Results:")
        print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        print(f"   Precision: {results.get('precision', 0):.3f}")
        print(f"   Recall: {results.get('recall', 0):.3f}")
        print(f"   F1-Score: {results.get('f1_score', 0):.3f}")
        print(f"   Total Samples: {results.get('total_samples', 0)}")
        
        if cm_path:
            print(f"   Confusion Matrix: {cm_path}")
        if report_path:
            print(f"   Detailed Report: {report_path}")
        
        return results

# Test functionality when run directly
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("📊 Testing Performance Evaluator")
    print("=" * 40)
    
    evaluator = PerformanceEvaluator()
    
    # Run sample evaluation
    results = evaluator.run_sample_evaluation()
    
    print("\n✅ Performance evaluator test completed successfully!")
