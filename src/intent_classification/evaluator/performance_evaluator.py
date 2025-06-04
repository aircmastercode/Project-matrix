"""
LendenClub Voice Assistant - Performance Evaluator
Comprehensive evaluation framework for intent classification models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available for performance evaluation")
    SKLEARN_AVAILABLE = False

from config.settings import (
    INTENT_CATEGORIES, EVALUATION_CONFIG, DATA_PATHS,
    get_intent_labels
)

class PerformanceEvaluator:
    """
    Comprehensive performance evaluation for intent classification models.
    
    Features:
    - Accuracy, precision, recall, F1-score calculation
    - Confusion matrix visualization
    - Confidence score analysis
    - Cross-validation support
    - Benchmark comparison across models
    - Error analysis and reporting
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the performance evaluator.
        
        Args:
            config: Optional custom evaluation configuration
        """
        self.config = config or EVALUATION_CONFIG
        self.intent_labels = get_intent_labels()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Storage for evaluation results
        self.evaluation_results = {}
        self.benchmark_results = {}
        
        # Ensure output directory exists
        self.output_dir = DATA_PATHS["processed_data"] / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Performance evaluator initialized")
    
    def evaluate_predictions(
        self, 
        predictions: List[str], 
        true_labels: List[str],
        confidence_scores: List[float] = None,
        model_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions against true labels.
        
        Args:
            predictions: List of predicted intent labels
            true_labels: List of true intent labels
            confidence_scores: Optional confidence scores for predictions
            model_name: Name of the model being evaluated
            
        Returns:
            Comprehensive evaluation metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available for evaluation"}
        
        if len(predictions) != len(true_labels):
            raise ValueError("Predictions and true labels must have same length")
        
        self.logger.info(f"Evaluating {len(predictions)} predictions for model: {model_name}")
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_report = classification_report(
            true_labels, predictions, 
            target_names=self.intent_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=self.intent_labels)
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence_scores(
            predictions, true_labels, confidence_scores
        ) if confidence_scores else {}
        
        # Error analysis
        error_analysis = self._analyze_errors(predictions, true_labels)
        
        # Compile results
        results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(predictions),
            "overall_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "per_class_metrics": per_class_report,
            "confusion_matrix": cm.tolist(),
            "confidence_analysis": confidence_analysis,
            "error_analysis": error_analysis,
            "performance_grade": self._calculate_performance_grade(accuracy, precision, recall, f1)
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        self.logger.info(f"Evaluation complete for {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
        
        return results
    
    def _analyze_confidence_scores(
        self, 
        predictions: List[str], 
        true_labels: List[str],
        confidence_scores: List[float]
    ) -> Dict[str, Any]:
        """Analyze confidence score distribution and calibration."""
        if not confidence_scores:
            return {}
        
        # Confidence statistics
        conf_stats = {
            "mean": np.mean(confidence_scores),
            "std": np.std(confidence_scores),
            "min": np.min(confidence_scores),
            "max": np.max(confidence_scores),
            "median": np.median(confidence_scores)
        }
        
        # Confidence buckets analysis
        buckets = self.config.get("confidence_buckets", [0.0, 0.5, 0.7, 0.8, 0.9, 1.0])
        bucket_analysis = {}
        
        for i in range(len(buckets) - 1):
            bucket_min, bucket_max = buckets[i], buckets[i + 1]
            
            # Find predictions in this bucket
            in_bucket = [
                (pred == true) for pred, true, conf in zip(predictions, true_labels, confidence_scores)
                if bucket_min <= conf < bucket_max
            ]
            
            if in_bucket:
                bucket_analysis[f"{bucket_min}-{bucket_max}"] = {
                    "count": len(in_bucket),
                    "accuracy": sum(in_bucket) / len(in_bucket),
                    "percentage": len(in_bucket) / len(confidence_scores) * 100
                }
        
        # Correlation between confidence and correctness
        correct_predictions = [pred == true for pred, true in zip(predictions, true_labels)]
        
        if len(set(correct_predictions)) > 1:  # Avoid correlation issues
            confidence_correlation = np.corrcoef(confidence_scores, correct_predictions)[0, 1]
        else:
            confidence_correlation = 0.0
        
        return {
            "statistics": conf_stats,
            "bucket_analysis": bucket_analysis,
            "confidence_correlation": confidence_correlation,
            "low_confidence_errors": sum(1 for pred, true, conf in zip(predictions, true_labels, confidence_scores)
                                       if pred != true and conf < 0.7)
        }
    
    def _analyze_errors(self, predictions: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """Analyze prediction errors and common failure patterns."""
        errors = []
        
        for pred, true in zip(predictions, true_labels):
            if pred != true:
                errors.append((true, pred))
        
        if not errors:
            return {"total_errors": 0, "error_patterns": {}}
        
        # Count error patterns
        error_patterns = {}
        for true_label, pred_label in errors:
            pattern = f"{true_label} -> {pred_label}"
            error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
        
        # Sort by frequency
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Most confused intent pairs
        top_confusions = sorted_patterns[:5]
        
        # Per-intent error rates
        intent_errors = {}
        for intent in self.intent_labels:
            total_intent = sum(1 for true in true_labels if true == intent)
            intent_wrong = sum(1 for true, pred in zip(true_labels, predictions) 
                             if true == intent and pred != intent)
            
            if total_intent > 0:
                intent_errors[intent] = {
                    "total_samples": total_intent,
                    "errors": intent_wrong,
                    "error_rate": intent_wrong / total_intent
                }
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(predictions),
            "error_patterns": dict(sorted_patterns),
            "top_confusions": top_confusions,
            "per_intent_errors": intent_errors
        }
    
    def _calculate_performance_grade(
        self, 
        accuracy: float, 
        precision: float, 
        recall: float, 
        f1: float
    ) -> str:
        """Calculate overall performance grade."""
        thresholds = self.config.get("performance_thresholds", {})
        
        scores = [accuracy, precision, recall, f1]
        avg_score = np.mean(scores)
        
        if avg_score >= 0.90:
            return "A"
        elif avg_score >= 0.85:
            return "B"
        elif avg_score >= 0.75:
            return "C"
        elif avg_score >= 0.65:
            return "D"
        else:
            return "F"
    
    def visualize_confusion_matrix(
        self, 
        model_name: str, 
        save_plot: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[str]:
        """
        Create and optionally save confusion matrix visualization.
        
        Args:
            model_name: Name of model to visualize
            save_plot: Whether to save the plot to file
            figsize: Figure size for the plot
            
        Returns:
            Path to saved plot file if save_plot=True
        """
        if model_name not in self.evaluation_results:
            self.logger.error(f"No evaluation results found for model: {model_name}")
            return None
        
        results = self.evaluation_results[model_name]
        cm = np.array(results["confusion_matrix"])
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.intent_labels,
            yticklabels=self.intent_labels,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {results["overall_metrics"]["accuracy"]:.3f}')
        plt.xlabel('Predicted Intent')
        plt.ylabel('True Intent')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / f"confusion_matrix_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confusion matrix saved to {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def visualize_confidence_distribution(
        self, 
        model_name: str, 
        save_plot: bool = True,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Optional[str]:
        """Visualize confidence score distribution."""
        if model_name not in self.evaluation_results:
            return None
        
        results = self.evaluation_results[model_name]
        confidence_analysis = results.get("confidence_analysis", {})
        
        if not confidence_analysis:
            self.logger.warning(f"No confidence analysis available for {model_name}")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Confidence bucket analysis
        bucket_data = confidence_analysis.get("bucket_analysis", {})
        if bucket_data:
            buckets = list(bucket_data.keys())
            accuracies = [bucket_data[bucket]["accuracy"] for bucket in buckets]
            counts = [bucket_data[bucket]["count"] for bucket in buckets]
            
            ax1.bar(buckets, accuracies, alpha=0.7, color='skyblue')
            ax1.set_xlabel('Confidence Buckets')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy by Confidence Bucket')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add count labels
            for i, (bucket, count) in enumerate(zip(buckets, counts)):
                ax1.text(i, accuracies[i] + 0.01, f'n={count}', ha='center', va='bottom')
        
        # Confidence statistics
        stats = confidence_analysis.get("statistics", {})
        if stats:
            ax2.text(0.1, 0.8, f"Mean: {stats.get('mean', 0):.3f}", transform=ax2.transAxes)
            ax2.text(0.1, 0.7, f"Std: {stats.get('std', 0):.3f}", transform=ax2.transAxes)
            ax2.text(0.1, 0.6, f"Median: {stats.get('median', 0):.3f}", transform=ax2.transAxes)
            ax2.text(0.1, 0.5, f"Min: {stats.get('min', 0):.3f}", transform=ax2.transAxes)
            ax2.text(0.1, 0.4, f"Max: {stats.get('max', 0):.3f}", transform=ax2.transAxes)
            
            correlation = confidence_analysis.get("confidence_correlation", 0)
            ax2.text(0.1, 0.2, f"Confidence-Accuracy Correlation: {correlation:.3f}", transform=ax2.transAxes)
        
        ax2.set_title('Confidence Statistics')
        ax2.axis('off')
        
        plt.suptitle(f'Confidence Analysis - {model_name}')
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / f"confidence_analysis_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Confidence analysis plot saved to {plot_path}")
            return str(plot_path)
        else:
            plt.show()
            return None
    
    def compare_models(self, model_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare performance across multiple models.
        
        Args:
            model_names: List of model names to compare, defaults to all evaluated models
            
        Returns:
            Comparison results
        """
        model_names = model_names or list(self.evaluation_results.keys())
        
        if not model_names:
            return {"error": "No models to compare"}
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name in self.evaluation_results:
                results = self.evaluation_results[model_name]
                metrics = results["overall_metrics"]
                
                comparison_data.append({
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "grade": results["performance_grade"],
                    "sample_size": results["sample_size"]
                })
        
        if not comparison_data:
            return {"error": "No valid model results found"}
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Find best performing model
        best_model = df.loc[df['f1_score'].idxmax()]
        
        # Calculate ranking
        df['rank'] = df['f1_score'].rank(method='dense', ascending=False)
        
        comparison_summary = {
            "models_compared": len(df),
            "best_model": {
                "name": best_model["model"],
                "f1_score": best_model["f1_score"],
                "accuracy": best_model["accuracy"]
            },
            "detailed_comparison": df.to_dict('records'),
            "performance_ranking": df.sort_values('f1_score', ascending=False)[['model', 'f1_score', 'rank']].to_dict('records')
        }
        
        self.benchmark_results["comparison"] = comparison_summary
        
        return comparison_summary
    
    def generate_report(self, model_name: str, include_visualizations: bool = True) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model_name: Name of model to report on
            include_visualizations: Whether to include plots in report
            
        Returns:
            Path to generated report file
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for model: {model_name}")
        
        results = self.evaluation_results[model_name]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate report content
        report_content = self._generate_report_content(results, include_visualizations)
        
        # Save report
        report_path = self.output_dir / f"evaluation_report_{model_name}_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Evaluation report generated: {report_path}")
        
        return str(report_path)
    
    def _generate_report_content(self, results: Dict[str, Any], include_viz: bool) -> str:
        """Generate markdown content for evaluation report."""
        model_name = results["model_name"]
        metrics = results["overall_metrics"]
        
        report = f"""# Intent Classification Evaluation Report

## Model: {model_name}
**Evaluation Date:** {results["timestamp"]}  
**Sample Size:** {results["sample_size"]}  
**Performance Grade:** {results["performance_grade"]}

## Overall Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | {metrics["accuracy"]:.3f} |
| Precision | {metrics["precision"]:.3f} |
| Recall | {metrics["recall"]:.3f} |
| F1-Score | {metrics["f1_score"]:.3f} |

## Per-Intent Performance

| Intent | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
"""
        
        # Add per-class metrics
        per_class = results["per_class_metrics"]
        for intent in self.intent_labels:
            if intent in per_class:
                intent_metrics = per_class[intent]
                report += f"| {intent} | {intent_metrics['precision']:.3f} | {intent_metrics['recall']:.3f} | {intent_metrics['f1-score']:.3f} | {intent_metrics['support']} |\n"
        
        # Error Analysis
        error_analysis = results.get("error_analysis", {})
        if error_analysis:
            report += f"""
## Error Analysis

**Total Errors:** {error_analysis["total_errors"]}  
**Error Rate:** {error_analysis["error_rate"]:.1%}

### Top Confusion Patterns
"""
            top_confusions = error_analysis.get("top_confusions", [])
            for pattern, count in top_confusions:
                report += f"- {pattern}: {count} errors\n"
        
        # Confidence Analysis
        confidence_analysis = results.get("confidence_analysis", {})
        if confidence_analysis:
            stats = confidence_analysis.get("statistics", {})
            report += f"""
## Confidence Score Analysis

**Average Confidence:** {stats.get("mean", 0):.3f}  
**Confidence Standard Deviation:** {stats.get("std", 0):.3f}  
**Confidence-Accuracy Correlation:** {confidence_analysis.get("confidence_correlation", 0):.3f}
"""
        
        return report
    
    def save_results(self, filename: str = None) -> str:
        """Save all evaluation results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluation_results": self.evaluation_results,
                "benchmark_results": self.benchmark_results
            }, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to {output_path}")
        
        return str(output_path)

# Convenience function for quick evaluation
def evaluate_model_performance(
    predictions: List[str],
    true_labels: List[str],
    confidence_scores: List[float] = None,
    model_name: str = "test_model"
) -> Dict[str, Any]:
    """
    Quick function to evaluate model performance.
    
    Args:
        predictions: Predicted intent labels
        true_labels: True intent labels
        confidence_scores: Optional confidence scores
        model_name: Name of the model
        
    Returns:
        Evaluation results
    """
    evaluator = PerformanceEvaluator()
    return evaluator.evaluate_predictions(predictions, true_labels, confidence_scores, model_name)

# Example usage and testing
if __name__ == "__main__":
    print("📊 Testing Performance Evaluator")
    print("=" * 40)
    
    # Create sample test data
    true_labels = ["loan_eligibility", "repayment_terms", "interest_rates", "documentation", "account_management"] * 10
    predictions = ["loan_eligibility", "repayment_terms", "interest_rates", "documentation", "account_management"] * 8 + ["general_inquiry"] * 10
    confidence_scores = [0.9, 0.8, 0.85, 0.75, 0.7] * 10
    
    # Initialize evaluator
    evaluator = PerformanceEvaluator()
    
    # Evaluate performance
    results = evaluator.evaluate_predictions(
        predictions=predictions,
        true_labels=true_labels,
        confidence_scores=confidence_scores,
        model_name="test_bart_model"
    )
    
    # Print results
    print(f"Model: {results['model_name']}")
    print(f"Accuracy: {results['overall_metrics']['accuracy']:.3f}")
    print(f"F1-Score: {results['overall_metrics']['f1_score']:.3f}")
    print(f"Performance Grade: {results['performance_grade']}")
    
    # Generate visualizations
    cm_path = evaluator.visualize_confusion_matrix("test_bart_model")
    if cm_path:
        print(f"Confusion matrix saved to: {cm_path}")
    
    # Generate report
    report_path = evaluator.generate_report("test_bart_model")
    print(f"Report generated: {report_path}")
    
    print("\n✅ Performance evaluation test completed!")
