"""Test model evaluation system with sample data."""
import sys
import numpy as np
sys.path.insert(0, ".")

from src.model_evaluation import ModelEvaluator

# Generate sample predictions
np.random.seed(42)
n_samples = 200

y_true = np.random.randint(0, 2, n_samples)

# Simulate different model predictions
models_data = [
    {
        "name": "Xception",
        "type": "image",
        "y_true": y_true.tolist(),
        "y_pred": np.clip(y_true * 0.85 + np.random.normal(0, 0.15, n_samples), 0, 1).tolist(),
        "inference_times": np.random.normal(0.05, 0.01, n_samples).tolist(),
    },
    {
        "name": "EfficientNet-B4",
        "type": "image",
        "y_true": y_true.tolist(),
        "y_pred": np.clip(y_true * 0.90 + np.random.normal(0, 0.12, n_samples), 0, 1).tolist(),
        "inference_times": np.random.normal(0.04, 0.008, n_samples).tolist(),
    },
    {
        "name": "Audio-EfficientNet",
        "type": "audio",
        "y_true": y_true.tolist(),
        "y_pred": np.clip(y_true * 0.75 + np.random.normal(0, 0.20, n_samples), 0, 1).tolist(),
        "inference_times": np.random.normal(0.03, 0.005, n_samples).tolist(),
    },
    {
        "name": "Fusion-Bayesian",
        "type": "fusion",
        "y_true": y_true.tolist(),
        "y_pred": np.clip(y_true * 0.92 + np.random.normal(0, 0.10, n_samples), 0, 1).tolist(),
        "inference_times": np.random.normal(0.08, 0.02, n_samples).tolist(),
    },
]

# Run evaluation
evaluator = ModelEvaluator()
comp = evaluator.evaluate_all_models(models_data, dataset_name="test_dataset")

# Print summary
evaluator.print_summary(comp)

# Save results
output_path = evaluator.save_results(comp)
print(f"Results saved to: {output_path}")
