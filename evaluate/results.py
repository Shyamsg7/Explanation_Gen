import pandas as pd
from utils import bleu_score, rouge_score, mauve_score
import csv

def evaluate_from_csv(input_csv, output_csv):
    """
    Load generated reviews from CSV, compute BLEU-4, ROUGE-1, ROUGE-2, and MAUVE scores
    using functions from utils.py, and save results to a new CSV.
    
    Args:
        input_csv: Path to input CSV with 'Generated_Review' and 'Ground_Truth' columns
        output_csv: Path to output CSV for saving scores
    """
    # Load CSV
    df = pd.read_csv(input_csv)
    generated_reviews = df["Generated_Review"].tolist()
    ground_truths = df["Ground_Truth"].tolist()
    
    print(f"Loaded {len(generated_reviews)} samples from {input_csv}")

    # Compute BLEU-4 score
    bleu_4 = bleu_score(ground_truths, generated_reviews, n_gram=4, smooth=False)
    
    # Compute ROUGE scores
    rouge_results = rouge_score(ground_truths, generated_reviews)
    rouge_1 = rouge_results["rouge_1/f_score"]
    rouge_2 = rouge_results["rouge_2/f_score"]
    rouge_l = rouge_results["rouge_l/f_score"]
    
    # Compute MAUVE score
    mauve_s = mauve_score(ground_truths, generated_reviews)
    
    # Print results
    print("\nEvaluation Metrics:")
    print(f"Average BLEU-4 Score: {bleu_4:.4f}")
    print(f"Average ROUGE-1 Score: {rouge_1:.4f}")
    print(f"Average ROUGE-2 Score: {rouge_2:.4f}")
    print(f"Average ROUGE-L Score: {rouge_l:.4f}")
    print(f"MAUVE Score: {mauve_s:.4f}")
    
    # Save results to CSV
    results = pd.DataFrame({
        "Metric": ["BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "MAUVE"],
        "Score": [bleu_4, rouge_1, rouge_2, rouge_l, mauve_s]
    })
    results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "/home/shyamsg/Explanation_Generation/evaluation_results.csv"
    output_csv = "/home/shyamsg/Explanation_Generation/metric_scores.csv"
    evaluate_from_csv(input_csv, output_csv)