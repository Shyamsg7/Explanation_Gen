import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import csv

# Verify CUDA availability
assert torch.cuda.is_available(), "GPU not available. Please check CUDA setup."

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load the trained model and tokenizer
model_path = "./gpt2-final-model"  # Path from SFT training
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
tokenizer.padding_side = "right"

# Enable inference mode
model.eval()

# Step 2: Load and prepare test dataset
prompt_style = """
### Question:
{}
### Response:
{}
"""

def format_function(examples):
    """
    Format dataset to create prompts and retain ground truth explanations.
    
    Args:
        examples: Dataset examples
    
    Returns:
        Dictionary with prompts and completions
    """
    prompts = []
    for uid, iid, uname, rating, feature in zip(
        examples["userid"],
        examples["itemid"],
        examples["userName"],
        examples["rating"],
        examples["feature"]
    ):
        prompt = (
            f"User {uname} (ID: {uid}) rated the product ID: {iid} {rating}/5. "
            f"The product is related to features: {feature}. "
            f"Write a detailed review for this product, highlighting aspects related to {feature}."
        )
        prompts.append(prompt)
    return {
        "prompt": prompts,
        "completion": examples["explanation"]  # Assuming reviewText is the explanation column
    }

def formatting_prompts_func(examples):
    """
    Format prompts with the prompt style for inference.
    
    Args:
        examples: Dataset examples with prompts and completions
    
    Returns:
        Dictionary with formatted text and ground truth
    """
    prompts = examples["prompt"]
    completions = examples["completion"]
    texts = []
    for prompt, completion in zip(prompts, completions):
        text = prompt_style.format(prompt, "")  # Empty response for inference
        texts.append(text)
    return {
        "text": texts,
        "ground_truth": completions
    }

# Load test dataset
test_df = pd.read_csv("/home/shyamsg/Explanation_Generation/test_review.csv")
test_dataset = Dataset.from_pandas(test_df)

# Process test dataset
test_dataset = test_dataset.map(
    format_function,
    remove_columns=["userid", "itemid", "userName", "rating", "feature", "explanation"],
    batched=True,
    batch_size=1000
)

test_dataset = test_dataset.map(
    formatting_prompts_func,
    remove_columns=["prompt", "completion"],
    batched=True,
    batch_size=1000
)

print("Test dataset size:", len(test_dataset))

# Step 3: Generate reviews and save incrementally
def generate_review(prompt):
    """
    Generate a review for the given prompt using the GPT-2 model.
    
    Args:
        prompt: Input prompt string
    
    Returns:
        Generated review text
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract response part after ### Response:
    response_start = generated_text.find("### Response:") + len("### Response:")
    return generated_text[response_start:].strip()

generated_reviews = []
ground_truths = []
total_samples = len(test_dataset)

for i, sample in enumerate(tqdm(test_dataset, desc="Generating reviews")):
    prompt = sample["text"]
    ground_truth = sample["ground_truth"]
    generated_review = generate_review(prompt)
    
    # Append to lists for later evaluation
    generated_reviews.append(generated_review)
    ground_truths.append(ground_truth)
    
    if (i + 1) % 100 == 0:  # Print every 100 samples
        print(f"Processed {i + 1}/{total_samples} samples")
print(f"Completed processing {total_samples}/{total_samples} samples")

# Step 4: Save final results
results = pd.DataFrame({
    "Generated_Review": generated_reviews,
    "Ground_Truth": ground_truths,
})
results.to_csv("/home/shyamsg/Explanation_Generation/evaluation_results_gpt2.csv", index=False)
print("Final results saved to evaluation_results_gpt2.csv")