import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig
import numpy as np
from tqdm import tqdm
import os
import csv

# Disable tqdm for consistency with training script
# os.environ["DISABLE_TQDM"] = "1"

# Step 1: Configure Quantization (same as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Step 2: Load the trained model and tokenizer
model_path = "/home/shyamsg/final_model"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=1024,
    quantization_config=bnb_config,
    dtype=None,
    load_in_4bit=True,
)
tokenizer.padding_side = "right"
EOS_TOKEN = tokenizer.eos_token

# Enable inference mode
FastLanguageModel.for_inference(model)

# Step 3: Load and prepare test dataset
prompt_style = """
### Question:
{}
### Response:
{}
"""

def format_function(examples):
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
            "Write a detailed review for this product, highlighting aspects related to "
            f"{feature}."
        )
        prompts.append(prompt)
    return {
        "prompt": prompts,
        "completion": examples["explanation"]
    }

def formatting_prompts_func(examples):
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
    remove_columns=["userid", "itemid", "userName", "rating", "feature", "reviewText"],
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

# Step 4: Generate reviews and save incrementally
def generate_review(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract response part after ### Response:
    response_start = generated_text.find("### Response:") + len("### Response:")
    return generated_text[response_start:].strip()

output_file = "/home/shyamsg/Explanation_Generation/generated_reviews.csv"
# Initialize CSV file with headers
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Sample_Index", "Generated_Review", "Ground_Truth"])

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
    
    # Write to CSV immediately
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([i, generated_review, ground_truth])
    
    if (i + 1) % 100 == 0:  # Print every 100 samples
        print(f"Processed {i + 1}/{total_samples} samples")
print(f"Completed processing {total_samples}/{total_samples} samples")


# Step 7: Save final results with results
results = pd.DataFrame({
    "Generated_Review": generated_reviews,
    "Ground_Truth": ground_truths,
})
results.to_csv("/home/shyamsg/Explanation_Generation/evaluation_results.csv", index=False)
print("Final results saved to evaluation_results.csv")