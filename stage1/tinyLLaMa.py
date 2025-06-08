import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from tqdm import tqdm

assert torch.cuda.is_available(), "GPU not available. Please check CUDA setup."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def build_sft_dataset(dataset_path, max_length=1024):
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Build prompt and keep explanation separate
    def create_prompt(row):
        return (
            f"User {row['userName']} (ID: {row['userid']}) rated the product "
            f"(ID: {row['itemid']}) {row['rating']}/5. The product is related to features: {row['feature']}. "
            f"Write a detailed explanation for this product, highlighting aspects related to {row['feature']}."
        )

    df["prompt"] = df.apply(create_prompt, axis=1)
    dataset = Dataset.from_pandas(df[["prompt", "reviewText"]])

    # Tokenization with loss masking
    def tokenize(row):
        prompt = row["prompt"]
        explanation = row["reviewText"]

        # Encode separately without special tokens
        prompt_enc = tokenizer(prompt, add_special_tokens=False)
        explanation_enc = tokenizer(explanation, add_special_tokens=False)

        input_ids = prompt_enc["input_ids"] + explanation_enc["input_ids"]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_enc["input_ids"]) + explanation_enc["input_ids"]

        # Truncate to max_length
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

        # Pad to max_length
        pad_len = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Map without batching and format correctly
    dataset = dataset.map(tokenize, batched=False, load_from_cache_file=False)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("Dataset keys:", dataset[0].keys())
    print(f"Dataset size: {len(dataset)}")
    print("Sample input_ids length:", len(dataset[0]['input_ids']))

    return dataset, tokenizer

train_dataset_path = "/raid/home/shyamsg/Explanation_generation/train_review.csv"
val_dataset_path = "/raid/home/shyamsg/Explanation_generation/val_review.csv"

train_sft_dataset, tokenizer = build_sft_dataset(train_dataset_path)
val_dataset, _ = build_sft_dataset(val_dataset_path)

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
print(f"Model device: {model.device}")

training_args = TrainingArguments(
    output_dir="./tinyllama-sft-product-explanations-22nd",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    logging_dir="./sft_logs",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_sft_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Testing
model.eval()

sample_row = pd.read_csv(val_dataset_path).iloc[9425]
prompt = (
    f"User {sample_row['userName']} (ID: {sample_row['userid']}) rated the product "
    f"(ID: {sample_row['itemid']}) {sample_row['rating']}/5. The product is related to features: {sample_row['feature']}. "
    f"Write a detailed explanation for this product, highlighting aspects related to {sample_row['feature']}."
)

input_ids = tokenizer(prompt, return_tensors="pt").to(device)

output = model.generate(
    **input_ids,
    max_new_tokens=128,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    top_p=0.9,
    temperature=0.5,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True,
    no_repeat_ngram_size=3,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
generated_explanation = generated_text.replace(prompt, "").strip()
ground_truth = sample_row["reviewText"]

print("Prompt:")
print(prompt)
print("\nGenerated Explanation:")
print(generated_explanation)
print("\nGround Truth Explanation:")
print(ground_truth)