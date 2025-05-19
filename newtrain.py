import torch
import unsloth
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re

os.environ["DISABLE_TQDM"] = "1"

# Quantization config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# Load model and tokenizer
model_path = "/home/prabhasreddy/Explanation_Generation/DeepSeek-R1-Distill-Qwen-1.5B-unsloth"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=2048,
    # quantization_config=bnb_config,
    dtype=None,
    load_in_4bit=True,
)
tokenizer.padding_side = "right"
EOS_TOKEN = tokenizer.eos_token

# LoRA setup
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# ----------- Helper functions --------------

def clean_text(text):
    return re.sub(r"[^\w\s]", "", str(text).lower())

def extract_keywords(text, top_k=5):
    text = clean_text(text)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_k)
    try:
        X = vectorizer.fit_transform([text])
        return list(vectorizer.get_feature_names_out())
    except:
        return []

prompt_style = """
### Question:
{}
### Response:
{}
"""

def format_function(examples):
    prompts = []
    for uid, iid, uname, rating, review, summary in zip(
        examples["reviewerID"],
        examples["asin"],
        examples["reviewerName"],
        examples["overall"],
        examples["reviewText"],
        examples["summary"]
    ):
        user_features = ", ".join(extract_keywords(review))
        item_features = ", ".join(extract_keywords(summary))
        prompt = (
            f"User {uname} (ID: {uid}) likes {user_features}. "
            f"Item {iid} has {item_features}. "
            f"Explain why user {uname} may like item {iid}."
        )
        prompts.append(prompt)
    return {
        "prompt": prompts,
        "completion": examples["reviewText"]
    }

def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    completions = examples["completion"]
    texts = [
        prompt_style.format(prompt, completion) + EOS_TOKEN
        for prompt, completion in zip(prompts, completions)
    ]
    return {"text": texts}

# ----------- Load and format dataset --------------

train_df = pd.read_csv("/home/prabhasreddy/Explanation_Generation/train_review.csv")
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(format_function, remove_columns=train_df.columns.tolist(), batched=True, batch_size=1000)
train_dataset = train_dataset.map(formatting_prompts_func, remove_columns=["prompt", "completion"], batched=True, batch_size=1000)

val_df = pd.read_csv("/home/prabhasreddy/Explanation_Generation/valid_review.csv")
val_dataset = Dataset.from_pandas(val_df)
val_dataset = val_dataset.map(format_function, remove_columns=val_df.columns.tolist(), batched=True, batch_size=1000)
val_dataset = val_dataset.map(formatting_prompts_func, remove_columns=["prompt", "completion"], batched=True, batch_size=1000)

# ----------- Training Arguments --------------

training_args = TrainingArguments(
    output_dir="/home/prabhasreddy/Explanation_Generation/results",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=500,
    logging_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    bf16=True,
    report_to="none",
    gradient_accumulation_steps=4
)

# ----------- Train the model --------------

last_checkpoint = "/home/prabhasreddy/Explanation_Generation/results/checkpoint-1241"
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=1,
    packing=True,
    args=training_args,
)

trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

# Save model
trainer.save_model("final_model")
tokenizer.save_pretrained("final_model")
