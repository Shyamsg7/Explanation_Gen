import torch
import unsloth
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
import os

os.environ["DISABLE_TQDM"] = "1"
torch.backends.cuda.enable_flash_sdp(True)  # Enable Flash Attention

# Configuration Constants
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = "/home/prabhasreddy/Explanation_Generation/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH = "/home/prabhasreddy/Explanation_Generation/test_review_small.csv"
OUTPUT_DIR = "/home/prabhasreddy/Explanation_Generation/results_fake"
MAX_SEQ_LENGTH = 2048

# 1. Model Loading with Full Precision
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path = MODEL_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = torch.bfloat16,  # Full precision
    load_in_4bit = False,     # Disable quantization
    token = None,
)

tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token

# 2. Enhanced LoRA Configuration
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,                   # Increased from 8
    lora_alpha = 64,          # Increased from 16
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
)

# 3. Flash Attention Integration
model, tokenizer = FastLanguageModel.postprocess(
    model = model,
    tokenizer = tokenizer,
    use_flash_attention = True,  # Critical for speed
    use_original_flash_attention = True,
)

# 4. Dataset Preparation
def format_function(examples):
    return {
        "prompt": [f"User {uid} rated Item {iid} {rat}/5. Generate detailed review:"
                  for uid, iid, rat in zip(examples["userid"], examples["itemid"], examples["rating"])],
        "completion": examples["reviewText"]
    }

def formatting_prompts_func(examples):
    return {
        "text": [f"### Question:\n{p}\n\n### Response:\n{c}{EOS_TOKEN}"
                for p, c in zip(examples["prompt"], examples["completion"])]
    }

# Load and process dataset
df = pd.read_csv(DATASET_PATH)
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.02)
dataset = dataset.map(format_function, batched=True, batch_size=1000, 
                     remove_columns=["userid", "itemid", "rating", "reviewText"])
dataset = dataset.map(formatting_prompts_func, batched=True, batch_size=1000,
                     remove_columns=["prompt", "completion"])

# 5. Optimized Training Arguments
training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    per_device_train_batch_size = 16,  # 8x increase
    per_device_eval_batch_size = 16,
    gradient_accumulation_steps = 2,    # Better memory utilization
    num_train_epochs = 3,
    learning_rate = 5e-5,              # Increased from 2e-5
    optim = "adamw_bnb_8bit",          # Memory-efficient optimizer
    warmup_ratio = 0.1,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    logging_strategy = "epoch",
    bf16 = True,                       # Use bfloat16
    fp16 = False,
    save_total_limit = 2,
    report_to = "none",
)

# 6. Trainer Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    packing = True,
    args = training_args,
)

# 7. Training Execution
trainer.train()

# 8. Model Saving
trainer.save_model("final_model_fake")
tokenizer.save_pretrained("final_model_fake")
