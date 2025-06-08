import torch
import unsloth
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd

import os
# os.environ["DISABLE_TQDM"] = "1"

from transformers import TrainerCallback
import csv

# class CSVLoggerCallback(TrainerCallback):
#     def __init__(self, log_file="training_log.csv"):
#         self.log_file = log_file
#         self.writer = None
#         self.file = None

#     def on_train_begin(self, args, state, control, **kwargs):
#         self.file = open(self.log_file, "w", newline="")
#         self.writer = csv.DictWriter(self.file, fieldnames=["epoch", "step", "loss", "learning_rate"])
#         self.writer.writeheader()

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is None:
#             return
#         row = {
#             "epoch": state.epoch,
#             "step": state.global_step,
#             "loss": logs.get("loss", ""),
#             "learning_rate": logs.get("learning_rate", "")
#         }
#         self.writer.writerow(row)
#         self.file.flush()

#     def on_train_end(self, args, state, control, **kwargs):
#         if self.file:
#             self.file.close()



# Step 1: Configure Quantization for Memory Efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Step 2: Load Model and Tokenizer
model_path = "/home/shyamsg/Explanation_Generation/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=1024,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    token=None,
)

# tokenizer.pad_token = "right"
tokenizer.padding_side = "right"
EOS_TOKEN = tokenizer.eos_token

# Step 3: LoRa Configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# Step 4: Dataset preparation

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
        text = prompt_style.format(prompt, completion) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }

# Load train Dataset
train_df = pd.read_csv("/home/shyamsg/Explanation_Generation/train_review.csv")
train_dataset = Dataset.from_pandas(train_df)

train_dataset = train_dataset.map(
    format_function,
    remove_columns=["userid", "itemid", "userName", "explanation", "rating", "feature", "reviewText"],
    batched=True,
    batch_size=1000
)

train_dataset = train_dataset.map(
    formatting_prompts_func,
    remove_columns=["prompt", "completion"],
    batched=True,
    batch_size=1000
)

#Load evaluation Dataset
val_df = pd.read_csv("/home/shyamsg/Explanation_Generation/valid_review.csv")
val_dataset = Dataset.from_pandas(val_df)
val_dataset = val_dataset.map(
    format_function,
    remove_columns=["userid", "itemid", "userName", "explanation", "rating", "feature", "reviewText"],
    batched=True,
    batch_size=1000
)
val_dataset = val_dataset.map(
    formatting_prompts_func,
    remove_columns=["prompt", "completion"],
    batched=True,
    batch_size=1000
)


# Print dataset sizes and samples
print("Train size:", len(train_dataset))
print("Test size:", len(val_dataset))

# # Step 5: Training Arguments
# training_args = TrainingArguments(
#     output_dir="/home/shyamsg/Explanation_Generation/results",
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=3,
#     learning_rate=2e-5,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir="./logs",
#     logging_steps=10,
#     save_total_limit=2,
#     fp16=True,
#     report_to="none",
#     gradient_accumulation_steps=1,
#     disable_tqdm=True, 
# )
# Step 5: Training Arguments

# last_checkpoint = "/home/shyamsg/Explanation_Generation/results/checkpoint-1248"

training_args = TrainingArguments(
    output_dir="/home/shyamsg/Explanation_Generation/results_19th",
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    num_train_epochs=10,
    learning_rate=2e-5,
    eval_strategy="epoch",      # ✅ evaluate once per epoch
    save_strategy="steps",            # ✅ save once per epoch
    save_steps=500,               # ✅ save every 1000 ste
    logging_strategy="epoch",         # ✅ log once per epoch
    # logging_dir="./logs",
    save_total_limit=3,
    fp16=False,
    bf16=True,
    report_to="none",  # ✅ log to CSV
    # gradient_accumulation_steps=2,
    # disable_tqdm=True                 # ✅ no progress bars
)

# Step 6: Training the model
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    dataset_num_proc=1,
    packing=True,
    args=training_args
    # callbacks=[CSVLoggerCallback()],
)
# Train the model
# trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)
trainer_stats = trainer.train()

# Step 7: Save Model
trainer.save_model("final_model")
tokenizer.save_pretrained("final_model")