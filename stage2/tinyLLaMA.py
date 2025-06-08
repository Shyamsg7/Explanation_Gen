import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# Verify CUDA availability
assert torch.cuda.is_available(), "GPU not available. Please check CUDA setup."

# Set device for GPU flexibility
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration for PPO
config = PPOConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    learning_rate=1.41e-5,
    mini_batch_size=8,
    batch_size=32,
)

# Generation settings
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": None  # Set after tokenizer initialization
}

def build_dataset(config, dataset_path, input_min_text_length=10, input_max_text_length=128):
    """
    Build and tokenize dataset from CSV file for PPO training.
    
    Args:
        config: PPOConfig object
        dataset_path: Path to CSV dataset
        input_min_text_length: Minimum length for input text
        input_max_text_length: Maximum length for input text
    
    Returns:
        Tokenized dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    # Load CSV
    df = pd.read_csv(dataset_path)

    # Create prompts
    def create_prompt(row):
        return (
            f"User {row['userName']} (ID: {row['userid']}) rated the product "
            f"(ID: {row['itemid']}) {row['rating']}/5. The product is related to features: {row['feature']}. "
            f"Write a detailed explanation for this product, highlighting aspects related to {row['feature']}."
        )

    df["prompt"] = df.apply(create_prompt, axis=1)
    dataset = Dataset.from_pandas(df[["prompt", "explanation"]])

    # Tokenize with fixed max_length
    max_length = input_max_text_length
    def tokenize(sample):
        encoded = tokenizer(
            sample["prompt"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True
        )
        sample["input_ids"] = encoded["input_ids"]
        sample["attention_mask"] = encoded["attention_mask"]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["ground_truth"] = sample.get("explanation", "")
        if "explanation" not in sample:
            print("Warning: missing explanation in sample:", sample)
        return sample

    dataset = dataset.map(tokenize, batched=False, load_from_cache_file=False)
    return dataset

# Load dataset
dataset_path = "/data2/home/shyamsg/Final_Project/notebooks/train_data.csv"
dataset = build_dataset(config, dataset_path)

# Verify dataset
print(f"Dataset size: {len(dataset)}")
print(dataset[0])

# Data collator
def collator(data):
    """
    Collate data for PPO training.
    
    Args:
        data: List of dataset samples
    
    Returns:
        Dictionary with input_ids, attention_mask, query, and ground_truth
    """
    return {
        "input_ids": torch.tensor([d["input_ids"] for d in data], device=device),
        "attention_mask": torch.tensor([d["attention_mask"] for d in data], device=device),
        "query": [d["query"] for d in data],
        "ground_truth": [d.get("ground_truth", "") for d in data]
    }

# Load models
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(device)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = "[PAD]"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

# Initialize PPOTrainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator
)

# Verify batch
batch = next(iter(ppo_trainer.dataloader))
print(batch.keys())

# Load reward model
reward_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
print(f"Reward model device: {reward_model.device}")

# Test reward model
embeddings = reward_model.encode(["testing", "test"], convert_to_tensor=True, device=device)
print(embeddings.shape)

# Reward function: Cosine similarity
def compute_reward(generated_text, ground_truth):
    """
    Compute reward based on cosine similarity between generated text and ground truth.
    
    Args:
        generated_text: Generated explanation
        ground_truth: Reference explanation
    
    Returns:
        Reward as a tensor
    """
    embeddings = reward_model.encode([generated_text, ground_truth], convert_to_tensor=True, device=device)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return torch.tensor(similarity, dtype=torch.float32, device=device)

# Training loop
output_min_length = 20
output_max_length = 100
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    attention_masks = batch["attention_mask"]

    # Generate responses
    response_tensors = []
    for query, mask in zip(query_tensors, attention_masks):
        gen_len = output_length_sampler()
        gen_kwargs["max_new_tokens"] = gen_len
        query_response = ppo_trainer.generate(
            query,
            attention_mask=mask.unsqueeze(0),
            **gen_kwargs
        ).squeeze()
        response_len = len(query_response) - len(query)
        response_tensors.append(query_response[-response_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute rewards
    rewards = []
    for query, response, ground_truth in zip(batch["query"], batch["response"], batch["ground_truth"]):
        reward = compute_reward(response, ground_truth)
        rewards.append(reward)

    # Run PPO step
    stats = ppo_trainer.step(
        [q for q in query_tensors],
        [r for r in response_tensors],
        rewards
    )