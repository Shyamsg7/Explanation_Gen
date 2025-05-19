# Explanation Generation Recommender System

This project focuses on generating detailed, explainable product reviews using large language models (LLMs) with efficient fine-tuning and quantization techniques. The workflow includes data preparation, model training (with LoRA and quantization), and evaluation using standard NLP metrics.


## File Descriptions

- **Dataprep.ipynb**  
  Jupyter notebook for data preprocessing. It loads raw review data, processes and cleans it, maps user/item IDs, and prepares CSV files for training, validation, and testing. It also demonstrates how to enrich review data with product metadata.

- **notebook6.ipynb**  
  Example notebook for training a language model using the Unsloth library with quantization (BitsAndBytesConfig) and LoRA. It covers dataset formatting, prompt engineering, and model saving.

- **FullModel.py**  
  Script for full-precision model training using the DeepSeek-R1-Distill-Qwen-1.5B model. It includes advanced LoRA configuration, Flash Attention integration, and optimized training arguments for efficient large-batch training.

- **QuantizedModel.py**  
  Script for training with quantized models (4-bit), using LoRA for parameter-efficient fine-tuning. It prepares the dataset, configures quantization, and saves the final model.

- **newtrain.py**  
  Alternative training script that uses TF-IDF keyword extraction to enhance prompt construction. It demonstrates custom prompt engineering and supports resuming training from checkpoints.

- **Testing.py**  
  Script for evaluating the trained model. It generates reviews for the test set, saves outputs, and computes BLEU and ROUGE scores to assess generation quality.

## Workflow

1. **Data Preparation**  
   Use `Dataprep.ipynb` to process raw review data and product metadata, producing clean CSVs for model training and evaluation.

2. **Model Training**  
   - For full-precision training, use [`FullModel.py`](FullModel.py).
   - For quantized/efficient training, use [`QuantizedModel.py`](QuantizedModel.py) or [`newtrain.py`](newtrain.py).
   - Notebooks like [`notebook6.ipynb`](notebook6.ipynb) provide interactive examples.

3. **Evaluation**  
   Use [`Testing.py`](Testing.py) to generate reviews on the test set and compute evaluation metrics.

## Requirements

- Python 3.8+
- PyTorch
- [Unsloth](https://github.com/unslothai/unsloth)
- HuggingFace Transformers, Datasets, TRL
- pandas, scikit-learn, numpy
- nltk, rouge-score
- tqdm

Install dependencies with:
```sh
pip install torch unsloth transformers datasets trl pandas scikit-learn nltk rouge-score tqdm
```

## Notes

- Model checkpoints and data paths are hardcoded; update them as needed for your environment.
- For large-scale training, ensure sufficient GPU memory and adjust batch sizes accordingly.
---

**Author:**  
Shyam Sundar Gowda