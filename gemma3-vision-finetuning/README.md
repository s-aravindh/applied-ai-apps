# Gemma-3 Vision Model Fine-tuning

This project implements fine-tuning of the Gemma-3 model

## Setup

1. Install dependencies:
```bash
pip install -U unsloth vllm transformers
```


### Running Training

```bash
python scripts/main.py
```

This will:
- Initialize the model and tokenizer
- Prepare the dataset
- Run the training process
- Save the fine-tuned model

## Model Output

The model takes an image as input and generates LaTeX code representing the mathematical expression in the image.

Example usage:
```python
from unsloth import FastModel
import torch

# Load model
model, tokenizer = FastModel.from_pretrained(
    "gemma-3-finetune",
    max_seq_length=2048,
    load_in_4bit=True
)

# Generate LaTeX code
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Write the LaTeX representation for this image"}
    ]
}]
```

## Model Saving

The fine-tuned model can be saved in different formats:

1. LoRA adapters:
```python
model.save_pretrained("gemma-3")
tokenizer.save_pretrained("gemma-3")
```

2. Float16 for VLLM:
```python
model.save_pretrained_merged("gemma-3-finetune", tokenizer)
```

3. GGUF format:
```python
model.save_pretrained_gguf(
    "gemma-3-finetune",
    quantization_type="Q8_0"
)
```
