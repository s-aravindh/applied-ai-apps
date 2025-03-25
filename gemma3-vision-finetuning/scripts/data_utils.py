from datasets import load_dataset
from config import DataConfig

def load_latex_dataset(num_samples=DataConfig.NUM_SAMPLES):
    dataset = load_dataset(DataConfig.DATASET_NAME, split="train")
    return dataset.select(range(num_samples))

def convert_to_conversation(sample, instruction=DataConfig.INSTRUCTION):
    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": instruction},
             {"type": "image", "image": sample["image"]}]
        },
        {"role": "assistant",
         "content": [
             {"type": "text", "text": sample["text"]}]
        },
    ]
    return {"messages": conversation}

def apply_chat_template(examples, tokenizer):
    texts = tokenizer.apply_chat_template(examples["messages"])
    return {"text": texts}

def prepare_dataset(dataset, tokenizer):
    converted_dataset = dataset.map(convert_to_conversation)
    return converted_dataset.map(
        lambda x: apply_chat_template(x, tokenizer), 
        batched=True
    )
