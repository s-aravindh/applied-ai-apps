from model_utils import initialize_model, add_lora_adapters, setup_tokenizer
from data_utils import load_latex_dataset, prepare_dataset
from trainer import setup_trainer
import torch

def main():
    # Initialize model and tokenizer
    model, tokenizer = initialize_model()
    model = add_lora_adapters(model)
    tokenizer = setup_tokenizer(tokenizer)
    
    # Prepare dataset
    dataset = load_latex_dataset(num_samples=100)
    train_dataset = prepare_dataset(dataset, tokenizer)
    
    # Setup and run training
    trainer = setup_trainer(model, tokenizer, train_dataset)
    trainer_stats = trainer.train()
    
    # Save the model
    model.save_pretrained("gemma-3")
    tokenizer.save_pretrained("gemma-3")
    
    # Print training stats
    print(f"Training completed in {trainer_stats.metrics['train_runtime']} seconds")

if __name__ == "__main__":
    main()
