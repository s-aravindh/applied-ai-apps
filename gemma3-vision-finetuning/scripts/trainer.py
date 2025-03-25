from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from config import TrainerConfig, ChatConfig

def setup_trainer(model, tokenizer, train_dataset, max_steps=TrainerConfig.MAX_STEPS):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=TrainerConfig.BATCH_SIZE,
            gradient_accumulation_steps=TrainerConfig.GRAD_ACCUM_STEPS,
            warmup_steps=TrainerConfig.WARMUP_STEPS,
            max_steps=max_steps,
            learning_rate=TrainerConfig.LEARNING_RATE,
            logging_steps=TrainerConfig.LOGGING_STEPS,
            optim=TrainerConfig.OPTIM,
            weight_decay=TrainerConfig.WEIGHT_DECAY,
            lr_scheduler_type=TrainerConfig.LR_SCHEDULER,
            seed=TrainerConfig.SEED,
            report_to=TrainerConfig.REPORT_TO,
        ),
    )
    
    return train_on_responses_only(
        trainer,
        instruction_part=ChatConfig.INSTRUCTION_PART,
        response_part=ChatConfig.RESPONSE_PART,
    )
