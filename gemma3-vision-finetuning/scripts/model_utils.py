from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from config import ModelConfig, LoRAConfig, ChatConfig

def initialize_model(model_name=ModelConfig.NAME, max_seq_length=ModelConfig.MAX_SEQ_LENGTH):
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=ModelConfig.LOAD_IN_4BIT,
        load_in_8bit=ModelConfig.LOAD_IN_8BIT,
        full_finetuning=ModelConfig.FULL_FINETUNING,
    )
    return model, tokenizer

def add_lora_adapters(model):
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=LoRAConfig.VISION_LAYERS,
        finetune_language_layers=LoRAConfig.LANGUAGE_LAYERS,
        finetune_attention_modules=LoRAConfig.ATTENTION_MODULES,
        finetune_mlp_modules=LoRAConfig.MLP_MODULES,
        r=LoRAConfig.R,
        lora_alpha=LoRAConfig.ALPHA,
        lora_dropout=LoRAConfig.DROPOUT,
        bias=LoRAConfig.BIAS,
        random_state=LoRAConfig.RANDOM_STATE,
    )
    return model

def setup_tokenizer(tokenizer):
    return get_chat_template(
        tokenizer,
        chat_template=ChatConfig.TEMPLATE,
    )
