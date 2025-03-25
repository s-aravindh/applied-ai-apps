class ModelConfig:
    NAME = "unsloth/gemma-3-4b-it"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True
    LOAD_IN_8BIT = False
    FULL_FINETUNING = False

class LoRAConfig:
    R = 8
    ALPHA = 8
    DROPOUT = 0
    BIAS = "none"
    RANDOM_STATE = 3407
    VISION_LAYERS = False
    LANGUAGE_LAYERS = True
    ATTENTION_MODULES = True
    MLP_MODULES = True

class DataConfig:
    DATASET_NAME = "unsloth/LaTeX_OCR"
    NUM_SAMPLES = 100
    INSTRUCTION = "Write the LaTeX representation for this image."

class TrainerConfig:
    MAX_STEPS = 30
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 4
    WARMUP_STEPS = 5
    LEARNING_RATE = 2e-4
    LOGGING_STEPS = 1
    OPTIM = "adamw_8bit"
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER = "linear"
    SEED = 3407
    REPORT_TO = "none"

class ChatConfig:
    TEMPLATE = "gemma-3"
    INSTRUCTION_PART = "<start_of_turn>user\n"
    RESPONSE_PART = "<start_of_turn>model\n"

class GenerationConfig:
    MAX_NEW_TOKENS = 64
    TEMPERATURE = 1.0
    TOP_P = 0.95
    TOP_K = 64
