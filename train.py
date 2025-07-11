from datasets import load_dataset
from colorama import Fore
import datetime
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

print(f"[{datetime.datetime.now()}] Starting training script...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print(f"\n[{datetime.datetime.now()}] Loading dataset...")
dataset = load_dataset("data", split='train')
print(f"Dataset loaded with {len(dataset)} examples")
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET) 

def format_chat_template(batch, tokenizer):

    system_prompt =  """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

    samples = []

    # Access the inputs from the batch
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]

        # Apply chat template and append the result to the list
        # Qwen uses a different chat template, let tokenizer handle it
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    # Return a dictionary with lists as expected for batched processing
    return {
        "instruction": questions,
        "response": answers,
        "text": samples  # The processed chat template text for each row
    }

base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
print(f"\n[{datetime.datetime.now()}] Loading tokenizer from {base_model}...")
tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
        token=None,  # Will use HF_TOKEN env variable
)
print(f"Tokenizer loaded successfully")

print(f"\n[{datetime.datetime.now()}] Formatting dataset with chat template...")
train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=8, batched=True, batch_size=10)
print(f"Dataset formatted. Total examples: {len(train_dataset)}")
print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET) 


print(f"\n[{datetime.datetime.now()}] Setting up quantization configuration...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"\n[{datetime.datetime.now()}] Loading model {base_model} with 4-bit quantization...")
print(f"This may take a few minutes...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="cuda:0",
    quantization_config=quant_config,
    token=os.environ.get("HF_TOKEN"),  # Use environment variable
    cache_dir="./workspace",
)
print(f"Model loaded successfully!")

print(f"\n[{datetime.datetime.now()}] Preparing model for training...")
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print(f"Model prepared for k-bit training")

print(f"\n[{datetime.datetime.now()}] Setting up LoRA configuration...")
peft_config = LoraConfig(
    r=256,
    lora_alpha=512,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
print(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

print(f"\n[{datetime.datetime.now()}] Initializing trainer...")
trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir="Qwen2.5-Coder-7B-LoRA",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        save_steps=50,
        gradient_checkpointing=True,
        report_to="none",
        fp16=True,
    ),
    peft_config=peft_config,
)
print(f"Trainer initialized successfully")
print(f"Training parameters:")
print(f"  - Batch size: 1")
print(f"  - Gradient accumulation: 4")
print(f"  - Max steps: 100")
print(f"  - Learning rate: 2e-4")
print(f"  - Save steps: 50")

print(f"\n[{datetime.datetime.now()}] Starting training...")
print(f"=" * 50)
trainer.train()
print(f"=" * 50)
print(f"\n[{datetime.datetime.now()}] Training completed!")

print(f"\n[{datetime.datetime.now()}] Saving complete checkpoint...")
trainer.save_model('complete_checkpoint')
print(f"Complete checkpoint saved to: complete_checkpoint/")

print(f"\n[{datetime.datetime.now()}] Saving final model...")
trainer.model.save_pretrained("final_model")
print(f"Final model saved to: final_model/")

print(f"\n[{datetime.datetime.now()}] All done! 🎉")