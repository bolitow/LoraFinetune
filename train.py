from datasets import load_dataset
from colorama import Fore
import datetime
import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
import torch
from huggingface_hub import HfApi, create_repo

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

base_model = "../models/qwen2.5_coder_7b_instruct"
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
        save_strategy="no",  # Disable checkpoint saving during training
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

# Get HuggingFace username
hf_username = os.environ.get("HF_USERNAME")
if not hf_username:
    print(f"WARNING: HF_USERNAME not set. Will not publish to HuggingFace.")
    print(f"To publish, set: export HF_USERNAME='your-huggingface-username'")

# Prepare repository name
repo_name = f"qwen2.5-coder-7b-lora-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
full_repo_id = f"{hf_username}/{repo_name}" if hf_username else None

print(f"\n[{datetime.datetime.now()}] Merging LoRA weights with base model...")
print(f"This will create a complete model with all weights merged...")

# Get the base model without quantization for merging
print(f"Loading base model without quantization...")
base_model_for_merge = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),
)

# Save LoRA adapter first
print(f"Saving LoRA adapter...")
trainer.model.save_pretrained("temp_lora_adapter")

# Merge LoRA weights
print(f"Merging LoRA weights...")
merged_model = PeftModel.from_pretrained(base_model_for_merge, "temp_lora_adapter")
merged_model = merged_model.merge_and_unload()

# Clean up temporary adapter
shutil.rmtree("temp_lora_adapter")

# Save the complete merged model locally
print(f"\n[{datetime.datetime.now()}] Saving merged model locally...")
merged_model.save_pretrained("final_merged_model")
tokenizer.save_pretrained("final_merged_model")
print(f"Complete merged model saved to: final_merged_model/")

# Publish to HuggingFace if username is set
if hf_username:
    print(f"\n[{datetime.datetime.now()}] Publishing to HuggingFace...")
    print(f"Repository: {full_repo_id}")
    
    try:
        # Create repository
        api = HfApi()
        create_repo(
            repo_id=full_repo_id,
            token=os.environ.get("HF_TOKEN"),
            private=True,  # Make it private by default
            exist_ok=True
        )
        
        # Push model and tokenizer
        merged_model.push_to_hub(
            full_repo_id,
            token=os.environ.get("HF_TOKEN"),
            commit_message="Initial model upload"
        )
        tokenizer.push_to_hub(
            full_repo_id,
            token=os.environ.get("HF_TOKEN"),
            commit_message="Add tokenizer"
        )
        
        print(f"✓ Model successfully published to: https://huggingface.co/{full_repo_id}")
        
        # Create model card
        model_card_content = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
- generated_from_trainer
- lora
- qwen
- code
datasets:
- custom
---

# {repo_name}

This model is a fine-tuned version of [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) using LoRA.

## Training Details

- **Base model**: Qwen2.5-Coder-7B-Instruct
- **Training method**: LoRA (r=256, alpha=512)
- **Training date**: {datetime.datetime.now().strftime('%Y-%m-%d')}
- **Framework**: transformers, peft, trl

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{full_repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{full_repo_id}")

# Use the model
messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Your question here"}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```
"""
        
        # Save and push model card
        with open("final_merged_model/README.md", "w") as f:
            f.write(model_card_content)
        
        api.upload_file(
            path_or_fileobj="final_merged_model/README.md",
            path_in_repo="README.md",
            repo_id=full_repo_id,
            token=os.environ.get("HF_TOKEN"),
            commit_message="Add model card"
        )
        
    except Exception as e:
        print(f"✗ Error publishing to HuggingFace: {e}")
        print(f"Model is saved locally in: final_merged_model/")

# Clean up checkpoint directory if it exists
if os.path.exists("Qwen2.5-Coder-7B-LoRA"):
    print(f"\n[{datetime.datetime.now()}] Cleaning up checkpoint directory...")
    shutil.rmtree("Qwen2.5-Coder-7B-LoRA")
    print(f"✓ Checkpoint directory removed")

print(f"\n[{datetime.datetime.now()}] All done! 🎉")
print(f"\nSummary:")
print(f"- Merged model saved locally: final_merged_model/")
if full_repo_id:
    print(f"- Published to HuggingFace: https://huggingface.co/{full_repo_id}")
else:
    print(f"- To publish to HuggingFace, set HF_USERNAME and run again")