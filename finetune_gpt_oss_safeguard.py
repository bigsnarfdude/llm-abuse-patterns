#!/usr/bin/env python3
"""
Fine-tune GPT-OSS Safeguard with Unsloth QLoRA
-----------------------------------------------
Fine-tunes OpenAI's GPT-OSS Safeguard 20B model on JailbreakHub dataset
using QLoRA (4-bit quantization) for efficient training on 16GB VRAM.

Hardware Requirements:
- GPU: 16GB VRAM (e.g., RTX 4070 Ti)
- RAM: 32GB+ recommended
- CUDA 12.x

Expected Training Time:
- 320 examples, 3 epochs: ~2-3 hours on RTX 4070 Ti
"""

import os
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

print("=" * 80)
print("GPT-OSS SAFEGUARD FINE-TUNING WITH UNSLOTH")
print("=" * 80)
print()

# Configuration
MODEL_NAME = "openai/gpt-oss-20b"  # Base model (not pre-quantized)
MAX_SEQ_LENGTH = 4096  # GPT-OSS supports 128k but we'll use 4k for efficiency
LOAD_IN_4BIT = True    # Apply 4-bit quantization for QLoRA
DTYPE = None           # Auto-detect (bfloat16 on RTX 4070 Ti)

# LoRA hyperparameters (from Unsloth recommendations)
LORA_R = 16                    # LoRA rank
LORA_ALPHA = 16                # LoRA alpha
LORA_DROPOUT = 0.05            # LoRA dropout
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]  # All linear layers

# Training hyperparameters
BATCH_SIZE = 2                 # Per device batch size (adjust based on VRAM)
GRADIENT_ACCUMULATION = 4      # Effective batch size = 2 * 4 = 8
LEARNING_RATE = 2e-4           # Standard for QLoRA
EPOCHS = 3                     # Number of training epochs
WARMUP_STEPS = 10              # Warmup steps
WEIGHT_DECAY = 0.01            # Weight decay for regularization
LR_SCHEDULER_TYPE = "cosine"   # Cosine learning rate schedule

# Output directory
OUTPUT_DIR = "./gpt-oss-safeguard-finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model_and_tokenizer():
    """Load GPT-OSS Safeguard model with 4-bit quantization."""

    print("🦥 Loading GPT-OSS 20B with Unsloth...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Quantization: {'4-bit BitsAndBytes (QLoRA)' if LOAD_IN_4BIT else 'None'}")
    print(f"   Max sequence length: {MAX_SEQ_LENGTH}")
    print()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("✅ Model loaded successfully")
    print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print()

    return model, tokenizer


def configure_lora(model):
    """Configure LoRA adapters for efficient fine-tuning."""

    print("🔧 Configuring LoRA adapters...")
    print(f"   Rank: {LORA_R}")
    print(f"   Alpha: {LORA_ALPHA}")
    print(f"   Dropout: {LORA_DROPOUT}")
    print(f"   Target modules: {', '.join(TARGET_MODULES)}")
    print()

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params

    print(f"✅ LoRA configured")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"   Total parameters: {total_params:,}")
    print()

    return model


def load_datasets(train_path: str = "train_finetune.json",
                  val_path: str = "val_finetune.json"):
    """Load training and validation datasets from JSONL files."""

    print(f"📥 Loading datasets...")
    print(f"   Training: {train_path}")
    print(f"   Validation: {val_path}")
    print()

    def load_jsonl(filepath):
        """Load JSONL file."""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    print(f"✅ Datasets loaded")
    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Validation examples: {len(val_dataset)}")
    print()

    return train_dataset, val_dataset


def create_trainer(model, tokenizer, train_dataset, val_dataset):
    """Create SFTTrainer with optimal settings for Unsloth."""

    print("🏋️  Setting up trainer...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {EPOCHS}")
    print()

    # Format dataset for training (convert messages to text)
    def format_prompts(examples):
        """Format chat messages into text for training."""
        texts = []
        for messages in examples["messages"]:
            # Simple concatenation of user/assistant messages
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|system|>\n{content}\n"
                elif role == "user":
                    text += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}\n"
            texts.append(text)
        return {"text": texts}

    # Apply formatting
    train_dataset = train_dataset.map(format_prompts, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_prompts, batched=True, remove_columns=val_dataset.column_names)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=TrainingArguments(
            # Training configuration
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,

            # Optimization
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",  # 8-bit AdamW for memory efficiency

            # Logging and evaluation
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="epoch",
            save_total_limit=2,

            # Output
            output_dir=OUTPUT_DIR,
            report_to="none",  # Disable wandb/tensorboard

            # Memory optimization
            gradient_checkpointing=True,

            # Reproducibility
            seed=42,
        ),
    )

    print("✅ Trainer created")
    print()

    return trainer


def main():
    """Main training pipeline."""

    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! This script requires a GPU.")

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Configure LoRA
    model = configure_lora(model)

    # Load datasets
    train_dataset, val_dataset = load_datasets()

    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset)

    # Train
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    train_result = trainer.train()

    print()
    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print()
    print(f"📊 Training metrics:")
    print(f"   Final loss: {train_result.metrics['train_loss']:.4f}")
    print(f"   Training time: {train_result.metrics['train_runtime']:.1f}s")
    print(f"   Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
    print()

    # Save model
    print("💾 Saving fine-tuned model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Also save LoRA adapters separately
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapters")

    print(f"✅ Model saved to {OUTPUT_DIR}")
    print()

    # Memory summary
    print("📈 Final memory usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"   Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print()

    print("=" * 80)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"1. Model saved to: {OUTPUT_DIR}")
    print(f"2. LoRA adapters saved to: {OUTPUT_DIR}/lora_adapters")
    print("3. Run evaluation script to test fine-tuned model")
    print("4. Compare results with baseline (69% recall, 77.1% F1)")


if __name__ == "__main__":
    main()
