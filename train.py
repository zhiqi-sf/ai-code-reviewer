"""
Training script for StarCoder models.

This script provides training capabilities for fine-tuning StarCoder models on custom datasets.
"""

import argparse
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train StarCoder model")
    parser.add_argument("--model_name", type=str, default="THUDM/ChatGLM3-6B")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Load and prepare dataset
    dataset = load_dataset(args.dataset_name)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model()


if __name__ == "__main__":
    main()