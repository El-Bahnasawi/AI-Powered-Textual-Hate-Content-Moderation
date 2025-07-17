import wandb
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments)
from Code_build.utils import compute_metrics
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from Code_build.sweep import sweep_id
from Code_build.dataset import tokenized_dataset

def train():
    # Initialize W&B run
    run = wandb.init()
    
    # Get hyperparameters from W&B
    config = wandb.config

    model_name = 'distilbert-base-cased'

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16)


    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, quantization_config=bnb_config)
    
    # Setup LoRA
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules= ["attention.q_lin", "attention.k_lin", "attention.v_lin"]
    )
    
    # other_layers = ["attention.out_lin", "ffn.lin1", "ffn.lin2"]

    model = get_peft_model(model, peft_config)
    
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output/",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        fp16=True,
        learning_rate=config.learning_rate,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        max_grad_norm=1.0,  # Gradient clipping
        warmup_steps=100,   # More gradual warmup
        weight_decay=0.01,  # Regularization
    )
    
    # Initialize trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=compute_metrics
        )
    
    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    
    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    # Log metrics and parameters
    wandb.log({
        "matthews_corrcoef": eval_results["eval_matthews_corrcoef"],
        "trainable_parameters": trainable_params,

    })

# Run the sweep
wandb.agent(sweep_id, function=train, count=5)