import torch
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.cuda.amp import autocast, GradScaler

# ------------------------------
# Model and LoRA Setup
# ------------------------------
model_name = 'vinai/bertweet-base'

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set up the LoRA configuration (adjust parameters as needed)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,      # Specify your task
    r=8,                            # Low-rank dimension for the adapters
    lora_alpha=32,                  # Scaling factor for LoRA layers
    lora_dropout=0.1,               # Dropout ratio for the new layers
    target_modules=["query", "value"]  # Target modules to adapt (common choices in transformers)
)

# Wrap the model with LoRA adapters (freezing base weights)
model = get_peft_model(model, lora_config)
print("Trainable parameters after applying LoRA:")
model.print_trainable_parameters()

# Move the modified model to GPU
model.to("cuda")

# ------------------------------
# Testing with Gradient Accumulation and Mixed Precision
# ------------------------------
def get_max_effective_batch_size_lora(model, accumulation_steps=4, use_amp=True):
    """
    Robust effective batch size finder for LoRA fine-tuning,
    using both gradient accumulation and mixed precision training.
    
    The effective batch size is (micro_batch_size x accumulation_steps).
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    seq_length = 70  # Test input dimensions (should match your data)
    # Candidate effective batch sizes to test (total sample count per update)
    effective_batch_sizes = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 16, 8, 4, 2]
    working_effective_batch = None
    scaler = GradScaler(enabled=use_amp)
    
    for effective_bs in effective_batch_sizes:
        # Compute micro-batch size based on accumulation steps
        micro_batch_size = effective_bs // accumulation_steps
        
        try:
            # Reset gradients
            model.zero_grad()
            
            # Simulate gradient accumulation over several micro-batches
            for step in range(accumulation_steps):
                inputs = {
                    "input_ids": torch.randint(0, 10000, (micro_batch_size, seq_length), device="cuda"),
                    "attention_mask": torch.ones((micro_batch_size, seq_length), device="cuda"),
                    "labels": torch.randint(0, 2, (micro_batch_size,), device="cuda")
                }
                
                # Use mixed precision if enabled
                with autocast(enabled=use_amp):
                    outputs = model(**inputs)
                    # Scale the loss to distribute it over accumulation steps
                    loss = outputs.logits.sum() / accumulation_steps
                
                scaler.scale(loss).backward()
            
            torch.cuda.synchronize()
            mem_used = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Effective batch {effective_bs} (micro: {micro_batch_size} x {accumulation_steps}) | Memory used: {mem_used:.2f}GB")
            working_effective_batch = effective_bs
            break
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Effective batch {effective_bs} failed - OOM")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise
        
        finally:
            # Reset gradients for the next iteration
            model.zero_grad()
    
    torch.cuda.empty_cache()
    gc.collect()
    return working_effective_batch

# Try testing with gradient accumulation (e.g., 4 steps) and mixed precision enabled
effective_bs = get_max_effective_batch_size_lora(model, accumulation_steps=4, use_amp=True)
print(f"Maximum effective batch size using gradient accumulation and mixed precision: {effective_bs}")




























import time
import torch
import gc
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.cuda.amp import autocast, GradScaler

# ------------------------------
# Model and LoRA Setup
# ------------------------------
model_name = 'vinai/bertweet-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,      # Specify your task
    r=8,                             # Low-rank dimension for the adapters
    lora_alpha=32,                   # Scaling factor for LoRA layers
    lora_dropout=0.1,                # Dropout ratio for the new layers
    target_modules=["query", "value"]  # Target modules to adapt
)

model = get_peft_model(model, lora_config)
print("Trainable parameters after applying LoRA:")
model.print_trainable_parameters()

model.to("cuda")

# ------------------------------
# Prepare Your Dataset
# ------------------------------
# Assume that you have already executed:
# tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# and that tokenized_dataset has splits (e.g., "train", "test", "validation").
# Select the "train" split:
data_split = tokenized_dataset["train"]

# Create two versions of the dataset:
# 1. CPU dataset: remains on CPU and each batch will be transferred.
dataloader_cpu = DataLoader(data_split, batch_size=256, shuffle=False)

# 2. GPU dataset: Pre-load the entire dataset onto GPU.
input_ids_gpu = data_split["input_ids"].to("cuda")
attention_mask_gpu = data_split["attention_mask"].to("cuda")
labels_gpu = data_split["labels"].to("cuda")
gpu_dataset = TensorDataset(input_ids_gpu, attention_mask_gpu, labels_gpu)
dataloader_gpu = DataLoader(gpu_dataset, batch_size=256, shuffle=False)

# ------------------------------
# Setup for Gradient Accumulation & Mixed Precision
# ------------------------------
accumulation_steps = 4
use_amp = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scaler = GradScaler(enabled=use_amp)

# ------------------------------
# Training Loop Function
# ------------------------------
def run_training(model, dataloader, accumulation_steps, use_amp, num_batches=50):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    start_time = time.perf_counter()
    
    # Iterate only over num_batches to get a timing estimate.
    for step, batch in enumerate(dataloader):
        if step >= num_batches:
            break

        # For the CPU dataset, each batch comes as a dictionary
        # For the GPU dataset, each batch is a tuple from TensorDataset.
        if isinstance(batch, dict):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
        else:
            input_ids, attention_mask, labels = batch  # Already on GPU

        with autocast(enabled=use_amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs.logits.sum()
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        total_loss += loss.item()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    end_time = time.perf_counter()
    epoch_time = end_time - start_time
    return epoch_time, total_loss

# ------------------------------
# Run and Compare Training Time
# ------------------------------
num_batches = 50  # Number of batches to simulate for timing

# Run training on CPU-based dataset (with per-batch transfers)
time_cpu, loss_cpu = run_training(model, dataloader_cpu, accumulation_steps, use_amp, num_batches)
print(f"CPU Dataset: Time for {num_batches} batches: {time_cpu:.2f} seconds, Total Loss: {loss_cpu:.4f}")

# Cleanup before next run.
optimizer.zero_grad()
torch.cuda.empty_cache()
gc.collect()

# Run training on GPU-resident dataset
time_gpu, loss_gpu = run_training(model, dataloader_gpu, accumulation_steps, use_amp, num_batches)
print(f"GPU Dataset: Time for {num_batches} batches: {time_gpu:.2f} seconds, Total Loss: {loss_gpu:.4f}")