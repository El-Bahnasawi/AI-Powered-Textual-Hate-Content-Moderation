from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
import torch

# Load full model
model = AutoModelForSequenceClassification.from_pretrained("artifacts/final_model-v4")
model.eval()

# Apply dynamic quantization for CPU inference
quantized_8_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # only quantize linear layers
    dtype=torch.qint8
)

quantized_8_model.to("cpu");

# BitsAndBytes
# - only works with GPU
# - Achieve the same results as the fp32 [normal] model
# - int4 has inference time: 4.5s, normal model: 6.5s


# Define quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False
)

# Load the model with quantization config
quantized_8_model = AutoModelForSequenceClassification.from_pretrained(
    "artifacts/final_model-v5",
    quantization_config=bnb_config,
    device_map="auto"
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=False,
)

# Load the model with quantization config
quantized_4_model = AutoModelForSequenceClassification.from_pretrained(
    "artifacts/final_model-v5",
    quantization_config=bnb_config,
    device_map="cuda"
)