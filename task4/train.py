"""
Задание 4: Дообучение LoRA-адаптера на базе dolphin-llama3-8B
Сценарий: генерация adversarial-запросов класса C5 (текстовые искажения)
"""

import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ── Конфиг ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "cognitivecomputations/dolphin-2.9-llama3-8b"
DATASET     = "training_dataset.jsonl"   # подготовленная выборка
OUTPUT_DIR  = "./zoo-lora-c5"
SEED        = 42

# ── Квантизация 4-bit NF4 ─────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Загрузка модели ───────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ── LoRA конфигурация ─────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Датасет ───────────────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files=DATASET, split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
train_ds = dataset["train"]
eval_ds  = dataset["test"]

def format_prompt(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )

# ── Обучение ──────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=10,
    seed=SEED,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    formatting_func=format_prompt,
    max_seq_length=512,
    packing=False,
)

trainer.train()
trainer.model.save_pretrained(f"{OUTPUT_DIR}-final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")
print("Обучение завершено. Адаптер сохранён в", f"{OUTPUT_DIR}-final")
