# Задание 4 — Инструкция по запуску

## Требования

- Ubuntu 22.04 / Windows WSL2
- Python 3.10+
- NVIDIA GPU с 24 ГБ VRAM (RTX 4090 или 2× RTX 3090)
- CUDA 12.1

## Установка зависимостей

```bash
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 peft==0.10.0 trl==0.8.0
pip install bitsandbytes==0.43.0 datasets==2.19.0
pip install pandas openpyxl accelerate
```

## Структура файлов

```
task4/
├── train.py              # скрипт обучения LoRA-адаптера
├── generate.py           # скрипт генерации adversarial-датасета
├── config.json           # все гиперпараметры
├── metrics.json          # результаты метрик
├── examples.json         # примеры до/после дообучения
└── README.md             # эта инструкция
```

## Шаг 1 — Подготовка датасета

Положите файл `training_dataset.jsonl` рядом с `train.py`.
Каждая строка — JSON с полями: `instruction`, `input`, `output`.

## Шаг 2 — Обучение

```bash
python train.py
```

Адаптер сохранится в папку `zoo-lora-c5-final/`.
Время обучения: ~4–6 часов на RTX 4090.

## Шаг 3 — Генерация датасета

```bash
python generate.py
```

Результат: файл `adversarial_dataset_c5.xlsx` с 1000 примерами.

## Параметры (config.json)

Все гиперпараметры вынесены в `config.json` — при необходимости
меняйте там, не трогая код.

## Ограничения

- Модель весит ~16 ГБ, скачивается с Hugging Face автоматически
- Нужен токен HF если модель приватная: `huggingface-cli login`
- Без GPU обучение невозможно; inference на CPU крайне медленный
