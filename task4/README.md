# Задание 4 — Инструкция по запуску

## Требования

- Python 3.10+
- NVIDIA GPU с 24 ГБ VRAM (RTX 4090 или 2× RTX 3090)
- CUDA 12.1

## Установка зависимостей

```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 peft==0.10.0 trl==0.8.0
pip install bitsandbytes==0.43.0 datasets==2.19.0
pip install pandas openpyxl accelerate
```

## Структура файлов

```
task4/
├── train.py                          # Обучение LoRA-адаптера
├── generate.py                       # Генерация adversarial-датасета
├── config.json                       # Гиперпараметры
├── metrics.json                      # Метрики качества
├── examples.json                     # Примеры до/после
├── adversarial_dataset_c5_real.xlsx  # Готовый датасет (100 примеров)
└── adversarial_results.json          # Датасет в формате JSON
```

## Запуск

**Шаг 1** — Обучение адаптера:
```bash
python train.py
```
Адаптер сохранится в папку `zoo-lora-c5-final/`. Время: ~4–6 часов на RTX 4090.

**Шаг 2** — Генерация датасета:
```bash
python generate.py
```
Результат: `adversarial_dataset_c5.xlsx` с 1000 примерами.

## Примечание

Все гиперпараметры в `config.json` — меняй там, не трогая код.  
Модель (~16 ГБ) скачается автоматически с Hugging Face.  
Без GPU обучение невозможно.
