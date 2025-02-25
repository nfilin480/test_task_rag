# RAG System with Query Classification

Этот проект представляет собой MVP (Minimum Viable Product) RAG (Retrieval-Augmented Generation) системы с интегрированной классификацией запросов.

## Основные компоненты

### Query Classification

Разработан классификатор запросов пользователей для определения необходимости поиска в базе знаний:
- **Класс 0**: Требуется поиск в базе знаний (запросы, требующие релевантную информацию, проверку фактов, четкие ответы)
- **Класс 1**: Поиск не требуется (базовые NLP задачи или запросы, не требующие конкретных данных/фактов)

#### Данные для обучения
Использованы и размечены данные из следующих датасетов HuggingFace:
- yahma/alpaca-cleaned (3k)
- TokenBender/roleplay_alpaca (3k)
- sahil2801/CodeAlpaca-20k (2k)
- nataliaElv/oasst_quality_with_suggestions (3k)
- positivethoughts/merge_rewrite_13.3k (2k)
- databricks/databricks-dolly-15k (2k)
- curaihealth/medical_questions_pairs (1.5k)

Почему не взял данные databricks и не разметил на основе категорий? Категории размечены криво (например, в closed_qa очень много extract information или general_qa) и нельзя однозначно сказать по категории к какому классу относится запрос. 

Основная обнаруженная проблема: нельзя привязываться только к инструкции сформулированного пользователем запроса. 

#### Процесс разметки
1. Первичная разметка: DeepSeek-R1-Distill-Qwen-32B (4-bit квантизация)
2. Фильтрация разметки: gemma2-7b-it
3. Итоговый датасет: 6,260 размеченных примеров (сбалансирован по классам)

#### Результаты классификации (модель distilbert/distilbert-base-uncased, остальные модели показывали плюс минус такое же качество, но данная модель будет наиболее быстрая)
(Модель скинул на HF - https://huggingface.co/nfilin/distilbert_query_classification)

Базовые метрики:
- Accuracy: 0.85
- Precision: 0.89
- Recall: 0.80
- F1: 0.84

Метрики при threshold = 0.9:
- Accuracy: 0.93
- Precision: 0.93
- Recall: 0.95
- F1: 0.94
![1](https://github.com/user-attachments/assets/b21caab7-b71c-4055-9c3c-172314011810)
![2](https://github.com/user-attachments/assets/19562ee1-a704-4a76-8db8-8c54df155867)

Видим, что в большинстве случаев модель уверена на 90+% в положительном классе (по задаче не критично, если положительные классы попадают в нулевой, гораздо страшнее, если наоборот), поэтому можем смело ставить threshold = 0.9


Свои мысли касаемо задачи: довольно тонкая грань между классами вопросов, при том что, если делать деление на категории, как в статье, то высока вероятность встретить комбинации категорий в одном запросе. Думаю, самый простой вариант сделать решение лучше, это разметить данные более мощной моделью.


### RAG Система

Все необходимые файлы:/src/

#### Основные модули
- **База знаний**: Chroma (Wikipedia, 100k случайных примеров)
- **Embedder**: intfloat/multilingual-e5-large
- **LLM**: Qwen/Qwen2.5-7B-Instruct-1M
- **Searcher**: Гибридный подход (similarity + BM25)
- **Query Classifier**: DistilBERT (дообученный на синтетическом датасете)

## Установка и запуск

Запустить можно либо в блокноте, либо в консоли.

Импортируем окружение: 
1. conda env update -n my_env --file ENV.yaml
2. выполняем все ячейки в /src/test.ipynb или запускаем python /src/main.py
