# IU1 Machine Learning course

## Содержание

-- **sem1**
- **lab1** — Регрессия (восстановление линейной функции)
- **lab2** — Регрессия (восстановление поверхности `Y = f(X1, X2)`)
- **lab3** — Классификация изображения (определение COVID-19)
- **lab4** — Классификация токсичности комментариев (NLP)
- **lab5** – Определение сепсиса по массиву признаков

-- **sem2**
- **lab6** – Предсказание траектории движения (GAN)
- **lab7** – ...
- **lab8** – ...
- **lab9** – ...
- **lab10** – ...

## Установка

Проект использует [uv](https://docs.astral.sh/uv/) в качестве пакетного менеджера.

```bash
# установка uv (Linux / macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Установка зависимостей и создание окружения:

```bash
uv sync
```

## Доступ к Kaggle

Некоторые датасеты скачиваются с Kaggle через API, для чего нужен токен.

1. На [kaggle.com](https://www.kaggle.com) → Settings → API → **Generate New Token**. Скачается `kaggle.json` с полями `username` и `key`.
2. В корне проекта создать файл `.env`:

```
KAGGLE_USERNAME=ваш_username
KAGGLE_KEY=ваш_key
```

`.env` читается автоматически (`python-dotenv`) при скачивании данных.
Либо второй вариант: разместить ключ `kaggle.json` в системной директории

## Загрузка датасетов

Скрипты загрузки запускаются из корня проекта:

```bash
# lab3 — КТ-снимки
uv run python -m lab3.src.utils.download_raw_data

# lab4 — комментарии Jigsaw
uv run python -m lab4.src.utils.download_data

# lab6 — ...
```