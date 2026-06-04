# IU1 Machine Learning course

## Содержание

- **lab1** — аппроксимация функции c помощью МНК.
- **lab2** — задача регрессии: восстановление поверхности `Y = f(X1, X2)` (седло).
- **lab3** — задача классификации изображений: определение COVID-19.
- **lab4** — задача многоклассовой классификация токсичности комментариев
- **lab5** – задача определения сепсиса по массиву признаков

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

## Доступ к Kaggle (для lab3 и lab4)

Датасеты lab3 и lab4 скачиваются с Kaggle через API, для чего нужен токен.

1. На [kaggle.com](https://www.kaggle.com) → Settings → API → **Create New Token**. Скачается `kaggle.json` с полями `username` и `key`.
2. В корне проекта создать файл `.env`:

```
KAGGLE_USERNAME=ваш_username
KAGGLE_KEY=ваш_key
```

`.env` читается автоматически (`python-dotenv`) при скачивании данных.

## Загрузка датасетов

Скрипты загрузки запускаются из корня проекта:

```bash
# lab3 — КТ-снимки
uv run python -m lab3.src.utils.download_raw_data

# lab4 — комментарии Jigsaw
uv run python -m lab4.src.utils.download_data
```