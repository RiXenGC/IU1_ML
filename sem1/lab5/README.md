## Загрузка датасета
 
Запуск из корня проекта (нужен настроенный `.env`, см. корневой README):
 
```bash
uv run python -m lab4.src.utils.download_data
```

Проверьте файл .env
`KAGGLE_API_TOKEN` и `KAGGLE_USERNAME`


## Разделение датасета на train, val, test

```python
python -m lab5.src.data_split
```

## Data Preprocessing

```python
python -m lab5.src.preprocess
```