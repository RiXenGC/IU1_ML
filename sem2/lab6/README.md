## Загрузка датасета
 
Запуск из корня проекта (нужен настроенный `.env` и параметры `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, см. корневой README; либо должен быть настроен глобальный `kaggle.json` с вашим ключом)

```bash
uv run python -m lab6.src.utils.download_data
```

## Разделение датасета на train, val, test

