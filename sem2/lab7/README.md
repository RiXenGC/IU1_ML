# Лабораторная работа №4 

## Задание

Задача многоклассовой классификации

## Решение

Датасет: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).


## Структура
 
```
lab4/
├── config.py                  
├── src/
│   ├── text_processor.py     
│   └── utils/
│       └── download_data.py   
└── notebooks/
    └── baseline.ipynb
```


## Загрузка датасета
 
Датасет взят с соревнования Kaggle, поэтому необходимо авторизовации на [странице соревнования](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules).
 
Запуск из корня проекта (нужен настроенный `.env`, см. корневой README):
 
```bash
uv run python -m sem1.lab4.src.utils.download_data
```