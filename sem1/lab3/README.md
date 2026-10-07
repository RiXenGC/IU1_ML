# Лабораторная работа №3

## Задание

Необходимо по одному из наборов данных, предложенных по ссылкам ниже, обучить нейронную сеть. Нейронная сеть должна определять вирус COVID-19 по снимкам лёгких.

Наборы данных:

    1. https://www.kaggle.com/code/thesnak/covid-19-x-ray-using-resnet50-accuracy-96-26/input
    2. https://www.kaggle.com/code/kaledhoshme/x-ray-covid-19-pneumonia-heat-map/input
    3. https://www.kaggle.com/code/bachrr/detecting-covid-19-in-x-ray-images-with-tensorflow/input
    4. https://www.kaggle.com/code/ziadwael/covid-detection-98-genetic-algorithm-mobilenetv2

Необходимая точность модели: 98,5%

## Решение

Используемый датасет: [SARS-CoV-2 CT-scan](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset).

Выполнено:
- Реализован собственный class `Dataset` для чтения снимков.
- Аугментации через `albumentations`.
- Базовая модель: **ResNet18**.
- Класс `Trainer` с остановкой и сохранением лучших весов.
- Альтернативная модель: **Random Forest** на признаках для сравнения.


## Структура
 
```
lab3/
├── config.py                    
├── src/
│   ├── models/baseline.py       
│   ├── trainer.py                
│   └── utils/
│       ├── dataset.py            
│       └── download_raw_data.py
└── notebooks/
    ├── baseline.ipynb           
    └── random_forest.ipynb  
```
 
## Загрузка датасета
 
Запуск из корня проекта (нужен настроенный `.env`, см. корневой README):
 
```bash
uv run python -m sem1.lab3.src.utils.download_raw_data
```

