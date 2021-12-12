# Облачные технологии: Курсовая
Distributed training with PyTorch and AzureML.

## Prerequisites
```bash
pip install torch torchvision azureml-core azureml-dataprep
```

## Usage
Sample
```bash
python job.py \
    --epochs 25 \
    --train-script train.py \
    --experiment-name my-experiment \
    --cluster-name my-cluster \
    --vm-size STANDARD_NC6 \
    --num_nodes 1
```

## TODO:
- [ ] Добавить модель из ноутбука на Kaggle
- [ ] Загрузка датасета Intel
- [x] Добавить аргументы для запуска job.py (script_name, experiment_name, cluster_name, epochs, ...)
- [ ] Добавить валидацию модели на каждой эпохе обучения
- [ ] Добавить отписывание графика с результатами обучения (loss, accuracy, etc..) в .png

