# Облачные технологии: Курсовая
Distributed training with PyTorch and AzureML.

## Prerequisites
```bash
pip install torch torchvision azureml-core azureml-dataprep
```

## Usage
Sample
```bash
# Download dataset
python load_cifar10.py

# Run training job
python job.py \
    --epochs 25 \
    --batch-size 16 \
    --train-script train.py \
    --data-dir cifar-10 \
    --experiment-name my-experiment \
    --cluster-name my-cluster \
    --vm-size STANDARD_NC6 \
    --num_nodes 1 \
    --environment-name AzureML-PyTorch-1.6-GPU
```

## TODO:
- [ ] Добавить модель из ноутбука на Kaggle
- [ ] Загрузка датасета Intel (load_intel.py + preprocessing)
- [x] Добавить аргументы для запуска job.py (script_name, experiment_name, cluster_name, epochs, ...)
- [ ] Добавить валидацию модели на каждой эпохе обучения
- [ ] Добавить отписывание графика с результатами обучения (loss, accuracy, etc..) в .png

