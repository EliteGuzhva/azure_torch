# Облачные технологии: Курсовая
Distributed training with PyTorch and AzureML.

## Prerequisites

1. Download ```config.json``` from AzureML workspace and copy to the root of this project.
![](res/azure_config.png)

### Devcontainer (Recommended)
2. In VSCode open Command Palette and run ```Rebuild and Reopen in DevContainer``` or build your own image using ```Dockerfile```.

### Local
2. Install python packages
```bash
pip install -r requirements.txt
```

3. Authentication
```bash
az login
```

## Usage
Sample
```bash
# Download dataset
python3 load_cifar10.py

# Run training job
python3 job.py --config cifar_dist
```

