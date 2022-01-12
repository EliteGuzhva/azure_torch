import gdown
import os
import zipfile


url = 'https://drive.google.com/uc?id=1Q-DTO-aRkruZN7Pg1LVjlC3rPXuNctO6'
filename = 'intel-image-classification.zip'
data_root = 'intel-image-classification'
filepath = os.path.join(data_root, filename)

if not os.path.isdir(data_root):
    os.makedirs(data_root, exist_ok=True)
    gdown.download(url, filepath, quiet=False)
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(data_root)
    os.remove(filepath)
