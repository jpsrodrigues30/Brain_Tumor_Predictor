import kagglehub
import os
import shutil

# Faz o download da versão mais recente do dataset
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files (cache):", path)

project_data_path = os.path.join(os.getcwd(), "data", "raw")

os.makedirs(project_data_path, exist_ok=True)

if os.path.exists(project_data_path):
    shutil.rmtree(project_data_path)

shutil.copytree(path, project_data_path)

print("Dataset copiado para:", project_data_path)
