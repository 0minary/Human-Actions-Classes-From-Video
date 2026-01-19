import kagglehub

# Download latest version
path = kagglehub.dataset_download("ipythonx/k4testset")

print("Path to dataset files:", path)