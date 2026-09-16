# Deployment and dependency files

## `requirements.txt`

Full local development environment. It includes the serve-time dependencies plus PyTorch, torchvision, Ultralytics, ONNX tooling, scikit-learn, YAML support, Kaggle CLI, pytest, and coverage tools. Use it for data preparation, training-related work, and tests.

## `requirements-deploy.txt`

Small CPU-only runtime environment for Render. It intentionally excludes PyTorch, Ultralytics, OpenCV, and training tools. The deployed app uses FastAPI, Uvicorn, multipart upload handling, Pydantic, ONNX Runtime, Pillow, and NumPy.

## `render.yaml`

Render service definition. It selects Python 3.11.9, installs `requirements-deploy.txt`, starts `uvicorn app:app` on Render's assigned port, and checks `/api/v1/health` as the service health endpoint.

## `.gitignore`

Excludes Python caches, test output, local virtual environments, raw/prepared data, temporary PyTorch checkpoints, ONNX external-data sidecars, Colab upload archives, Ultralytics run directories, and local environment files. The committed ONNX model files remain tracked.
