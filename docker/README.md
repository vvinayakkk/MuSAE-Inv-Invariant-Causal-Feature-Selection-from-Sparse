# Docker

Build and run MuSAE-Inv in a reproducible container.

## Build

```bash
docker build -t musae-inv:latest -f docker/Dockerfile .
```

## Run

```bash
# Full pipeline with GPU
docker run --gpus all -v $(pwd)/outputs:/app/outputs musae-inv:latest

# Interactive shell
docker run --gpus all -it musae-inv:latest bash

# Custom config
docker run --gpus all -v $(pwd)/outputs:/app/outputs \
    musae-inv:latest python scripts/train.py --icfs-top-k 512

# Feature extraction only
docker run --gpus all -v $(pwd)/outputs:/app/outputs \
    musae-inv:latest python scripts/extract_features.py --counterfactual
```

## Docker Compose (optional)

```yaml
version: "3.8"
services:
  musae-inv:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ../outputs:/app/outputs
    environment:
      - HF_TOKEN=${HF_TOKEN}
```
