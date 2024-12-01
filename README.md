# Docker web server with a local LLM model


## Building the Docker Image

To build the Docker image, run the following command:
```sh
docker-compose up --build -d
```

> [!WARNING]
> Ensure that your Docker configuration allows sufficient RAM allocation. You can check the current usage with:
```sh
docker stats
```

To access the running container, use:
```sh
docker exec -it llama-2 /bin/bash
```

Login huggingface to obtain token for llama
"Llama 2 is licensed under the LLAMA 2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved."
```sh
huggingface-cli login
```

## Visualization

### 1. Jupyter Notebook

To run Jupyter Notebook in Docker, use the container ID:
```sh
jupyter notebook --ip=172.28.0.10 --allow-root
```

On your local machine, replace the resulting URL with `localhost`. For example:
```
http://172.18.0.2:8888/tree?token=e8b6df7a2a -> http://localhost:8888/tree?token=e8b6df7a2a
```

### 2. XServer

Install an XServer on your Windows host, such as VcXsrv. Use `play_xserver.py` as an example.

### 3. Matplotlib Web Server

Map the desired port in `docker-compose`. Use `play_web.py` as an example.