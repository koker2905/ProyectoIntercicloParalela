# Usamos una imagen con CUDA ya instalado
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /app

# Instalar dependencias de Python
RUN pip3 install pycuda fastapi uvicorn numpy opencv-python python-multipart

COPY . .

# Exponer el puerto de la API
EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]