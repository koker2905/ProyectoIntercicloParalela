from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import base64
from processor import GPUProcessor

app = FastAPI()
gpu = GPUProcessor()

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    # 1. Leer imagen de la petición
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Procesar en la RTX 2060
    result_img, metrics = gpu.apply_filter(img)

    # 3. Codificar para enviar de vuelta
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "image": img_base64,
        "metrics": metrics
    }