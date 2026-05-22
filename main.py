from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from processor import GPUProcessor

app = FastAPI(title="UPSGlam 3.0 - GPU Processing API")

# Configuración CORS para pruebas locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instanciamos el procesador una sola vez al arrancar el servidor
gpu_engine = GPUProcessor()

@app.post("/process")
async def process_image(
    file: UploadFile = File(...), 
    filter_type: str = Form(...),
    slider_value: int = Form(15) # Valor por defecto si no lo envían
):
    try:
        # 1. Leer y decodificar la imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_np is None:
            raise HTTPException(status_code=400, detail="Archivo de imagen no válido.")

        # 2. Enviar a la GPU
        result_img, metrics = gpu_engine.apply_filter(img_np, filter_type, slider_value)

        # 3. Codificar la imagen de salida a Base64
        _, buffer = cv2.imencode('.jpg', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 4. Retornar el contrato JSON esperado
        return {
            "image": img_base64,
            "metrics": metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))