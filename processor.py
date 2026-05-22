import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

class GPUProcessor:
    def __init__(self):
        # Cargamos el archivo .cu
        with open("filters_kernel.cu", "r") as f:
            self.mod = SourceModule(f.read())
        self.ups_filter = self.mod.get_function("ups_palette_kernel")

    def apply_filter(self, img_np):
        h, w, _ = img_np.shape
        img_np = img_np.astype(np.float32) / 255.0
        img_out = np.empty_like(img_np)

        # Configuración de hilos y bloques
        block = (16, 16, 1)
        grid = (int(np.ceil(w/16)), int(np.ceil(h/16)))

        # Eventos para métricas de tiempo
        start = cuda.Event()
        end = cuda.Event()

        # Memoria GPU
        start.record()
        img_gpu_in = cuda.to_device(img_np)
        img_gpu_out = cuda.mem_alloc(img_np.nbytes)

        self.ups_filter(img_gpu_in, img_gpu_out, np.int32(w), np.int32(h), block=block, grid=grid)

        cuda.memcpy_dtoh(img_out, img_gpu_out)
        end.record()
        end.synchronize()

        # Cálculo de métricas
        gpu_time_ms = start.time_since(end) # Tiempo en ms
        # Como pediste en el historial: mostrar en minutos si es necesario
        gpu_time_min = abs(gpu_time_ms) / 60000 

        metrics = {
            "filtro": "UPS_Institucional",
            "size": f"{w}x{h}",
            "block": "16x16",
            "grid": f"{grid[0]}x{grid[1]}",
            "threads": block[0] * block[1] * grid[0] * grid[1],
            "execution_time_min": round(gpu_time_min, 6)
        }

        return (img_out * 255).astype(np.uint8), metrics