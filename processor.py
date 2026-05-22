import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import math
import cv2

class GPUProcessor:
    def __init__(self):
        # Cargar los kernels C++ desde el archivo
        with open("filters_kernel.cu", "r") as f:
            self.mod = SourceModule(f.read())
            
        # Punteros a las funciones en C++
        self.kernel_conv = self.mod.get_function("convolucionGPU")
        self.kernel_ups = self.mod.get_function("upsIdentidadLogo")
        self.kernel_pixel = self.mod.get_function("pixeladoGPU")

    def _get_kernel_matrix(self, filter_type: str, k_size: int) -> np.ndarray:
        """Genera la matriz de convolución según el filtro y el slider."""
        if filter_type == "box_blur":
            return np.full((k_size, k_size), 1.0 / (k_size * k_size), dtype=np.float32)
            
        elif filter_type == "high_boost":
            kernel = np.full((k_size, k_size), -1.0 / (k_size * k_size), dtype=np.float32)
            centro = k_size // 2
            kernel[centro, centro] = 2.0
            return kernel
            
        elif filter_type == "emboss":
            kernel = np.zeros((k_size, k_size), dtype=np.float32)
            for y in range(k_size):
                for x in range(k_size):
                    if x + y < k_size - 1:
                        kernel[y, x] = -1.0
                    elif x + y > k_size - 1:
                        kernel[y, x] = 1.0
            return kernel
            
        elif filter_type == "gaussian_blur":
            sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8 
            kernel = np.zeros((k_size, k_size), dtype=np.float32)
            offset = k_size // 2
            suma = 0.0
            for y in range(-offset, offset + 1):
                for x in range(-offset, offset + 1):
                    g = math.exp(-(x**2 + y**2) / (2 * sigma**2))
                    kernel[y + offset, x + offset] = g
                    suma += g
            return (kernel / suma).astype(np.float32)
            
        return None

    def apply_filter(self, img_np: np.ndarray, filter_type: str, slider_val: int):
        h, w, channels = img_np.shape
        img_out = np.empty_like(img_np)

        # Configuración de la topología CUDA
        block = (16, 16, 1)
        grid = (int(np.ceil(w / block[0])), int(np.ceil(h / block[1])))

        start_event = cuda.Event()
        end_event = cuda.Event()

        # Reservar memoria en GPU para la imagen
        img_gpu_in = cuda.mem_alloc(img_np.nbytes)
        img_gpu_out = cuda.mem_alloc(img_out.nbytes)
        cuda.memcpy_htod(img_gpu_in, img_np)

        start_event.record()

        # --- Lógica de enrutamiento de filtros ---
        if filter_type in ["box_blur", "high_boost", "emboss", "gaussian_blur"]:
            # Asegurar que el tamaño del kernel sea impar
            k_size = slider_val if slider_val % 2 != 0 else slider_val + 1
            matrix = self._get_kernel_matrix(filter_type, k_size)
            
            matrix_gpu = cuda.mem_alloc(matrix.nbytes)
            cuda.memcpy_htod(matrix_gpu, matrix)
            
            self.kernel_conv(
                img_gpu_in, img_gpu_out, matrix_gpu,
                np.int32(w), np.int32(h), np.int32(channels), np.int32(k_size),
                block=block, grid=grid
            )
            matrix_gpu.free()

        elif filter_type == "ups":
            # Creamos un logo simulado en amarillo para evitar cargar un archivo externo ahora
            logo_w, logo_h = 100, 50
            logo_np = np.full((logo_h, logo_w, channels), [0, 215, 255], dtype=np.uint8) # BGR
            logo_gpu = cuda.mem_alloc(logo_np.nbytes)
            cuda.memcpy_htod(logo_gpu, logo_np)
            
            self.kernel_ups(
                img_gpu_in, logo_gpu, img_gpu_out,
                np.int32(w), np.int32(h), np.int32(channels),
                np.int32(logo_w), np.int32(logo_h), np.int32(slider_val),
                block=block, grid=grid
            )
            logo_gpu.free()

        elif filter_type == "pixelate":
            block_size = max(2, slider_val) # Evitar divisiones por cero
            self.kernel_pixel(
                img_gpu_in, img_gpu_out,
                np.int32(w), np.int32(h), np.int32(channels), np.int32(block_size),
                block=block, grid=grid
            )
            
        else:
            raise ValueError(f"Filtro no reconocido: {filter_type}")

        # --- Fin de ejecución y métricas ---
        end_event.record()
        end_event.synchronize()
        cuda.memcpy_dtoh(img_out, img_gpu_out)

        # Liberar memoria principal
        img_gpu_in.free()
        img_gpu_out.free()

        # Tiempo reportado estrictamente en minutos
        gpu_time_ms = start_event.time_since(end_event)
        gpu_time_min = abs(gpu_time_ms) / 60000.0

        metrics = {
            "filtro": filter_type,
            "slider_value": slider_val,
            "img_size": f"{w}x{h}",
            "block_dim": f"{block[0]}x{block[1]}",
            "grid_dim": f"{grid[0]}x{grid[1]}",
            "threads_total": block[0] * block[1] * grid[0] * grid[1],
            "execution_time_min": round(gpu_time_min, 8)
        }

        return img_out, metrics