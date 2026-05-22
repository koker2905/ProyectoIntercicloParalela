// filters_kernel.cu

extern "C" {

    // ========================================================================
    // 1. KERNEL DE CONVOLUCIÓN GENÉRICA (Box, High-Boost, Emboss, Gaussian)
    // ========================================================================
    // Este kernel es versátil. Recibe una matriz calculada en Python y la aplica.
    // Maneja los bordes replicando el píxel más cercano (clamp) para evitar
    // que la imagen se reduzca o tenga bordes negros.
    __global__ void convolucionGPU(
        unsigned char* input,
        unsigned char* output,
        float* kernel,
        int width,
        int height,
        int channels,
        int kSize)
    {
        // Cálculo del índice global del hilo
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int offset = kSize / 2;

        // Asegurar que el hilo está dentro de los límites de la imagen
        if (x >= 0 && x < width && y >= 0 && y < height) {
            
            // Procesamos cada canal de color (B, G, R en el caso de OpenCV)
            for(int c = 0; c < channels; c++) {
                float suma = 0.0f;
                
                // Recorremos la máscara de convolución
                for(int ky = -offset; ky <= offset; ky++) {
                    for(int kx = -offset; kx <= offset; kx++) {
                        
                        // Clamp para los bordes (evita desbordamiento de memoria)
                        int px = min(max(x + kx, 0), width - 1);
                        int py = min(max(y + ky, 0), height - 1);
                        
                        // Índice del píxel y de la máscara
                        int pixel_idx = (py * width + px) * channels + c;
                        int mask_idx = (ky + offset) * kSize + (kx + offset);
                        
                        suma += input[pixel_idx] * kernel[mask_idx];
                    }
                }
                
                // Clamping estricto entre 0 y 255
                int resultado = min(max((int)suma, 0), 255);
                output[(y * width + x) * channels + c] = resultado;
            }
        }
    }

    // ========================================================================
    // 2. KERNEL DE IDENTIDAD UPS (Duotono + Marca de Agua)
    // ========================================================================
    // Transforma la imagen a los colores institucionales (Azul y Oro) basándose
    // en la luminancia, e inserta un logo en la esquina inferior derecha.
    __global__ void upsIdentidadLogo(
        unsigned char* img_in, 
        unsigned char* logo_in, 
        unsigned char* img_out, 
        int w, int h, int channels, 
        int logo_w, int logo_h, 
        int umbral_luz) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < w && y < h) {
            int idx = (y * w + x) * channels;
            
            // Coordenadas donde empieza el logo (esquina inferior derecha con margen de 20px)
            int startX = w - logo_w - 20;
            int startY = h - logo_h - 20;

            // 1. ZONA DEL LOGO
            if (x >= startX && y >= startY && x < w - 20 && y < h - 20) {
                int logo_x = x - startX;
                int logo_y = y - startY;
                int logo_idx = (logo_y * logo_w + logo_x) * channels;
                
                // Copiar colores del logo directamente
                img_out[idx]     = logo_in[logo_idx];
                img_out[idx + 1] = logo_in[logo_idx + 1];
                img_out[idx + 2] = logo_in[logo_idx + 2];
            } 
            // 2. ZONA DE LA FOTO (DUOTONO)
            else {
                // OpenCV lee las imágenes en formato BGR (Blue, Green, Red)
                int blue  = img_in[idx];
                int green = img_in[idx+1];
                int red   = img_in[idx+2];

                // Calcular luminancia (Escala de grises simple)
                int gris = (blue + green + red) / 3;

                // Aplicar paleta UPS según el umbral dictado por el Slider
                if (gris > umbral_luz) {
                    // Amarillo/Oro UPS (Aproximación BGR)
                    img_out[idx]     = 0;     // B
                    img_out[idx + 1] = 215;   // G
                    img_out[idx + 2] = 255;   // R
                } else {
                    // Azul UPS (Aproximación BGR)
                    img_out[idx]     = 102;   // B
                    img_out[idx + 1] = 51;    // G
                    img_out[idx + 2] = 0;     // R
                }
            }
        }
    }

    // ========================================================================
    // 3. KERNEL DE PIXELADO (Efecto Censura / 8-Bits)
    // ========================================================================
    // Divide la imagen en bloques y pinta todo el bloque del color del píxel
    // que se encuentra en la esquina superior izquierda de dicho bloque.
    __global__ void pixeladoGPU(
        unsigned char* input, 
        unsigned char* output, 
        int w, int h, int channels, 
        int block_size) 
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < w && y < h) {
            // Encontrar la coordenada "ancla" del bloque actual
            int anchor_x = (x / block_size) * block_size;
            int anchor_y = (y / block_size) * block_size;

            int idx = (y * w + x) * channels;
            int anchor_idx = (anchor_y * w + anchor_x) * channels;

            // Todos los hilos del mismo bloque leen el color del píxel ancla
            for(int c = 0; c < channels; c++) {
                output[idx + c] = input[anchor_idx + c];
            }
        }
    }

} // Fin de extern "C"