// filters_kernel.cu
extern "C" {
    __global__ void ups_palette_kernel(float *img_in, float *img_out, int w, int h) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;

        if (x < w && y < h) {
            int idx = (y * w + x) * 3;
            
            // Luminancia (Escala de grises)
            float gray = 0.299f * img_in[idx] + 0.587f * img_in[idx+1] + 0.114f * img_in[idx+2];

            // Efecto Marca de Agua (UPS en la esquina inferior derecha)
            // Si está en los últimos 50x20 píxeles, pintamos "UPS" (muy simplificado)
            bool is_logo_area = (x > w - 60 && x < w - 10 && y > h - 30 && y < h - 10);

            if (is_logo_area) {
                // Color Dorado UPS para el logo
                img_out[idx]   = 1.0f; // R
                img_out[idx+1] = 0.84f; // G
                img_out[idx+2] = 0.0f; // B
            } else {
                // Aplicar Paleta Institucional según brillo
                if (gray > 0.5f) {
                    // Amarillo/Oro UPS
                    img_out[idx]   = 1.0f; 
                    img_out[idx+1] = 0.84f; 
                    img_out[idx+2] = 0.0f;
                } else {
                    // Azul UPS
                    img_out[idx]   = 0.0f; 
                    img_out[idx+1] = 0.2f; 
                    img_out[idx+2] = 0.4f;
                }
            }
        }
    }
    // Aquí podrías agregar los kernels de Sobel y Gaussian después...
}