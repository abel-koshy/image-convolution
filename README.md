# image-convolution
a project for parallel computing MIT '24



objective
The objective is to address the performance disparities between CPU and GPU implementations of 2D convolution operations on grayscale images. Given the parallel processing capabilities of GPUs, GPU-accelerated convolution is hypothesized to significantly enhance computational speed compared to traditional CPU-based approaches. The core challenge is to analyze and quantify the performance gains of using CUDA-enabled GPUs for image processing tasks, particularly focusing on the efficiency and speed of applying a 3x3 convolution kernel to large images, providing insights into optimizing convolution operations for real-time image processing applications.



result
The results demonstrate that the GPU implementation, leveraging CUDA, significantly outperforms the CPU implementation, especially with larger images. This notable performance difference underscores the effectiveness of GPU acceleration for computationally intensive tasks like image convolution, highlighting its suitability for real-time image processing applications.
