#ifndef CUBLAS_INV_MATRIX_H
#define CUBLAS_INV_MATRIX_H


int my_gauss_inverse_gpu_by_cublas(float** d_in, int size, float** d_out, int my_np);

#endif // !CUBLAS_INV_MATRIX_H
