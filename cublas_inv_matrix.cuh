#ifndef CUBLAS_INV_MATRIX_H
#define CUBLAS_INV_MATRIX_H


int my_gauss_inverse_gpu_by_cublas(double** d_in, int size, double** d_out, int my_np);

#endif // !CUBLAS_INV_MATRIX_H
