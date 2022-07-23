/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"

static
void check_inversed_matrix(__DATA_TYPE* matrix, bool u, int n, int my_np, std::string type = std::string("definite")) {

	if (u) {
		printf("\n*****gpu inverse completely!!!*****\n");
		int res = getConfigInt(USE_COPY_MATRIX_C);
		double realVal = readInversedMatrix(n, type);
		if (res == 0) {
			for (int i = 0; i < my_np; i++) {
				if (i == 0 || i == 102) {
					//printf("check_inversed_matrix:%6.15lf\n", matrix[i*n*n]);
					/*tools_print_matrix(matrix + (i * n * n), n);
					printf("\n------------------------------\n");*/
					
					writeCheckedInfo(n, my_np, i, realVal, matrix[i * n * n]);
				}
			}
		}
		else {
			for (int i = 0; i < 2; i++) {
				/*tools_print_matrix(matrix + (i * n * n), n);
				printf("\n------------------------------\n");*/
				printf("check_inversed_matrix:%6.15lf\n", matrix[i * n * n]);
				writeCheckedInfo(n, my_np, i, realVal, matrix[i * n * n]);
			}
		}
	}
	else {
		printf("\nERROR!!!\n");
	}
}

static
void check_inversed_matrix2(__DATA_TYPE** matrix, bool u, int n, int my_np, std::string type = std::string("definite")) {

	if (u) {
		printf("\n*****gpu inverse completely!!!*****\n");
		double realVal = readInversedMatrix(n, type);
		for (int i = 0; i < my_np; i++) {
			if (i == 0 || i == 40000) {
				writeCheckedInfo(n, my_np, i, realVal, matrix[i][0]);
				//tools_print_matrix(matrix[i], n);
			}
		}
	}
	else {
		printf("\nERROR!!!\n");
	}
}



void test_cublas(int size, int my_np, std::string type) {
	printf("\n\n*****TEST CUBLAS INVERSE!!!*****\n");
	printf("\n\n*****size:%d,number of matrix is %d\n", size, my_np);
	float** matrix_gpu = NULL, ** d_out_gpu = new float* [my_np];
	cout << size << endl;
	cout << my_np << endl;
	cout << string("./data/").append(type).append("/").append(num2str(size)).append("/data1.txt") << endl;
	matrix_gpu = random_matrix_generate_by_matlab2(size, my_np, string("./data/").append(type).append("/").append(num2str(size)).append("/data1.txt"));


	for (int i = 0; i < my_np; i++) {
		cudaMalloc((void**)&d_out_gpu[i], sizeof(float) * size * size);
		//d_out_gpu[i] = (float*)malloc(sizeof(float) * size * size);
		//memset(d_out_gpu[i], 0, sizeof(float) * size * size);
	}

	CYW_TIMER timer;
	timer.start_my_timer();

	int u = my_gauss_inverse_gpu_by_cublas(matrix_gpu, size, d_out_gpu, my_np);



	timer.stop_my_timer();
	timer.print();

	for (int i = 0; i < my_np; i++) {
		cudaFree(matrix_gpu[i]);
		cudaFree(d_out_gpu[i]);
		//if (matrix_gpu[i]) free(matrix_gpu[i]);
		//if (d_out_gpu[i]) free(d_out_gpu[i]);
	}
	if (matrix_gpu) free(matrix_gpu);
	if (d_out_gpu) delete[] d_out_gpu;

}
