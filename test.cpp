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


void test_gauss_on_cpu(int n,int my_np, std::string type=std::string("definite")) {
	__DATA_TYPE* matrix;
	CYW_TIMER timer;
	
	matrix = random_matrix_generate_by_matlab(n,my_np, string("./data/").append(type).append("/").append(num2str(n)).append("/data1.txt"));
	timer.start_my_timer();
	int res = getConfigInt(USE_COPY_MATRIX_C);
	if (res == 0) {
		for (int i = 0; i < my_np; i++) {
			gauss_inverse_cpu(matrix + i * n * n, n);
		}
	}
	else {
		for (int i = 0; i < my_np; i++) {
			gauss_inverse_cpu(matrix, n);
		}
	}
	timer.stop_my_timer();
	
	double costed_time = timer.get_my_timer();
	printf("\n*****CPU costed time:");
	timer.print();
	check_inversed_matrix(matrix, true, n,my_np,type);
	if (matrix) free(matrix);
	//save results to the file of reaulst_cpu.txt.
	writeCPUResults(costed_time);
}



void test_single_block_mul_gauss_inverse_gpu(int n,int my_np, float& elapse_time) {
	printf("\n\n*****TEST SINGLE BLOCK MULTIBL GAUSS INVERSE ON GPU!!!*****\n");
	printf("\n\n*****size:%d,number of matrix is %d\n", n, my_np);
	__DATA_TYPE* matrix = NULL;
	matrix = random_matrix_generate_by_matlab(n,my_np);
	int u = my_single_block_mul_gauss_inverse_gpu(matrix, n,my_np, elapse_time);
	check_inversed_matrix(matrix, u, n,my_np);
	if (matrix) free(matrix);
	test_gauss_on_cpu(n,my_np);
}

void test_single_block_mul_gauss_inverse_gpu(int n, int my_np, std::string type, float& elapse_time) {
	printf("\n\n*****TEST SINGLE BLOCK MULTIBL GAUSS INVERSE ON GPU!!!*****\n");
	printf("\n\n*****size:%d,number of matrix is %d\n", n, my_np);
	__DATA_TYPE* matrix = NULL;
	matrix = random_matrix_generate_by_matlab(n, my_np, string("./data/").append(type).append("/").append(num2str(n)).append("/data1.txt"));
	int u = my_single_block_mul_gauss_inverse_gpu(matrix, n, my_np, elapse_time);
	check_inversed_matrix(matrix, u, n, my_np, type);
	if (matrix) free(matrix);
	//test_gauss_on_cpu(n, my_np,type);
}

void test_algorithm2(int n, int my_np, std::string type) {
	printf("\n\n*****TEST ALGORITHM2 ON GPU!!!*****\n");
	printf("\n\n*****size:%d,number of matrix is %d\n", n, my_np);
	__DATA_TYPE* matrix = NULL;
	matrix = random_matrix_generate_by_matlab(n, my_np, string("./data/").append(type).append("/").append(num2str(n)).append("/data1.txt"));
	
	int u = my_algorithm2(matrix, n, my_np);
	check_inversed_matrix(matrix, u, n, my_np, type);
	if (matrix) free(matrix);
	//test_gauss_on_cpu(n, my_np, type);
}


void test_batched_single_block_mul_gauss_inverse_gpu(int n, int my_np, std::string type) {
	//1. get Q from satuation curve Q = C/n^2 C = 1396100
	// the value of C:
	//2080 super:1396100;  3060ti:1783698    3090:3667363  1080ti:1024842 498683
	int C = 498683;
	float n2 = pow(n,2);
	int Q = C / n/n;
	//Q = 81920;
	int len = my_np / Q;
	int yu = my_np % Q;

	if (Q > my_np) {
		Q = my_np;
		yu = 0;
		len = 1;
	}
	float elapse_time = 0.0;
	printf("Q=C/(n^2):%d, len=%d/Q...%d\n", Q, my_np, yu);
	setConfigFloat("TEMP_ELAPSE_TIME", 0.0f);
	CheckMemoryInfo(n, Q);
	for (int i = 0; i < len; i++) {
		test_single_block_mul_gauss_inverse_gpu(n, Q, type, elapse_time);
		printf("elapse_time_%d:%lf", i, elapse_time);
		addConfigFloat("TEMP_ELAPSE_TIME", elapse_time);
	}
	if (yu > 0) {
		test_single_block_mul_gauss_inverse_gpu(n, yu, type, elapse_time);
		addConfigFloat("TEMP_ELAPSE_TIME", elapse_time);
	}
	writeGPUResults(getConfigFloat("TEMP_ELAPSE_TIME"));
}

void test_more_gpu_for_single_block_mul_gauss_inverse_gpu(int size,int my_np) {
	printf("\n\n*****TEST SINGLE BLOCK MULTIBL GAUSS INVERSE ON MULTIPLE GPUs!!!*****\n");
	printf("\n\n*****size:%d,number of matrix is %d\n", size, my_np);
	__DATA_TYPE* matrix = NULL;
	matrix = random_matrix_generate_by_matlab(size,my_np);
	int u = my_more_gpu_for_single_block_mul_gauss_inverse_gpu(matrix, size,my_np);
	check_inversed_matrix(matrix, u, size,my_np);
	if (matrix) free(matrix);
	test_gauss_on_cpu(size,my_np);
}

void test_more_gpu_for_single_block_mul_gauss_inverse_gpu(int size, int my_np, std::string type) {
	printf("\n\n*****TEST SINGLE BLOCK MULTIBL GAUSS INVERSE ON MULTIPLE GPUs!!!*****\n");
	printf("\n\n*****size:%d,number of matrix is %d\n", size, my_np);
	__DATA_TYPE* matrix = NULL;
	//"./data/definite/3/data1.txt"
	matrix = random_matrix_generate_by_matlab(size, 
		my_np,string("./data/").append(type).append("/").append(num2str(size)).append("/data1.txt")
		);
	int u = my_more_gpu_for_single_block_mul_gauss_inverse_gpu(matrix, size, my_np);
	check_inversed_matrix(matrix, u, size, my_np,type);
	if (matrix) free(matrix);
	test_gauss_on_cpu(size, my_np,type);
}

void test_cublas(int size, int my_np, std::string type) {
	printf("\n\n*****TEST CUBLAS INVERSE!!!*****\n");
	printf("\n\n*****size:%d,number of matrix is %d\n", size, my_np);
	double** matrix_gpu = NULL, ** d_out_gpu = new double* [my_np];
	matrix_gpu = random_matrix_generate_by_matlab2(size, my_np, 
		string("./data/").append(type).append("/").append(num2str(size)).append("/data1.txt"));


	for (int i = 0; i < my_np; i++) {
		d_out_gpu[i] = (double*)malloc(sizeof(double) * size * size);
		memset(d_out_gpu[i], 0, sizeof(double) * size * size);
	}
	cudaEvent_t start, stop;
	cudaError_t cudaerror = cudaSuccess;
	float time_elapsed = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int u = my_gauss_inverse_gpu_by_cublas(matrix_gpu, size, d_out_gpu, my_np);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("cublas gpu执行时间：%f(ms)\n", time_elapsed);

	//check_inversed_matrix2(d_out_gpu, u, size, my_np, type);
	

	for (int i = 0; i < my_np; i++) {
		if (matrix_gpu[i]) free(matrix_gpu[i]);
		if (d_out_gpu[i]) free(d_out_gpu[i]);
	}
	if (matrix_gpu) free(matrix_gpu);
	if (d_out_gpu) free(d_out_gpu);
	test_gauss_on_cpu(size, my_np, type);

}

void about(__DATA_TYPE a) {
	cout << a << endl;
}