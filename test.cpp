/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"

using namespace std;

static
void check_inversed_matrix(__DATA_TYPE* matrix, bool u, int n, int my_np, std::string type = std::string("definite")) {

	if (u) {
		printf("\n*****gpu inverse completely!!!*****\n");
		double realVal = readInversedMatrix(n, type);
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
	for (int i = 0; i < my_np; i++) {
		gauss_inverse_cpu(matrix + i * n * n, n);
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
