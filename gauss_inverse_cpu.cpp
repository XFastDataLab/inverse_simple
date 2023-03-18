/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"

int gauss_inverse_cpu(__DATA_TYPE* out, int size) {

	int i, j, k;
	__DATA_TYPE c;
	for (k = 0; k < size; k++) {
		//1.m(k,k) = 1/m(k,k)
		if (out[k * size + k] == 0) {
			printf("The matrix row:%d,col:%d equal 0, 0 cannot be denominator,this bug not be solved yet!\n",k,k);
			exit(0);
		}
		out[k * size + k] = 1.0 / out[k * size + k];
		c = out[k * size + k];
		//2.m(i,k) = -m(k,k) * m(i,k), i!=k
		for (i = 0; i < k; i++)    out[i * size + k] *= -1 * c;
		for (i = k + 1; i < size; i++)    out[i * size + k] *= -1 * c;

		//3.m(i,j) = m(i,j) + m(i,k) * m(k,j), i,j != k
		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++)   out[i * size + j] += out[i * size + k] * out[k * size + j];
			for (j = k + 1; j < size; j++)   out[i * size + j] += out[i * size + k] * out[k * size + j];
		}
		for (i = k + 1; i < size; i++) {
			for (j = 0; j < k; j++)   out[i * size + j] += out[i * size + k] * out[k * size + j];
			for (j = k + 1; j < size; j++)   out[i * size + j] += out[i * size + k] * out[k * size + j];
		}

		//4.m(k,j) = m(k,k) * m(k,j), j != k
		for (j = 0; j < k; j++)    out[k * size + j] *= c;
		for (j = k + 1; j < size; j++)    out[k * size + j] *= c;
	}


	return 1;
}

int gauss_inverse_omp(__DATA_TYPE* out, int size) {

	int i, j, k;
	__DATA_TYPE c;
	for (k = 0; k < size; k++) {
		//1.m(k,k) = 1/m(k,k)
		if (out[k * size + k] == 0) {
			printf("The matrix row:%d,col:%d equal 0, 0 cannot be denominator,this bug not be solved yet!\n", k, k);
			exit(0);
		}
		out[k * size + k] = 1.0 / out[k * size + k];
		c = out[k * size + k];
		//2.m(i,k) = -m(k,k) * m(i,k), i!=k
		#pragma omp parallel for 
		for (i = 0; i < size; i++) {	
			if(i !=k ) out[i * size + k] *= -1 * c;
		}


		//3.m(i,j) = m(i,j) + m(i,k) * m(k,j), i,j != k
		
		for (i = 0; i < k; i++) {
			#pragma omp parallel for 
			for (j = 0; j < size; j++) {
				if (j != k) {
					out[i * size + j] += out[i * size + k] * out[k * size + j];
				}
			}
			
		}
		
		for (i = k + 1; i < size; i++) {
			#pragma omp parallel for 
			for (j = 0; j < size; j++) {
				if (j != k) {
					out[i * size + j] += out[i * size + k] * out[k * size + j];
				}
			}
		}

		//4.m(k,j) = m(k,k) * m(k,j), j != k
		#pragma omp parallel for 
		for (j = 0; j < size; j++) {
			if(j!=k) out[k * size + j] *= c;
		}

	}


	return 1;
}