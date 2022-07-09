/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#pragma once
#ifndef MY_TEST_H
#define MY_TEST_H

void test_gauss_on_cpu(int n, int my_np, std::string type);

void test_single_block_mul_gauss_inverse_gpu(int size, int my_np, float& elapse_time);

void test_single_block_mul_gauss_inverse_gpu(int n, int my_np, std::string type, float& elapse_time);

void test_algorithm2(int n, int my_np, std::string type);

void test_batched_single_block_mul_gauss_inverse_gpu(int n, int my_np, std::string type);

void test_more_gpu_for_single_block_mul_gauss_inverse_gpu(int size, int my_np);

void test_more_gpu_for_single_block_mul_gauss_inverse_gpu(int size, int my_np,std::string type);

void test_cublas(int size, int my_np, std::string type);
void test_cublas(int size, int my_np, std::string type);

void about(__DATA_TYPE a);
#endif // !MY_TEST_H

