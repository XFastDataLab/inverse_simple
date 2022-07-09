/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#pragma once
#ifndef SINGLE_BLOCK_MUL_GAUSS_INVERSE_GPU_H
#define SINGLE_BLOCK_MUL_GAUSS_INVERSE_GPU_H

int my_single_block_mul_gauss_inverse_gpu(__DATA_TYPE* out, int size, int my_np, float& elapse_time);


#endif // !SINGLE_BLOCK_MUL_GAUSS_INVERSE_GPU_H

