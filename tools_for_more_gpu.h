/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#pragma once
#ifndef TOOLS_FOR_MORE_GPU_H
#define TOOLS_FOR_MORE_GPU_H

int getCumulativeSumOfTasks(int start, int end, int n, int sum);

void global_memory_check(int size, int tasks, Device* d);

int get_devided_number_of_single_to_mul_sm(int size, int my_np);



#endif // !TOOLS_FOR_MORE_GPU_H

