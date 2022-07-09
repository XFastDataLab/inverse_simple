/*
 *
 * Copyright (C) 2020-  Yewang Chen<ywchen@hqu.edu.cn;nalandoo@gmail.com>
 * License: GPL v1
 * This software may be modified and distributed under the terms
 * of license.
 *
 */

#include "def.h"

// ***** global macros ***** //  
static const int kMaxInfoBuffer = 256;
#define  GBYTES  1073741824    
#define  MBYTES  1048576    
#define  KBYTES  1024    
#define  DKBYTES 1024.0    

static
void GetMemoryInfo(double &memory_avaliable) {
	std::string memory_info;
	MEMORYSTATUSEX statusex;
	statusex.dwLength = sizeof(statusex);

	if (GlobalMemoryStatusEx(&statusex)) {
		unsigned long long total = 0, remain_total = 0, avl = 0, remain_avl = 0;
		double decimal_total = 0, decimal_avl = 0;
        remain_total = statusex.ullTotalPhys % GBYTES;
        total = statusex.ullTotalPhys / GBYTES;
        avl = statusex.ullAvailPhys / GBYTES;
        remain_avl = statusex.ullAvailPhys % GBYTES;
        if (remain_total > 0)
            decimal_total = (remain_total / MBYTES) / DKBYTES;
        if (remain_avl > 0)
            decimal_avl = (remain_avl / MBYTES) / DKBYTES;

        decimal_total += (double)total;
        decimal_avl += (double)avl;
        char  buffer[kMaxInfoBuffer];
        sprintf_s(buffer, kMaxInfoBuffer, "Host memory infomation %.2f(available)/ %.2f GB ", decimal_avl, decimal_total);
        memory_info.append(buffer);
        memory_avaliable = decimal_avl;
	}
    std::cout << memory_info << std::endl;

}

//amount of data in the program.
bool CheckMemoryInfo(int size,int my_np) {
    //How much memory do you need?
    printf("\n*****HOST MEMORY CHECKING*****\n");
    double space;
    GetMemoryInfo(space);
    unsigned long long need_bytes = (sizeof(__DATA_TYPE) * size * size * my_np);
    double need = (sizeof(__DATA_TYPE) * size * size * my_np) * 1.0 / GBYTES;
    printf("Avaliable space is :%.5f GB\n", space);
    printf("The space you need is:%.5f GB\n", need);

    if (need - space > 0.02) {
        printf("Warnning:Avaliable space is not enough!!!\n");
        printf("Would you like to use the matrixs copied first one just for testing? That's a helpful way just spend a little host memory.(Y/N)\n");
        char answer;
        int i = scanf("%c", &answer);
        if (answer == 'Y' || answer == 'y') setConfigString(USE_COPY_MATRIX_C, "1");
        else {
            setConfigString(USE_COPY_MATRIX_C, "0");
            printf("The pragram is failed because of lacking host memory!!!\n");
            exit(0);
        }
        return false;
    }
    else {
        setConfigString(USE_COPY_MATRIX_C, "0");
    }

    return true;
}