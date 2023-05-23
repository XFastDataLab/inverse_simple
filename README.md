# inverse_simple

#### Introduce
The inverse_simple project aims to solve the parallel inverse problem of a large number of small matrices. It can realize the simultaneous inverse of millions of small matrices in a short time. The algorithm is called a Revise In-Place GPU, which adopts a more refined parallelization scheme and outperforms other algorithms,  achieving a speedup of up to 20.9572 times over the batch matrix inverse kernel in CUBLAS. 

Inverting a matrix is time-consuming, and many works focus on accelerating the inversion of a single large matrix by GPU. However, the problem of parallelizing the inversion of a large number of small matrices has received little attention. These problems are widely applied in computer science, including accelerating cryptographic algorithms and image processing algorithms. In this paper, we propose a Revised In-Place Inversion algorithm for inverting a large number of small matrices on the CUDA platform, which adopts a more refined parallelization scheme and outperforms other algorithms, achieving a speedup of up to 20.9572 times over the batch matrix inverse kernel in CUBLAS. Additionally, we found that there is an upper bound on the input data size for each GPU device, and the performance will degrade if the input data size is too large. Therefore, we propose the Saturation Size Curve based on this finding to divide matrices into batches and improve the algorithm performance. Experimental results show that this strategy increases the algorithm's performance by 1.75 times and effectively alleviates the problem of performance degradation.

Please refer to relevant papers： [Fast algorithm for parallel solving inversion of large scale small matrices based on GPU](https://doi.org/10.1007/s11227-023-05336-7)

Cite our paper: Xuebin, J., Yewang, C., Wentao, F. et al. Fast algorithm for parallel solving inversion of large scale small matrices based on GPU. J Supercomput (2023). https://doi.org/10.1007/s11227-023-05336-7

#### Environment

Windows 10.
Microsoft Visual Studio Community 2019 (16.11.19)
CUBLAS API version: 12.1

#### Branch introduce

1.  main: original code
2.  dev_in_place_3060_float:  the Algorithm of In-Place GPU.
3.  dev_cublas_float: the batched inverse kernel of cuBlas API.
4.  dev_cpu_float: the Algorithm of In-Place CPU.
5.  dev_cpu_openMP2: the Algorithm of In-Place OpenMP


#### Guideline

1.  Switch to the corresponding branch and run directly.

