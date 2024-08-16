// CS 683 (Autumn 2023)
// PA 1: The Matrix

// includes
#include <stdio.h>
#include <time.h>	   // for time-keeping
#include <xmmintrin.h> // for intrinsic functions
#include <immintrin.h>
// #include <valgrind/callgrind.h>

// defines
// NOTE: you can change this value as per your requirement
#define BLOCK_SIZE 100	// size of the block
#define SIZE_FOR_SIMD 0 // 0 for 256 1 for 512
#define CorAforTemp 1	// 0 if want to reuse A else 1 for reusing C
#define magic_number 30 // simd_prefetch_mat_mul

/**
 * @brief 		Generates random numbers between values fMin and fMax.
 * @param 		fMin 	lower range
 * @param 		fMax 	upper range
 * @return 		random floating point number
 */
double fRand(double fMin, double fMax)
{

	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_matrix(double *matrix, int rows, int cols)
{

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			matrix[i * cols + j] = fRand(0.0001, 1.0000); // random values between 0 and 1
		}
	}
}

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 */
void normal_mat_mul(double *A, double *B, double *C, int dim)
{

	for (int i = 0; i < dim; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			for (int k = 0; k < dim; k++)
			{
				C[i * dim + j] += A[i * dim + k] * B[k * dim + j];
			}
		}
	}
	// printf("\n Normal mat mul is as follows :\n");
	// for (int i = 0; i < dim; i++)
	// {
	// 	for (int j = 0; j < dim; j++)
	// 	{
	// 		printf("%lf ", C[i * dim + j]);
	// 	}
	// 	printf("\n");
	// }
	// printf("normal matrix by normal A MUL\n");

	// for(int i=0;i<dim;i++)
	// {
	// 	for(int j=0;j<dim;j++)
	// 	{
	// 		printf("%f\t",C[i*dim+j]);
	// 	}
	// 	printf("\n");
	// }
	// 		for(int i=0;i<dim;i++)
	// 		{
	// 			for(int j=0;j<dim;j++)
	// 			{
	// 				printf("%f\t",A[i*dim+j]);
	// 			}
	// 			printf("\n");
	// 		}
	// printf("normal matrix by normal B MUL\n");

	// 		for(int i=0;i<dim;i++)
	// 		{
	// 			for(int j=0;j<dim;j++)
	// 			{
	// 				printf("%f\t",B[i*dim+j]);
	// 			}
	// 			printf("\n");
	// 		}
}

/**
 * @brief 		Task 1: Performs matrix multiplication of two matrices using blocking.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the block size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
 */
// void blocking_mat_mul(double *A, double *B, double *C, int dim, int block_size) {

// 	for(int i = 0;i< dim;i+=block_size)
// 	{
// 		for(int j = 0; j<dim;j+=block_size)
// 		{
// 			for(int k = 0;k<dim;k+=block_size)
// 			{
// 				for(int ib = i; ib < i + block_size; ib++)
// 				{
// 					for(int jb = j; jb < j + block_size; jb++)
// 					{
// 						for(int kb = k; kb < k + block_size; kb++)
// 						{
// 							C[ib*dim + jb] += A[ib*dim + kb]*B[kb*dim +jb];
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }

void blocking_mat_mul(double *A, double *B, double *C, int dim, int block_size)
{
	// int CorAforTemp = 1; // 0 if want to reuse A else 1 for reusing C

	if (CorAforTemp)
	{
		register double tempA ;
		for (int i = 0; i < dim; i += block_size)
		{
			for (int k = 0; k < dim; k += block_size)
			{
				for (int j = 0; j < dim; j += block_size)
				{
					for (int ib = i; ib < i + block_size; ib++)
					{
						for (int jb = j; jb < j + block_size; jb++)
						{
							tempA = 0.0;
							for (int kb = k; kb < k + block_size; kb++)
							{
								tempA += A[ib * dim + kb] * B[kb * dim + jb]; // Reuse tempA
							}
							C[ib * dim + jb] = tempA;
						}

						// else
						// {
						// 	for (int kb = k; kb < k + block_size; kb++)
						// 	{
						// 		double tempA = A[ib * dim + kb];
						// 		for (int jb = j; jb < j + block_size; jb++)

						// 		{
						// 			C[ib * dim + jb] += tempA * B[kb * dim + jb]; // Reuse tempA
						// 		}
						// 	}
						// }
					}
				}
			}
		}
	}
	else
	{
		register double tempA;
		for (int i = 0; i < dim; i += block_size)
		{
			for (int k = 0; k < dim; k += block_size)
			{
				for (int j = 0; j < dim; j += block_size)
				{
					for (int ib = i; ib < i + block_size; ib++)
					{
						for (int kb = k; kb < k + block_size; kb++)
						{
							tempA= A[ib * dim + kb];
							for (int jb = j; jb < j + block_size; jb++)
							{
								C[ib * dim + jb] += tempA * B[kb * dim + jb]; // Reuse tempA
							}
						}
					}
				}
			}
		}
	}

	// printf("\n Matrix Blocked mul is as follows :\n");
	// for (int i = 0; i < dim; i++)
	// {
	// 	for (int j = 0; j < dim; j++)
	// 	{
	// 		printf("%lf ", D[i * dim + j]);
	// 	}
	// 	printf("\n");
	// }
}

/* we need to try different block sizes as per our own cache line size and try to find reason for the speedup. By default, it gives a max speedup of 1.2 to 1.5.
we need to clear the cache before running. Also, the block size needs to be a multiple of 10 if we donot change the matrix size.

Some results and observations :

We can change the block size in this C file and the size of the matrix can be changed in the run.sh file. In my laptop, and I think it should be common for all three of us, the
cache alignment is 64 (cache line size 64B). Therefore, the first optimization that comes to mind intuitively is making the dimensions and block size a multiple of 64, to avoid cache spills.

For Block size : 64.

				Size of matrix 	:   128  	    |		256  		|	  1024
				Speedup			:   0.9823		|		1.441		| 	  4.34552


For Block size : 128.

				Size of matrix 	:   128  	    |		256  		|	  1024
				Speedup			:   1.2264		|		0.9195		| 	  4.661415
*/

/**
 * @brief 		Task 2: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
 */

// Simple code that works correctly for SIMD 128 bit registers - sunanda
// void simd_mat_mul(double *A, double *B, double *C, int dim)
// {
// 	for (int i = 0; i < dim; i++) {
//         for (int j = 0; j < dim; j+=2) {
//             __m128d sum = _mm_setzero_pd();

//             for (int k = 0; k < dim; k += 2) {
//                 // Load A[i][k] and A[i][k+1] using row-major access
//                 __m128d a0 = _mm_load_pd(&A[i*dim + k]);

// 				__m128d b0 = _mm_loadu_pd(&B[k*dim + j]);
// 				__m128d b1 = _mm_loadu_pd(&B[(k+1) * dim + j]);

// 				sum = _mm_add_pd(sum, _mm_mul_pd(_mm_shuffle_pd(a0, a0, 0x00), b0));
//          		sum = _mm_add_pd(sum, _mm_mul_pd(_mm_shuffle_pd(a0, a0, 0xff), b1));
// 				// sum = _mm_shuffle_pd(a0, a0, 0xff);

//             }
// 			// sum = _mm_shuffle_pd(a0, a0, 0xff);

//             _mm_storeu_pd(&C[i * dim + j], sum);
// 		}
// 	}
// 		printf("SIMD matrix MUL\n");

// }

// Simple code that works correctly for SIMD 256 bit registers
void simd_mat_mul(double *A, double *B, double *C, int dim)
{
	if (SIZE_FOR_SIMD == 0)
	{
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j += 4)
			{
				__m256d sum = _mm256_setzero_pd();

				for (int k = 0; k < dim; k += 4)
				{
					// __m256d a0 = _mm256_load_pd(&A[i*dim + k]);

					__m256d b0 = _mm256_loadu_pd(&B[k * dim + j]);
					__m256d b1 = _mm256_loadu_pd(&B[(k + 1) * dim + j]);
					__m256d b2 = _mm256_loadu_pd(&B[(k + 2) * dim + j]);
					__m256d b3 = _mm256_loadu_pd(&B[(k + 3) * dim + j]);

					sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * dim + k]), b0));
					sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * dim + k + 1]), b1));
					sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * dim + k + 2]), b2));
					sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * dim + k + 3]), b3));
				}
				_mm256_storeu_pd(&C[i * dim + j], sum);
			}
		}
	}

	else
	{
		for (int i = 0; i < dim; i++)
		{
			// printf("inside 512 simd 1\n");
			for (int j = 0; j < dim; j += 8)
			{
				// printf("inside 512 simd 2\n");

				__m512d sum = _mm512_setzero_pd();

				for (int k = 0; k < dim; k += 8)
				{
					// printf("inside 512 simd 3\n");

					// __m512d a0 = _mm512_load_pd(&A[i*dim + k]);

					__m512d b0 = _mm512_loadu_pd(&B[k * dim + j]);
					__m512d b1 = _mm512_loadu_pd(&B[(k + 1) * dim + j]);
					__m512d b2 = _mm512_loadu_pd(&B[(k + 2) * dim + j]);
					__m512d b3 = _mm512_loadu_pd(&B[(k + 3) * dim + j]);
					__m512d b4 = _mm512_loadu_pd(&B[(k + 4) * dim + j]);
					__m512d b5 = _mm512_loadu_pd(&B[(k + 5) * dim + j]);
					__m512d b6 = _mm512_loadu_pd(&B[(k + 6) * dim + j]);
					__m512d b7 = _mm512_loadu_pd(&B[(k + 7) * dim + j]);

					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k]), b0));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 1]), b1));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 2]), b2));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 3]), b3));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 4]), b4));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 5]), b5));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 6]), b6));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 7]), b7));
				}
				_mm512_storeu_pd(&C[i * dim + j], sum);
			}
		}
	}

	// printf("\n Matrix Blocked-simd mul is as follows :\n");
	// for (int i = 0; i < dim; i++)
	// {
	// 	for (int j = 0; j < dim; j++)
	// 	{
	// 		printf("%lf ", D[i * dim + j]);
	// 	}
	// 	printf("\n");
	// }
}


/**
 * @brief 		Task 3: Performs matrix multiplication of two matrices using software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
 */

void prefetch_mat_mul(double *A, double *B, double *C, int dim)
{
	// const int prefetch_distance = 128; // Based on cache line size
	// int magic_number = 20; // should be less than dimension and factor of dimension
	register double sum ;
	for (int i = 0; i < dim; i++)
	{
		for (int p = 0; p < dim; p += 8)
		{
			__builtin_prefetch(&A[(i)*dim + p], 0, 3);
		}
		for (int j = 0; j < dim; j++)
		{
			sum = 0.0;
			for (int k = 0; k < dim; k += 1)
			{
				sum += A[i * dim + k] * B[k * dim + j];
			}
			C[i * dim + j] = sum;
		}
	}
	// printf("\n Normal mat mul is as follows :\n");
	// for (int i = 0; i < dim; i++)
	// {
	// 	for (int j = 0; j < dim; j++)
	// 	{
	// 		printf("%lf ", D[i * dim + j]);
	// 	}
	// 	printf("\n");
	// }
}

/**
 * @brief 		Bonus Task 1: Performs matrix multiplication of two matrices using blocking along with SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 */
void blocking_simd_mat_mul(double *A, double *B, double *C, int dim, int block_size)
{
	if (SIZE_FOR_SIMD == 0)
	{
		for (int i = 0; i < dim; i += block_size)
		{
			for (int k = 0; k < dim; k += block_size)
			{
				for (int j = 0; j < dim; j += block_size)
				{
					for (int ii = i; ii < i + block_size; ii++)
					{
						for (int jj = j; jj < j + block_size; jj += 4)
						{
							__m256d sum = _mm256_setzero_pd();
							for (int kk = k; kk < k + block_size; kk += 4)
							{
								__m256d b0 = _mm256_loadu_pd(&B[kk * dim + jj]);
								__m256d b1 = _mm256_loadu_pd(&B[(kk + 1) * dim + jj]);
								__m256d b2 = _mm256_loadu_pd(&B[(kk + 2) * dim + jj]);
								__m256d b3 = _mm256_loadu_pd(&B[(kk + 3) * dim + jj]);

								// __m256d a = _mm256_broadcast_sd(&A[ii * dim + kk]);

								sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[ii * dim + kk]), b0));
								sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[ii * dim + kk + 1]), b1));
								sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[ii * dim + kk + 2]), b2));
								sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[ii * dim + kk + 3]), b3));
							}
							_mm256_storeu_pd(&C[ii * dim + jj], _mm256_add_pd(_mm256_loadu_pd(&C[ii * dim + jj]), sum));
						}
					}
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < dim; i += block_size)
		{
			for (int k = 0; k < dim; k += block_size)
			{
				for (int j = 0; j < dim; j += block_size)
				{
					for (int ii = i; ii < i + block_size; ii++)
					{
						for (int jj = j; jj < j + block_size; jj += 8)
						{
							__m512d sum = _mm512_setzero_pd();
							for (int kk = k; kk < k + block_size; kk += 8)
							{
								// __m512d a0 = _mm512_load_pd(&A[i*dim + kk]);

								__m512d b0 = _mm512_loadu_pd(&B[kk * dim + jj]);
								__m512d b1 = _mm512_loadu_pd(&B[(kk + 1) * dim + jj]);
								__m512d b2 = _mm512_loadu_pd(&B[(kk + 2) * dim + jj]);
								__m512d b3 = _mm512_loadu_pd(&B[(kk + 3) * dim + jj]);
								__m512d b4 = _mm512_loadu_pd(&B[(kk + 4) * dim + jj]);
								__m512d b5 = _mm512_loadu_pd(&B[(kk + 5) * dim + jj]);
								__m512d b6 = _mm512_loadu_pd(&B[(kk + 6) * dim + jj]);
								__m512d b7 = _mm512_loadu_pd(&B[(kk + 7) * dim + jj]);

								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk]), b0));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 1]), b1));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 2]), b2));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 3]), b3));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 4]), b4));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 5]), b5));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 6]), b6));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 7]), b7));
							}
							_mm512_storeu_pd(&C[ii * dim + jj], _mm512_add_pd(_mm512_loadu_pd(&C[ii * dim + jj]), sum));
						}
					}
				}
			}
		}
	}
}

/**
 * @brief 		Bonus Task 2: Performs matrix multiplication of two matrices using blocking along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 */
void blocking_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size)
{
	if (CorAforTemp)
	{
		register double tempA;
		for (int i = 0; i < dim; i += block_size)
		{
			for (int p = 0; p < dim; p+=8)
					{
						for (int s = 0; s < magic_number; s += 1)
						{
							// __builtin_prefetch(&A[ib * dim + kb], 0, 1);
							__builtin_prefetch(&B[(i+magic_number + s) * dim + p], 0, 1);
						}
					}
			for (int k = 0; k < dim; k += block_size)
			{
				for (int j = 0; j < dim; j += block_size)
				{
				
					for (int ib = i; ib < i + block_size; ib++)
					{
						for (int jb = j; jb < j + block_size; jb++)
						{
							tempA = 0.0;
							for (int kb = k; kb < k + block_size; kb++)
							{
								tempA += A[ib * dim + kb] * B[kb * dim + jb]; // Reuse tempA
							}
							C[ib * dim + jb] = tempA;
						}
					}
				}
			}
		}
	}
	else
	{
		register double tempA ;
		for (int i = 0; i < dim; i += block_size)
		{
			for (int p = 0; p < dim; p+=8)
					{
						for (int s = 0; s < magic_number; s += 1)
						{
							// __builtin_prefetch(&A[ib * dim + kb], 0, 1);
							__builtin_prefetch(&B[(i+magic_number + s) * dim + p], 0, 1);
						}
					}
			for (int k = 0; k < dim; k += block_size)
			{
				for (int j = 0; j < dim; j += block_size)
				{
					for (int ib = i; ib < i + block_size; ib++)
					{
						for (int kb = k; kb < k + block_size; kb++)
						{
							tempA = A[ib * dim + kb];
							for (int jb = j; jb < j + block_size; jb++)

							{
								C[ib * dim + jb] += tempA * B[kb * dim + jb]; // Reuse tempA
							}
						}
					}
				}
			}
		}
	}
}

/**
 * @brief 		Bonus Task 3: Performs matrix multiplication of two matrices using SIMD instructions along with software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
 */
void simd_prefetch_mat_mul(double *A, double *B, double *C, int dim)
{
	// int magic_number = 20;
	if (SIZE_FOR_SIMD == 0)
	{
		for (int i = 0; i < dim; i++)
		{
			for (int p = 0; p < dim; p += 8)
			{
				// __builtin_prefetch(&A[(i) * dim + p], 0, 3);

				for (int s = 0; s < magic_number; s += 1)
				{
					__builtin_prefetch(&B[(i * magic_number + s) * dim + p], 0, 3);
				}
			}
			for (int j = 0; j < dim; j += 4)
			{
				__m256d sum = _mm256_setzero_pd();

				for (int k = 0; k < dim; k += 4)
				{
					// __m256d a0 = _mm256_load_pd(&A[i*dim + k]);

					__m256d b0 = _mm256_loadu_pd(&B[k * dim + j]);
					__m256d b1 = _mm256_loadu_pd(&B[(k + 1) * dim + j]);
					__m256d b2 = _mm256_loadu_pd(&B[(k + 2) * dim + j]);
					__m256d b3 = _mm256_loadu_pd(&B[(k + 3) * dim + j]);

					sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * dim + k]), b0));
					sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * dim + k + 1]), b1));
					sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * dim + k + 2]), b2));
					sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * dim + k + 3]), b3));
				}
				_mm256_storeu_pd(&C[i * dim + j], sum);
			}
		}
	}
	else
	{
		for (int i = 0; i < dim; i++)
		{
			for (int p = 0; p < dim; p += 8)
			{

				for (int s = 0; s < magic_number; s += 1)
				{
					__builtin_prefetch(&B[(i * magic_number + s) * dim + p], 0, 3);
					__builtin_prefetch(&A[(i * magic_number + s) * dim + p], 0, 3);
					__builtin_prefetch(&C[(i * magic_number + s) * dim + p], 1, 1);
				}
			}
			for (int j = 0; j < dim; j += 8)
			{
				__m512d sum = _mm512_setzero_pd();

				for (int k = 0; k < dim; k += 8)
				{
					// __m512d a0 = _mm512_load_pd(&A[i*dim + k]);

					__m512d b0 = _mm512_loadu_pd(&B[k * dim + j]);
					__m512d b1 = _mm512_loadu_pd(&B[(k + 1) * dim + j]);
					__m512d b2 = _mm512_loadu_pd(&B[(k + 2) * dim + j]);
					__m512d b3 = _mm512_loadu_pd(&B[(k + 3) * dim + j]);
					__m512d b4 = _mm512_loadu_pd(&B[(k + 4) * dim + j]);
					__m512d b5 = _mm512_loadu_pd(&B[(k + 5) * dim + j]);
					__m512d b6 = _mm512_loadu_pd(&B[(k + 6) * dim + j]);
					__m512d b7 = _mm512_loadu_pd(&B[(k + 7) * dim + j]);

					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k]), b0));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 1]), b1));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 2]), b2));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 3]), b3));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 4]), b4));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 5]), b5));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 6]), b6));
					sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[i * dim + k + 7]), b7));
				}
				_mm512_storeu_pd(&C[i * dim + j], sum);
			}
		}
	}
}

/**
 * @brief 		Bonus Task 4: Performs matrix multiplication of two matrices using blocking along with SIMD instructions and software prefetching.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		dim 		dimension of the matrices
 * @param 		block_size 	size of the block
 * @note 		The block size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
 */
void blocking_simd_prefetch_mat_mul(double *A, double *B, double *C, int dim, int block_size)
{
	if (SIZE_FOR_SIMD == 0)
	{
		for (int i = 0; i < dim; i += block_size)
		{
			for (int p = 0; p < dim; p+=8)
					{
						for (int s = 0; s < magic_number; s += 1)
						{
							// __builtin_prefetch(&A[ib * dim + kb], 0, 1);
							__builtin_prefetch(&B[(i+magic_number + s) * dim + p], 0, 3);
						}
					}
			for (int k = 0; k < dim; k += block_size)
			{
				for (int j = 0; j < dim; j += block_size)
				{				
					for (int ii = i; ii < i + block_size; ii++)
					{
						for (int jj = j; jj < j + block_size; jj += 4)
						{
							__m256d sum = _mm256_setzero_pd();
							for (int kk = k; kk < k + block_size; kk += 4)
							{
								__m256d b0 = _mm256_loadu_pd(&B[kk * dim + jj]);
								__m256d b1 = _mm256_loadu_pd(&B[(kk + 1) * dim + jj]);
								__m256d b2 = _mm256_loadu_pd(&B[(kk + 2) * dim + jj]);
								__m256d b3 = _mm256_loadu_pd(&B[(kk + 3) * dim + jj]);

								// __m256d a = _mm256_broadcast_sd(&A[ii * dim + kk]);

								sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[ii * dim + kk]), b0));
								sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[ii * dim + kk + 1]), b1));
								sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[ii * dim + kk + 2]), b2));
								sum = _mm256_add_pd(sum, _mm256_mul_pd(_mm256_broadcast_sd(&A[ii * dim + kk + 3]), b3));
							}
							_mm256_storeu_pd(&C[ii * dim + jj], _mm256_add_pd(_mm256_loadu_pd(&C[ii * dim + jj]), sum));
						}
					}
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < dim; i += block_size)
		{
			for (int p = 0; p < dim; p+=8)
					{
						for (int s = 0; s < magic_number; s += 1)
						{
							// __builtin_prefetch(&A[ib * dim + kb], 0, 1);
							__builtin_prefetch(&B[(i+magic_number + s) * dim + p], 0, 1);
						}
					}
			for (int k = 0; k < dim; k += block_size)

			{
				for (int j = 0; j < dim; j += block_size)

				{
					for (int ii = i; ii < i + block_size; ii++)
					{
						for (int jj = j; jj < j + block_size; jj += 4)
						{
							__m512d sum = _mm512_setzero_pd();
							for (int kk = k; kk < k + block_size; kk += 8)
							{
								// __m512d a0 = _mm512_load_pd(&A[i*dim + kk]);

								__m512d b0 = _mm512_loadu_pd(&B[kk * dim + jj]);
								__m512d b1 = _mm512_loadu_pd(&B[(kk + 1) * dim + jj]);
								__m512d b2 = _mm512_loadu_pd(&B[(kk + 2) * dim + jj]);
								__m512d b3 = _mm512_loadu_pd(&B[(kk + 3) * dim + jj]);
								__m512d b4 = _mm512_loadu_pd(&B[(kk + 4) * dim + jj]);
								__m512d b5 = _mm512_loadu_pd(&B[(kk + 5) * dim + jj]);
								__m512d b6 = _mm512_loadu_pd(&B[(kk + 6) * dim + jj]);
								__m512d b7 = _mm512_loadu_pd(&B[(kk + 7) * dim + jj]);

								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk]), b0));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 1]), b1));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 2]), b2));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 3]), b3));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 4]), b4));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 5]), b5));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 6]), b6));
								sum = _mm512_add_pd(sum, _mm512_mul_pd(_mm512_set1_pd(A[ii * dim + kk + 7]), b7));
							}
							_mm512_storeu_pd(&C[ii * dim + jj], _mm512_add_pd(_mm512_loadu_pd(&C[ii * dim + jj]), sum));
						}
					}
				}
			}
		}
	}
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
 */
int main(int argc, char **argv)
{

	if (argc <= 1)
	{
		printf("Pass the matrix dimension as argument :)\n\n");
		return 0;
	}

	else
	{
		int matrix_dim = atoi(argv[1]);

		// variables definition and initialization
		clock_t t_normal_mult, t_blocking_mult, t_prefetch_mult, t_simd_mult, t_blocking_simd_mult, t_blocking_prefetch_mult, t_simd_prefetch_mult, t_blocking_simd_prefetch_mult;
		double time_normal_mult, time_blocking_mult, time_prefetch_mult, time_simd_mult, time_blocking_simd_mult, time_blocking_prefetch_mult, time_simd_prefetch_mult, time_blocking_simd_prefetch_mult;

		double *A = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));
		double *B = (double *)malloc(matrix_dim * matrix_dim * sizeof(double));
		double *C = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));
		// double *D = (double *)calloc(matrix_dim * matrix_dim, sizeof(double));

		// for (int i = 0; i < matrix_dim; i++)
		// {
		// 	for (int j = 0; j < matrix_dim; j++)
		// 	{
		// 		C[i * matrix_dim + j] = 0;
		// 	}
		// }

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, matrix_dim, matrix_dim);
		initialize_matrix(B, matrix_dim, matrix_dim);

		// perform normal matrix multiplication
		t_normal_mult = clock();
		normal_mat_mul(A, B, C, matrix_dim);
		t_normal_mult = clock() - t_normal_mult;

		time_normal_mult = ((double)t_normal_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Normal matrix multiplication took %f seconds to execute \n\n", time_normal_mult);

#ifdef OPTIMIZE_BLOCKING
		// Task 1: perform blocking matrix multiplication

		t_blocking_mult = clock();
		blocking_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_mult = clock() - t_blocking_mult;

		time_blocking_mult = ((double)t_blocking_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking matrix multiplication took %f seconds to execute \n", time_blocking_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_mult);
#endif

#ifdef OPTIMIZE_SIMD
		// Task 2: perform matrix multiplication with SIMD instructions
		t_simd_mult = clock();
		simd_mat_mul(A, B, C, matrix_dim);
		t_simd_mult = clock() - t_simd_mult;

		time_simd_mult = ((double)t_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD matrix multiplication took %f seconds to execute \n", time_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_mult);
		// printf("SIMD done\n");

#endif

#ifdef OPTIMIZE_PREFETCH
		// Task 3: perform matrix multiplication with prefetching
		// CALLGRIND_START_INSTRUMENTATION;
		// CALLGRIND_TOGGLE_COLLECT;
		t_prefetch_mult = clock();
		prefetch_mat_mul(A, B, C, matrix_dim);
		t_prefetch_mult = clock() - t_prefetch_mult;
		// CALLGRIND_TOGGLE_COLLECT;
		// CALLGRIND_STOP_INSTRUMENTATION;
		time_prefetch_mult = ((double)t_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Prefetching matrix multiplication took %f seconds to execute \n", time_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_prefetch_mult);
#endif

#ifdef OPTIMIZE_BLOCKING_SIMD
		// Bonus Task 1: perform matrix multiplication using blocking along with SIMD instructions
		t_blocking_simd_mult = clock();
		blocking_simd_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_mult = clock() - t_blocking_simd_mult;

		time_blocking_simd_mult = ((double)t_blocking_simd_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD matrix multiplication took %f seconds to execute \n", time_blocking_simd_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_mult);
#endif

#ifdef OPTIMIZE_BLOCKING_PREFETCH
		// Bonus Task 2: perform matrix multiplication using blocking along with software prefetching
		t_blocking_prefetch_mult = clock();
		blocking_prefetch_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_prefetch_mult = clock() - t_blocking_prefetch_mult;

		time_blocking_prefetch_mult = ((double)t_blocking_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with prefetching matrix multiplication took %f seconds to execute \n", time_blocking_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_prefetch_mult);
#endif

#ifdef OPTIMIZE_SIMD_PREFETCH
		// Bonus Task 3: perform matrix multiplication using SIMD instructions along with software prefetching
		t_simd_prefetch_mult = clock();
		simd_prefetch_mat_mul(A, B, C, matrix_dim);
		t_simd_prefetch_mult = clock() - t_simd_prefetch_mult;

		time_simd_prefetch_mult = ((double)t_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("SIMD with prefetching matrix multiplication took %f seconds to execute \n", time_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_simd_prefetch_mult);
#endif

#ifdef OPTIMIZE_BLOCKING_SIMD_PREFETCH
		// Bonus Task 4: perform matrix multiplication using blocking, SIMD instructions and software prefetching
		t_blocking_simd_prefetch_mult = clock();
		blocking_simd_prefetch_mat_mul(A, B, C, matrix_dim, BLOCK_SIZE);
		t_blocking_simd_prefetch_mult = clock() - t_blocking_simd_prefetch_mult;

		time_blocking_simd_prefetch_mult = ((double)t_blocking_simd_prefetch_mult) / CLOCKS_PER_SEC; // in seconds
		printf("Blocking with SIMD and prefetching matrix multiplication took %f seconds to execute \n", time_blocking_simd_prefetch_mult);
		printf("Normalized performance: %f \n\n", time_normal_mult / time_blocking_simd_prefetch_mult);
#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);
		return 0;
	}
}
