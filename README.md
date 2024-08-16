[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/mnOJa0WY)
# PA1-The-Matrix
Highlights: 
Perform matrix multiplication using various optimization techniques:
- blocked matrix multiplication
- SIMD instructions
- software prefetching

and combination of these techniques.

we have defined 3 macros :
SIZE_FOR_SIMD : To choose whether to use 256 bit for SIMD or 256 bit register for SIMD. ( 0 -> 256 bit, 1 -> 512 bit)
CorAforTemp : To choose between the two variants of blocked mat mul.
magic_number : Number of rows to be prefetched in prefetching technique

A bit on the "Magic Number"
Magic number: loop from 0 to magic number: brings # row blocks = magic number for matrix B 
This helps in SIMD becuase we fetch B columnwise. So if we prefetch he entire row beforehand it's early enough to hide the latency of fetch.


<br>

---
## Task 1: Blocked Matrix Multiplication
The first optimization that we tried was changing the loop order from ijk to ikj. By doing so, we are utilizing the cache spatial locality better than before. 

Instead of using normal variable to store the value of A[i*dim  + j], we are using register variable. This is because the compiler would get a hint to keep the intensively used value of temp in a register rather than in a memory address so that we can fetch it faster.

Here, we are using two variations, on one of the case, we have named them A and C. In variant A , we reduce the number of write access to the final matrix C, but we cannot utilize the spatial locality. In variant C, we can utilize the spatial locality of cache but we sacrifice the number of write accesses of matrix C. 


<br>

---
## Task 2: SIMD instructions

In SIMD, we are using 256 bit registers. Now we can calculate the result of 4 matrix elements at the same time, we sould get a speed up of 4 theoritically, but since there can be some processing delays and queueing delays, we get an approximate speedup of near 4. The actual values range from 3.50 - 3.85. Since we are already using registers to store values, we cannot optimize much on the data access once they have been brought to the registers.

<br>

---
## Task 3: Software Prefetching

In simple software prefetching, we are trying to prefetch all the values of one particular value of matrix A in the cache. This is because we are iterate over that particular row for matrix A. Since prefetcher fetches the values in units of cache line size and the cache size is of 64 bytes. It can accomodate 8 boubles. The loop therefore hops 8 positions after each iterations.After doing this, we would no longer be going to the main memory to fetch the matrix values when the program demands the value of matrix A. 

        
    for(int i = 0; i<dim; i++)
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
				sum += A[i * dim + k] * B[k * dim + j]; // A[i,k] getting prefetched in L1 cache.
			}
			C[i * dim + j] = sum;
		}
    }


<br>

---
## Bonus Task 1: Blocked Matrix Multiplication + SIMD instructions
Here we simply employ the SIMD registers in the blocked matrix multiplication code. The only thing that needs to change in the code of SIMD is the way the answer for each set of 4 cells is being stored. In normal SIMD code, we perform a write to any cell of matrix C only when the final answer of that cell is obtained. But in Blocked SIMD, we calculate the answer to one set of 4 cells of matrix C in multiple iterations. Therefore we need to keep adding the answer to the previously caclulated result in those values.

We need to be careful while choosing the block size. The block size should be divisible by 4 as we are usig 256 bit registers which store 4 values at a time & it should also divide the matrix dimension. This is to avoid segmentation fault. 
<br>

---
## Bonus Task 2: Blocked Matrix Multiplication + Software Prefetching

The blocked method takes place as usual, with the addition of prefetching technique.

In this technique, we prefetch values of matrix B. The magic number denots the number of rows ahead from the current point of execution that we are fetching. The optimal value that we obtained for this number was 30. 

        for (int s = 0; s < magic_number; s += 1)
		{			
			__builtin_prefetch(&B[(i+magic_number + s) * dim + p], 0, 1);
		}

<br>

---
## Bonus Task 3: SIMD instructions + Software Prefetching

In this technique also, the SIMD code runs as ususal, just with the addition of prefetching happening before we put the values of the matrix in the SIMD registers.

        for (int s = 0; s < magic_number; s += 1)
		{
			__builtin_prefetch(&B[(i * magic_number + s) * dim + p], 0, 3);
		}
Since the prefetched values are present in the cache, it would save main memory access.

<br>

---
## Bonus Task 4: Bloced Matrix Multiplication + SIMD instructions + Software Prefetching

In this technique, we combine all the techniques as described above. The matrix is divided into blocks that fit into the cache. SIMD registers fetch the values parallely and prefetcher does it's usual job. Combinely, this gives the best results. An astonishing speedup of around 6x is obtained.
<br>

The Graph of speedup looks like this :
[[![Speedup Graph](https://github.com/cs683-iitb-autumn-2023/pa1-the-matrix-aca/assets/142026579/56c1415a-8cb3-4390-95ca-09b4678dac79)](https://github.com/mk17499/Matrix_Multiplication_optimized/issues/1#issue-2469970224)
---](https://github.com/mk17499/Matrix_Multiplication_optimized/issues/1#issue-2469970224)
All the best! :smile:
