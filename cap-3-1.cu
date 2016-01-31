/*
 * Compile:  nvcc [-g] [-G] -arch=sm_21 -o mat_add mat_add.cu 
 * Run:      ./mat_add <m> <n>
 *              m is the number of rows
 *              n is the number of columns
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define BLOCK_DIM 512

/*
[1 1 1]
[1 1 1]    =>   [1 1 1][1 1 1][1 1 1]
[1 1 1]

*/

/*Kernel*/
__global__ void matrixAdd(float a[], float b[], float c[], int N) {
   int index = blockDim.x * blockIdx.x + threadIdx.x;

   if (blockIdx.x < N && threadIdx.x < N) 
      c[index] = a[index] + b[index];
}
__global__ void matrixAddRow(float a[], float b[], float c[], int N) {
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   for(index = index * N; index < N*N; index++)
   {
      c[index] = a[index] + b[index];
   }
}
__global__ void matrixAddColumn(float a[], float b[], float c[], int N) {
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   for(; index < N*N; index = index + N)
   {
      c[index] = a[index] + b[index];
   }
}

void Read_matrix(float A[], int s) {
   int i, j;

   for (i = 0; i < s; i++)
      for (j = 0; j < s; j++)
         scanf("%f", &A[i*s+j]);
}
void Print_matrix(char name[], float A[], int s) {
   int i, j;

   printf("%s\n", name);
   for (i = 0; i < s; i++) {
      for (j = 0; j < s; j++)
         printf("%.1f ", A[i*s+j]);
      printf("\n");
   }  
}
/* Host */
int main(int argc, char* argv[]) {
   int N;
   int size;
   float *dev_a, *dev_b, *dev_c;
   float *a, *b, *c;

   N = strtol(argv[1], NULL, 10);
   printf("size = %d", N);

   size = N*N*sizeof(float);


   a = (float*) malloc(size);
   b = (float*) malloc(size);
   c = (float*) malloc(size); 

   printf("Matriz A: \n");
   Read_matrix(a, N);
   printf("Matriz B: \n");
   Read_matrix(b, N);

   Print_matrix("A =", a, N);
   Print_matrix("B =", b, N);


   cudaMalloc(&dev_a, size);
   cudaMalloc(&dev_b, size);
   cudaMalloc(&dev_c, size);

   cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

//   dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
//   dim3 dimGrid((int)ceil(N/dimBlock.x),(int)ceil(N/dimBlock.y));

//   matrixAddColumn<<<N,N>>>(dev_a,dev_b,dev_c,N);
//   matrixAddRow<<<N,N>>>(dev_a,dev_b,dev_c,N);
   matrixAdd<<<N,N>>>(dev_a,dev_b,dev_c,N);

   cudaThreadSynchronize();

   cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
   Print_matrix("Result =", c, N);

   cudaFree(dev_a);
   cudaFree(dev_b);
   cudaFree(dev_c);

   free(a);
   free(b);
   free(c);

   return 0;
}