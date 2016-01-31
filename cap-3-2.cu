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
/*a is a matrix
b is a vector
c is a result (vector)*/
__global__ void matrixVectMult(float a[], float b[], float c[], int N) {
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   int i;
   float sum = 0;
   if(index < N)
   {
      for(i = 0; i < N; i++)
         sum += a[(i*N)+index]*b[i];
      c[index] = sum;   
   }
   
}

void Read_matrix(float A[], int s) {
   int i, j;

   for (i = 0; i < s; i++)
      for (j = 0; j < s; j++)
         scanf("%f", &A[i*s+j]);
}

void Read_vector(float A[], int s) {
   int i;

   for (i = 0; i < s; i++)
      scanf("%f", &A[i]);
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
void Print_vector(char name[], float A[], int s) {
   int i;

   printf("%s\n", name);
   for (i = 0; i < s; i++) {
      printf("%.1f ", A[i]);
      printf("\n");
   }  
}
/* Host */
int main(int argc, char* argv[]) {
   int N;
   int size_matrix;
   int size_vector;
   float *dev_a, *dev_b, *dev_c;
   float *a, *b, *c;

   N = strtol(argv[1], NULL, 10);
   printf("size = %d", N);

   size_matrix = N*N*sizeof(float);
   size_vector = N*sizeof(float);


   a = (float*) malloc(size_matrix);
   b = (float*) malloc(size_vector);
   c = (float*) malloc(size_vector); 

   printf("Matrix A: \n");
   Read_matrix(a, N);
   printf("Vector B: \n");
   Read_vector(b, N);

   Print_matrix("A =", a, N);
   Print_vector("B =", b, N);


   cudaMalloc(&dev_a, size_matrix);
   cudaMalloc(&dev_b, size_vector);
   cudaMalloc(&dev_c, size_vector);

   cudaMemcpy(dev_a, a, size_matrix, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_b, b, size_vector, cudaMemcpyHostToDevice);

//   dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
 //  dim3 dimGrid((int)ceil(N/dimBlock.x),(int)ceil(N/dimBlock.y));

   matrixVectMult<<<N,N>>>(dev_a,dev_b,dev_c,N);
   /* Wait for the kernel to complete */
   //cudaThreadSynchronize();

   cudaMemcpy(c, dev_c, size_vector, cudaMemcpyDeviceToHost);
   Print_vector("Result =", c, N);

   cudaFree(dev_a);
   cudaFree(dev_b);
   cudaFree(dev_c);

   free(a);
   free(b);
   free(c);

   return 0;
}