#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <string>

#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)
//#define SIZE1 512
//#define SIZE2 1024 
//#define SIZE3 2048
#define TILE_SIZE1 8  
#define TILE_SIZE2 16 // column/row size in a tile. It can only be 1,2,4,8,16 because of the limitation from the GPU 

using namespace std;
// C version of matrix multiplcation. Use this function for result validation and execution time comaprison
void matrix_mul_openCl(int *A, int *B, int *C, int *C_seq, string p_type, bool isTile, size_t SIZE, size_t TILE_SIZE);
void matrix_mul_sequence (int *A_mat,
                          int *B_mat,
                          int *C_mat,int SIZE)
{
	std::chrono::high_resolution_clock::time_point t1, t2;
	t1 = std::chrono::high_resolution_clock::now();
	for (int j=0; j<SIZE; j++) {
		for (int i=0; i<SIZE; i++)
			for (int k=0; k<SIZE; k++)
				C_mat[j*SIZE + i] += A_mat[j*SIZE + k] * B_mat[k*SIZE + i];
	}
	t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Reference C matrix multiplication: "
		<< (float)(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000
		<< " sec"
		<< std::endl;
}

void initialize_matrix(int *A, int *B, int *C, int *C_seq, int SIZE)
 {
	 //Initialize matrix
	srand(time(0));
    for(int j=0; j<SIZE; j++) {
		for(int i=0; i<SIZE; i++) {
			A[j*SIZE + i] = ((rand()%41) -20);
			B[j*SIZE + i] = ((rand()%41) -20);
			C[j*SIZE + i] = 0;
			C_seq[j*SIZE + i] = 0;
		}
    }
 }


int main(void)
{
    // A, B are input matrix, C is the output matrix for OpenCL, C_seq is the output matrix for reference implementation.
	//creating an array of three different sizes of matrix
	int mat_size[3] = {512, 1024, 2048};
	//Below loop executes three times for three different sizes of matrix
	for(int i: mat_size)
	{	
		int SIZE1 = i;
		int *A = new int[SIZE1*SIZE1];
		int *B = new int[SIZE1*SIZE1];
		int *C = new int[SIZE1*SIZE1];
		int *C_seq = new int[SIZE1*SIZE1];
		// calling initialize function to initialize matrix of required size
		initialize_matrix(A, B, C, C_seq, SIZE1);  
		std::cout << "Running for MatrixSize "<< SIZE1 << std::endl;
		// Calling Reference Multiplication 
		std::cout << "Running reference multiplication  " << std::endl;
		matrix_mul_sequence(A, B, C_seq, SIZE1);
		
		std::cout << "Running naive version for CPU  " << std::endl;
		matrix_mul_openCl(A, B, C, C_seq, "CPU" , 0, SIZE1, TILE_SIZE2); // Passing tile size=16 for naive matrix multiplication
		
		std::cout << "Running naive version for GPU  " << std::endl;
		matrix_mul_openCl(A, B, C, C_seq, "GPU" , 0, SIZE1, TILE_SIZE2); // Passing tile size=16 for naive matrix multiplication
		
		std::cout << "Running tiled version for CPU with tile size " <<TILE_SIZE1 << std::endl;
		matrix_mul_openCl(A, B, C, C_seq, "CPU" , 1, SIZE1, TILE_SIZE1);
		
		std::cout << "Running tiled version for GPU with tile size " <<TILE_SIZE1 << std::endl;
		matrix_mul_openCl(A, B, C, C_seq, "GPU" , 1, SIZE1, TILE_SIZE1);
		
		std::cout << "Running tiled version for CPU with tile size " <<TILE_SIZE2 << std::endl;
		matrix_mul_openCl(A, B, C, C_seq, "CPU" , 1, SIZE1, TILE_SIZE2);
		
		std::cout << "Running tiled version for GPU with tile size " <<TILE_SIZE2 << std::endl;
		matrix_mul_openCl(A, B, C, C_seq, "GPU" , 1, SIZE1, TILE_SIZE2); 
	
	}
	std::cout << "Press Enter to finish..." << std::endl;
	getchar();
 
   return 0;
}	

/*This function loads kernnel source code, create kernel and execute it according to the below mentioned input parameters.
*Four matrix are being passed to this function. It computes multiplicatoin of A and B and store it in matrix C. It verifies its results with matrix Cseq. 
* Parameter p_type decides the platform i.e. CPU/GPU. 
* Parameter isTile =1 will execute tiled kernel whereas isTile =0 will execute naive matrix multiplication.
* It also accepts SIZE and TILE_SIZE as input parameters required for matrix multiplication
*/
void matrix_mul_openCl(int *A, int *B, int *C, int *C_seq, string p_type, bool isTile, size_t SIZE, size_t TILE_SIZE)
{
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("matrix_mul.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
    cl_device_id device_id = NULL; // It is an API supported by openCL to expose _cl_platform_id * abstract datatype.
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, NULL, &ret_num_platforms); // returns no. of platforms available in &ret_num_platforms.
	//Returns CL_SUCCESS if the function is executed successfully. 
	
    cl_platform_id *platform_id = new cl_platform_id[ret_num_platforms]; // array of no. of available platforms
	
	
	ret = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
	
    // There can be multiple platforms such as intel and nvidia. Search for GPU/CPU device in each platform until we find one.
  
	for (int i=0; i<ret_num_platforms; i++) {
	    if(p_type=="CPU")
			ret = clGetDeviceIDs( platform_id[i], CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
		else
			ret = clGetDeviceIDs( platform_id[i], CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
			
        if (ret == CL_SUCCESS)
            break;
    }

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue with the capability of performance profiling for target device
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    // Create memory buffers on the device for each matrix
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE*SIZE*sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE*SIZE*sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE*SIZE*sizeof(int), NULL, &ret);

    // Copy the matrix A, B and C to each device memory counterpart
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), B, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), C, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build and compile the OpenCL kernel program
    std::string build_option = "-DTILE_SIZE=" + std::to_string(TILE_SIZE);
    ret = clBuildProgram(program, 1, &device_id, build_option.c_str(), NULL, NULL);
    if (ret == CL_BUILD_PROGRAM_FAILURE) { // If compile failed, print the error message
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char *) malloc(log_size);

		// Get the log and print it
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
	}

    // Create the OpenCL kernel
    cl_kernel kernel;
	if(isTile)
		kernel = clCreateKernel(program, "matrix_mul_tile", &ret);
	else
		kernel = clCreateKernel(program, "matrix_mul", &ret);
	
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

    int dimention = 2; // In this example, We will use 2 dimention index
    size_t global_item_size[] = {SIZE, SIZE, 1};
    size_t local_item_size[] = {TILE_SIZE, TILE_SIZE, 1};

	cl_event perf_event;
	cl_ulong start, end;

	// Execute the OpenCL kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, dimention, NULL, global_item_size, local_item_size, 0, NULL, &perf_event);
    // Capture performance event from target device. In this case the event is to retrive the execution time.
    ret = clWaitForEvents(1, &perf_event);
    ret = clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    ret = clGetEventProfilingInfo(perf_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	std::cout << "OpenCL matrix multiplication: " << (float)(end - start)/1000000000 << " sec" << std::endl;

    // Read the memory buffer C from the device into the local variable C
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, SIZE*SIZE*sizeof(int), C, 0, NULL, NULL);

	// Make sure all the command in the command queue has been executed
    ret = clFinish(command_queue);

    bool validate = true;
    for(int j=0; j<SIZE; j++) {
		for(int i=0; i<SIZE; i++) {
			if (C[j*SIZE + i] != C_seq[j*SIZE + i])
				validate = false;
		}
	}

	if (validate == false)
		std::cout << "The results are mismatched !!" << std::endl;

    // Clean up
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

	
}