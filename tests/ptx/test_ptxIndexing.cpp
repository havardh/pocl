#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "error_codes.h"

struct {
  uint x, y, z;
} uint3;

#define N 16

static const char* kernelSource = "\n"                                  \
  "__kernel void index( __global uint* out) \n"                         \
  "{ \n"                                                                \
  "  uint x_id = get_global_id(0); \n"                                  \
  "  uint y_id = get_global_id(1); \n"                                   \

  "int width = get_num_groups(0) * get_local_size(0); \n" \ 
  "int index = x_id + y_id * width; \n" \ 

  "int result = get_group_id(1) * get_num_groups(0) + get_group_id(0); \n" \ 

  "out[index] = result; \n" \ 

  "}";
  
int main(int argc, char **argv) 
{

  try {
    // Setup device
    std::vector<cl::Platform> platformList;

    cl::Platform::get(&platformList);

    cl_context_properties cprops[] = {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)(platformList[0])(),
      0
    };

    cl::Context context(CL_DEVICE_TYPE_DEFAULT, cprops);

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    assert(devices.size() == 1);

    cl::Device device = devices.at(0);

    uint hostBuffer[N];

    for (int i=0; i<N; i++) {
      hostBuffer[i] = 0;
    }

    // Setup kernel
    cl::Program::Sources sources(1, std::make_pair(kernelSource, 0));
    cl::Program program(context, sources);

    program.build(devices);

    // Setup buffers
    cl::Buffer devBuffer(context, CL_MEM_READ_WRITE, (size_t) N*sizeof(uint));

    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    // Write to GPU
    queue.enqueueWriteBuffer(devBuffer, CL_TRUE, 0, (size_t) N*sizeof(uint), hostBuffer);

    // Set arguments for kernel
    cl::Kernel kernel(program, "index");
    kernel.setArg(0, devBuffer);

    // Execute kernel
    cl::Event enqEvent;
    queue.enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      cl::NDRange(4, 4),
      cl::NDRange(2, 2));

    // Read from GPU
    queue.enqueueReadBuffer(devBuffer, CL_TRUE, 0, (size_t) N*sizeof(uint), hostBuffer);
    // Wait untill all tasks are performed
    queue.finish();

    // Print result to stdout
    int M = 4;
    for (int i=0; i<M; i++) {
      for (int j=0; j<M; j++) {
        printf("%d", hostBuffer[i*M + j]);
      }
      printf("\n");
    }

  } catch (cl::Error err) {
    std::cerr
      << "ERROR: "
      << err.what()
      << "("
      << ptx_cl_error(err.err())
      << ", " << err.err()
      << ")"
      << std::endl;
  }
}
