#include "poclu.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "error_codes.h"

#define N 3

static const char* kernelSource = "\n"                      \
  "__kernel void add( __global int* a, \n"                  \
  "                   __global int* b, \n"                  \
  "                   __global int* c) \n"                  \
  "{ \n"                                                    \
  "  int id = get_global_id(0); \n"                         \
  "  if (id < 3) \n"                                        \
  "    c[id] = a[id] + b[id];\n"                            \
  "} \n";

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

    cl::Context context(CL_DEVICE_TYPE_ALL, cprops);

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    assert(devices.size() == 1);

    cl::Device device = devices.at(0);

    int hostBufferA[N];
    int hostBufferB[N];
    int hostBufferC[N];

    for (int i=0; i<N; i++) {
      hostBufferA[i] = hostBufferB[i] = i+1;
      hostBufferC[i] = 0;
    }

    // Setup kernel
    cl::Program::Sources sources(1, std::make_pair(kernelSource, 0));
    cl::Program program(context, sources);

    program.build(devices);

    // Setup buffers
    cl::Buffer devBufferA(context, CL_MEM_READ_WRITE, (size_t) N*sizeof(int));
    cl::Buffer devBufferB(context, CL_MEM_READ_WRITE, (size_t) N*sizeof(int));
    cl::Buffer devBufferC(context, CL_MEM_READ_WRITE, (size_t) N*sizeof(int));

    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    // Write to GPU
    queue.enqueueWriteBuffer(devBufferA, CL_TRUE, 0, (size_t) N*sizeof(int), hostBufferA);
    queue.enqueueWriteBuffer(devBufferB, CL_TRUE, 0, (size_t) N*sizeof(int), hostBufferB);
    queue.enqueueWriteBuffer(devBufferC, CL_TRUE, 0, (size_t) N*sizeof(int), hostBufferC);

    // Set arguments for kernel
    cl::Kernel kernel(program, "add");
    kernel.setArg(0, devBufferA);
    kernel.setArg(1, devBufferB);
    kernel.setArg(2, devBufferC);

    // Execute kernel
    cl::Event enqEvent;
    queue.enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      cl::NDRange(3),
      cl::NullRange);

    // Read from GPU
    queue.enqueueReadBuffer(devBufferC, CL_TRUE, 0, (size_t) N*sizeof(int), hostBufferC);

    // Wait untill all tasks are performed
    queue.finish();

    // Print result to stdout
    for (int i=0; i<N; i++) {
      std::cout << hostBufferA[i] << " + " << hostBufferB[i] << " = " << hostBufferC[i] << "\n";
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
