#include "poclu.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#define N 1

static const char* kernelSource = "\n"          \
  "__kernel void add( __global int* a, \n"      \
  "                   __global int* b, \n"      \
  "                   __global int* c) \n"      \
  "{ \n"                                        \
  "  c[0] = a[0] + b[0];\n"                     \
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
    //assert (device.getInfo<CL_DEVICE_NAME>() == "ptx");

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
    cl::Buffer devBufferA(context, CL_MEM_READ_WRITE, (size_t) N);
    cl::Buffer devBufferB(context, CL_MEM_READ_WRITE, (size_t) N);
    cl::Buffer devBufferC(context, CL_MEM_READ_WRITE, (size_t) N);

    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    queue.enqueueWriteBuffer(devBufferA, CL_TRUE, 0, (size_t) N, hostBufferA);
    queue.enqueueWriteBuffer(devBufferB, CL_TRUE, 0, (size_t) N, hostBufferB);
    queue.enqueueWriteBuffer(devBufferC, CL_TRUE, 0, (size_t) N, hostBufferC);

    cl::Kernel kernel(program, "add");
    kernel.setArg(0, devBufferA);
    kernel.setArg(1, devBufferB);
    kernel.setArg(2, devBufferC);

    cl::Event enqEvent;
    queue.enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      cl::NDRange(1),
      cl::NullRange);

    queue.enqueueReadBuffer(devBufferC, CL_TRUE, 0, (size_t) N, hostBufferC);
    queue.finish();

    for (int i=0; i<N; i++) {
      std::cout << hostBufferA[i] << " + " << hostBufferB[i] << " = " << hostBufferC[i] << "\n";
    }
    

  } catch (cl::Error err) {
    std::cerr
      << "ERROR: "
      << err.what()
      << "("
      << err.err()
      << ")"
      << std::endl;
  }
}
