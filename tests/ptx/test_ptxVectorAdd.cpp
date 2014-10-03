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
    cl::Buffer devBufferA = cl::Buffer(
      context,
      (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR),
      (size_t) N, (void *) &hostBufferA[0]);

    cl::Buffer devBufferB = cl::Buffer(
      context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      (size_t) N, (void *) &hostBufferB[0]);

    cl::Buffer devBufferC = cl::Buffer(
      context,
      CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
      (size_t) N, (void *) &hostBufferC[0]);

    cl::Kernel kernel(program, "add");

    kernel.setArg(0, devBufferA);
    kernel.setArg(1, devBufferB);
    kernel.setArg(2, devBufferC);

    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    cl::Event enqEvent;
    queue.enqueueNDRangeKernel(
      kernel,
      cl::NullRange,
      cl::NDRange(1),
      cl::NullRange,
      NULL, &enqEvent);
    

    cl::Event mapEvent;
    queue.enqueueMapBuffer(
      devBufferC,
      CL_TRUE,
      CL_MAP_READ,
      0, 1, NULL, &mapEvent);

    for (int i=0; i<N; i++) {
      std::cout << hostBufferA[i] << " + " << hostBufferB[i] << " = " << hostBufferC[i] << "\n";
    }
    
    cl::Event unmapEvent;
    queue.enqueueUnmapMemObject(
      devBufferC,
      (void *) &hostBufferC[0],
      NULL, 
      &unmapEvent);

    queue.finish();
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
