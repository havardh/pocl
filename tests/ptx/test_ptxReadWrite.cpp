#include "poclu.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>



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
    cl::Device device = devices.at(0);

    int input[] = { 1, 2, 3 };
    int output[] = { 0, 0, 0 };

    cl::Buffer deviceBuffer(context, CL_MEM_READ_WRITE, sizeof(int)* 3);

    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    queue.enqueueWriteBuffer(deviceBuffer, CL_TRUE, 0, sizeof(int) * 3, input);
    queue.enqueueReadBuffer(deviceBuffer, CL_TRUE, 0, sizeof(int) * 3, output);
    queue.finish();

    for (int i=0; i<3; i++) {
      std::cout << output[i];
    }

    std::cout << std::endl;
    

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
