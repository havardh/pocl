/* ptx.c - a nvidia ptx device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
   2011-2014 Pekka Jääskeläinen / Tampere University of Technology
   2014 Håvard Wormdal Høiby / Norwegian University of Science and Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "ptx.h"
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include <cuda.h>

//#define DEBUG

int pocl_ptx_dev_count = 0;
CUdevice* pocl_ptx_devices = NULL;

void printd(char* msg) {
#ifdef DEBUG
  printf("%s\n", msg);
#endif
}


#define checkCudaErrors(err)                         \
  if (err != CUDA_SUCCESS) {                         \
     printf("%s:%d %s\n", __FILE__,  __LINE__, err); \
     exit(1);                                        \
  }

void pocl_ptx_init_device_ops(struct pocl_device_ops* ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "ptx";

  ops->init_device_infos = pocl_ptx_init_device_infos;
  ops->init = pocl_ptx_init;
  ops->alloc_mem_obj = pocl_ptx_alloc_mem_obj;
  ops->free = pocl_ptx_free;
  ops->read = pocl_ptx_read;
  ops->write = pocl_ptx_write;
  ops->run = pocl_ptx_run;
  ops->compile_submitted_kernels = pocl_ptx_compile_submitted_kernels;

}

void
pocl_ptx_init_device_infos(struct _cl_device_id* dev)
{
  printd("pocl_ptx_init_device_infos");
  pocl_basic_init_device_infos(dev);

  dev->type = CL_DEVICE_TYPE_GPU;
  dev->llvm_target_triplet = "nvptx64-nvidia-cuda";
  dev->llvm_cpu = "sm_20";
  dev->max_mem_alloc_size = 1024;
}

void
pocl_ptx_init (cl_device_id device, const char* parameters)
{
  printd("pocl_ptx_init\n");
  
  pocl_basic_init(device, parameters);

  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGetCount(&pocl_ptx_dev_count));

  
  pocl_ptx_devices = (CUdevice*)malloc(sizeof(CUdevice)* pocl_ptx_dev_count);

  for (unsigned i=0; i<pocl_ptx_dev_count; ++i) 
    {
      checkCudaErrors(cuDeviceGet(&pocl_ptx_devices[i], i));
    }

  CUcontext context;
  checkCudaErrors(cuCtxCreate(&context, 0, pocl_ptx_devices[0]));

}

cl_int
pocl_ptx_alloc_mem_obj (cl_device_id device, cl_mem mem_obj)
{

  CUdeviceptr* deviceBuffer = malloc(sizeof(CUdeviceptr));
  checkCudaErrors(cuMemAlloc(deviceBuffer, mem_obj->size));
  mem_obj->device_ptrs[device->dev_id].mem_ptr = deviceBuffer;

  return CL_SUCCESS;
}

void
pocl_ptx_free (void *data, cl_mem_flags flags, void *ptr)
{
  //checkCudaErrors(cudaMemFree(*(CUdeviceptr*)ptr));
}

void
pocl_ptx_read (void *data, void *host_ptr, const void *device_ptr, size_t cb)
{
  printd("pocl_ptx_read");

  checkCudaErrors(cuMemcpyDtoH(host_ptr, *(CUdeviceptr*)device_ptr, cb));

}

void
pocl_ptx_write (void *data, const void *host_ptr, void *device_ptr, size_t cb)
{
  printd("pocl_ptx_write");

  checkCudaErrors(cuMemcpyHtoD(*(CUdeviceptr*)device_ptr, host_ptr, cb));
}

void
pocl_ptx_run
(void *data,
 _cl_command_node* cmd)
{
  printd("pocl_ptx_run");

  cl_kernel kernel = cmd->command.run.kernel;
  char* tmpdir = cmd->command.run.tmp_dir;
  char* objfile[POCL_FILENAME_LENGTH];

  // Read the kernel ptx source code
  int error = snprintf
    (objfile, POCL_FILENAME_LENGTH,
         "%s/parallel.ptx", tmpdir);
  assert (error >= 0);
  
  int f = fopen(objfile, "rb");
  
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);
  
  char *string = malloc(fsize + 1);
  fread(string, fsize, 1, f);
  fclose(f);
  
  string[fsize] = 0;

  // Create cuda Module from ptx string
  CUmodule cudaModule;
  checkCudaErrors(cuModuleLoadDataEx(&cudaModule, string, 0, 0, 0));

  // Extract the kernel function by name
  CUfunction cudaFunction;
  checkCudaErrors(cuModuleGetFunction(&cudaFunction, cudaModule, kernel->function_name));

  // Setup the kernel parameter array
  CUdeviceptr* kernelParams = malloc(sizeof(CUdeviceptr*) * cmd->command.run.arg_buffer_count);
  cl_device_id device = cmd->device;
  cl_mem* buffers = cmd->command.run.arg_buffers;
  for (unsigned i=0; i<cmd->command.run.arg_buffer_count; i++)
    {
      kernelParams[i] = (CUdeviceptr*)buffers[i]->device_ptrs[device->dev_id].mem_ptr;
    }

  unsigned blockSizeX = cmd->command.run.local_x;
  unsigned blockSizeY = cmd->command.run.local_y;
  unsigned blockSizeZ = cmd->command.run.local_z;
  unsigned gridSizeX = cmd->command.run.pc.num_groups[0];
  unsigned gridSizeY = cmd->command.run.pc.num_groups[1];
  unsigned gridSizeZ = cmd->command.run.pc.num_groups[2];


  
  checkCudaErrors(cuLaunchKernel(cudaFunction, gridSizeX, gridSizeY, gridSizeZ,
                                 blockSizeX, blockSizeY, blockSizeZ,
                                 0, NULL, kernelParams, NULL));
  

}

void
pocl_ptx_compile_submitted_kernels(_cl_command_node* cmd)
{
  printd("pocl_ptx_compile_submitted_kernels");
  
  if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    {
      llvm_codegen (cmd->command.run.tmp_dir,
                    cmd->command.run.kernel,
                    cmd->device);
    }
}
