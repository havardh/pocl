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
#include <cuda.h>

//#define DEBUG

void printd(char* msg) {
#ifdef DEBUG
  printf("%s\n", msg);
#endif
}

void pocl_ptx_init_device_ops(struct pocl_device_ops* ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "ptx";

  ops->init_device_infos = pocl_ptx_init_device_infos;
  ops->init = pocl_ptx_init;
  ops->read = pocl_ptx_read;
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
}

void
pocl_ptx_init (cl_device_id device, const char* parameters)
{
  printd("pocl_ptx_init\n");
  
  pocl_basic_init(device, parameters);
}

void
pocl_ptx_run
(void *data,
 _cl_command_node* cmd)
{
  printd("pocl_ptx_run");


}

void
pocl_ptx_read (void *data, void *host_ptr, const void *device_ptr, size_t cb)
{
  printd("pocl_ptx_read");
  pocl_basic_read(data, host_ptr, device_ptr, cb);
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
