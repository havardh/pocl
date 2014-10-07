#include "poclu.h"
#include <stdio.h>

int main(int argc, char **argv)
{

  cl_context ctx;
  cl_device_id did;
  cl_command_queue queue;
  cl_int err;
  size_t rvs;
  char result[1024];

  poclu_get_any_device( &ctx, &did, &queue );

  err = clGetDeviceInfo( did, CL_DEVICE_NAME, sizeof(result), result, &rvs);

  result[3] = 0;

  printf("%s\n", result );

  return 0;
}
