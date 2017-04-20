/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// https://github.com/parallel-forall/code-samples/blob/master/posts/unified-memory/dataElem_um.cu

#include <string.h>
#include <stdio.h>

struct DataElement
{
  char *name;
  int value;
  float* data;
};

__global__
void Kernel(DataElement *elem) {
  printf("On device: name=%s, value=%d, threadidx=%d\n", elem->name, elem->value, threadIdx.x);

  elem->name[0] = 'd';
  elem->value++;
  elem->data[3+threadIdx.x] = 42;
}

void launch(DataElement *elem) {
  Kernel<<< 1, 2 >>>(elem);
  cudaDeviceSynchronize();
}

int main(void)
{
  DataElement *e;
  cudaMallocManaged((void**)&e, sizeof(DataElement));

  e->value = 10;
  cudaMallocManaged((void**)&(e->name), sizeof(char) * (strlen("hello") + 1) );
  strcpy(e->name, "hello");
  cudaMallocManaged((void**)&(e->data), sizeof(float) * 20 );

  launch(e);

  printf("On host: name=%s, value=%d, data[3]=%f, data[4]=%f\n", e->name, e->value, e->data[3], e->data[4]);

  cudaFree(e->name);
  cudaFree(e->data);
  cudaFree(e);

  cudaDeviceReset();
}
