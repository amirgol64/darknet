/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * SYCL version of src-lib/avgpool_layer_kernels.cu, generated with SYCLomatic and adapted by the DarkStudio project.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sycl/sycl.hpp>
#include "darknet_internal.hpp"

void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if(id >= n) return;

	int k = id % c;
	id /= c;
	int b = id;

	int i;
	int out_index = (k + c*b);
	output[out_index] = 0;
	for(i = 0; i < w*h; ++i){
		int in_index = i + h*w*(k + b*c);
		output[out_index] += input[in_index];
	}
	output[out_index] /= w*h;
}

void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if(id >= n) return;

	int k = id % c;
	id /= c;
	int b = id;

	int i;
	int out_index = (k + c*b);
	for(i = 0; i < w*h; ++i){
		int in_index = i + h*w*(k + b*c);
		in_delta[in_index] += out_delta[out_index] / (w*h);
	}
}

void forward_avgpool_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	size_t n = l.c * l.batch;

        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=, l_w = l.w, l_h = l.h, l_c = l.c, state_input = state.input, l_output_gpu = l.output_gpu](sycl::nd_item<3> item_ct1) {
                    forward_avgpool_layer_kernel(n, l_w, l_h, l_c, state_input,
                                                 l_output_gpu);
            });
        CHECK_CUDA(cudaPeekAtLastError());
}

void backward_avgpool_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	size_t n = l.c * l.batch;

        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=, l_w = l.w, l_h = l.h, l_c = l.c, state_delta = state.delta, l_delta_gpu = l.delta_gpu](sycl::nd_item<3> item_ct1) {
                    backward_avgpool_layer_kernel(n, l_w, l_h, l_c, state_delta,
                                                  l_delta_gpu);
            });
        CHECK_CUDA(cudaPeekAtLastError());
}
