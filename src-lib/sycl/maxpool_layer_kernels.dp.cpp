/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * SYCL version of src-lib/maxpool_layer_kernels.cu, generated with SYCLomatic and adapted by the DarkStudio project.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sycl/sycl.hpp>
#include "darknet_internal.hpp"

void forward_maxpool_depth_layer_kernel(int n, int w, int h, int c, int out_c, int batch, float *input, float *output, int *indexes)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if (id >= n) return;

	int j = id % w;
	id = id / w;
	int i = id % h;
	id = id / h;
	//int g = id % out_c;
	//id = id / out_c;
	int b = id % batch;

	int k;
	for (int g = 0; g < out_c; ++g)
	{
		int out_index = j + w*(i + h*(g + out_c*b));
		float max = -FLT_MAX;
		int max_i = -1;

		for (k = g; k < c; k += out_c)
		{
			int in_index = j + w*(i + h*(k + c*b));
			float val = input[in_index];

			max_i = (val > max) ? in_index : max_i;
			max = (val > max) ? val : max;
		}
		output[out_index] = max;
		if (indexes) indexes[out_index] = max_i;
	}
}


void backward_maxpool_depth_layer_kernel(int n, int w, int h, int c, int batch, float *delta, float *prev_delta, int *indexes)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if (id >= n) return;

	int index = indexes[id];
	prev_delta[index] += delta[id];
}


void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *input, float *output, int *indexes)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int h = (in_h + pad - size) / stride_y + 1;
        int w = (in_w + pad - size) / stride_x + 1;
	int c = in_c;

        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if(id >= n) return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -pad / 2;
	int h_offset = -pad / 2;

	int out_index = j + w*(i + h*(k + c*b));
	float max = -INFINITY;
	int max_i = -1;
	int l, m;
	for(l = 0; l < size; ++l){
		for(m = 0; m < size; ++m){
			int cur_h = h_offset + i*stride_y + l;
			int cur_w = w_offset + j*stride_x + m;
			int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
			int valid = (cur_h >= 0 && cur_h < in_h &&
					cur_w >= 0 && cur_w < in_w);
			float val = (valid != 0) ? input[index] : -INFINITY;
			max_i = (val > max) ? index : max_i;
			max   = (val > max) ? val   : max;
		}
	}
	output[out_index] = max;
	if (indexes) indexes[out_index] = max_i;
}

void forward_zero_nonmax_kernel(int n, float *input, float *output)
{

        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if (id >= n) return;

	if (input[id] != output[id]) output[id] = 0;
}

void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *delta, float *prev_delta, int *indexes)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int h = (in_h + pad - size) / stride_y + 1;
        int w = (in_w + pad - size) / stride_x + 1;
	int c = in_c;
	int area_x = (size - 1) / stride_x;
	int area_y = (size - 1) / stride_y;

        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if(id >= n) return;

	int index = id;
	int j = id % in_w;
	id /= in_w;
	int i = id % in_h;
	id /= in_h;
	int k = id % in_c;
	id /= in_c;
	int b = id;

	int w_offset = -pad / 2;
	int h_offset = -pad / 2;

	float d = 0;
	int l, m;
	for(l = -area_y; l < area_y+1; ++l){
		for(m = -area_x; m < area_x+1; ++m){
			int out_w = (j-w_offset)/stride_x + m;
			int out_h = (i-h_offset)/stride_y + l;
			int out_index = out_w + w*(out_h + h*(k + c*b));
			int valid = (out_w >= 0 && out_w < w &&
					out_h >= 0 && out_h < h);
			d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
		}
	}
	prev_delta[index] += d;
}

void backward_zero_nonmax_kernel(int n, int *indexes, float *prev_delta)
{

        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if (id >= n) return;

	if (indexes[id] != id) prev_delta[id] = 0;
}

void forward_maxpool_layer_gpu(Darknet::Layer &l,
                               Darknet::NetworkState state) try {
        TAT(TATPARMS);

	if (l.maxpool_depth)
	{
		int h = l.out_h;
		int w = l.out_w;
		int c = 1;// layer.out_c;

		size_t n = h*w*c*l.batch;

                get_cuda_stream()->parallel_for(
                    sycl::nd_range<3>(cuda_gridsize(n) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=, l_w = l.w, l_h = l.h, l_c = l.c, l_out_c = l.out_c, l_batch = l.batch, state_input = state.input, l_output_gpu = l.output_gpu, l_indexes_gpu = l.indexes_gpu](sycl::nd_item<3> item_ct1) {
                            forward_maxpool_depth_layer_kernel(
                                n, l_w, l_h, l_c, l_out_c, l_batch, state_input,
                                l_output_gpu, l_indexes_gpu);
                    });
                CHECK_CUDA(cudaPeekAtLastError());

                return;
	}

#ifdef CUDNN_DISABLED
	if (!state.train && l.stride == l.size)
	{
		// cudnnPoolingBackward
		cudnnStatus_t maxpool_status;

		float alpha = 1, beta = 0;
		maxpool_status = cudnnPoolingForward(
			cudnn_handle(),
			l.poolingDesc,
			&alpha,
			l.srcTensorDesc,
			state.input,
			&beta,
			l.dstTensorDesc,
			l.output_gpu);

		//maxpool_status = cudnnDestroyPoolingDescriptor(poolingDesc);
		//cudnnDestroyTensorDescriptor(l.srcTensorDesc);
		//cudnnDestroyTensorDescriptor(l.dstTensorDesc);
	}
	else
#endif
	{
		int h = l.out_h;
		int w = l.out_w;
		int c = l.out_c;

		size_t n = h * w * c * l.batch;

                get_cuda_stream()->parallel_for(
                    sycl::nd_range<3>(cuda_gridsize(n) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=, l_h = l.h, l_w = l.w, l_c = l.c, l_stride_x = l.stride_x, l_stride_y = l.stride_y, l_size = l.size, l_pad = l.pad, state_input = state.input, l_output_gpu = l.output_gpu, l_indexes_gpu = l.indexes_gpu](sycl::nd_item<3> item_ct1) {
                            forward_maxpool_layer_kernel(
                                n, l_h, l_w, l_c, l_stride_x, l_stride_y,
                                l_size, l_pad, state_input, l_output_gpu,
                                l_indexes_gpu);
                    });
                CHECK_CUDA(cudaPeekAtLastError());

                if (l.maxpool_zero_nonmax)
		{
                        get_cuda_stream()->parallel_for(
                            sycl::nd_range<3>(cuda_gridsize(n) *
                                                  sycl::range<3>(1, 1, BLOCK),
                                              sycl::range<3>(1, 1, BLOCK)),
                            [=, state_input = state.input, l_output_gpu = l.output_gpu](sycl::nd_item<3> item_ct1) {
                                    forward_zero_nonmax_kernel(n, state_input,
                                                               l_output_gpu);
                            });
                        CHECK_CUDA(cudaPeekAtLastError());
                }
	}

	if (l.antialiasing)
	{
		Darknet::NetworkState s = { 0 };
		s.train = state.train;
		s.workspace = state.workspace;
		s.net = state.net;
		if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
		s.input = l.output_gpu;
		forward_convolutional_layer_gpu(*(l.input_layer), s);
		simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.input_antialiasing_gpu);
		simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.input_layer->output_gpu, l.output_gpu);
	}
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void backward_maxpool_layer_gpu(Darknet::Layer &l,
                                Darknet::NetworkState state) try {
        TAT(TATPARMS);

	if (l.antialiasing)
	{
		Darknet::NetworkState s = { 0 };
		s.train = state.train;
		s.workspace = state.workspace;
		s.net = state.net;
		s.delta = l.delta_gpu;  // s.delta will be returned to l.delta_gpu
		s.input = l.input_antialiasing_gpu;
		//if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
		simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.delta_gpu, l.input_layer->delta_gpu);
		backward_convolutional_layer_gpu(*(l.input_layer), s);

		//simple_copy_ongpu(l.outputs*l.batch, l.input_antialiasing_gpu, l.output_gpu);
	}

	if (l.maxpool_depth)
	{
		int h = l.out_h;
		int w = l.out_w;
		int c = l.out_c;

		size_t n = h * w * c * l.batch;

                get_cuda_stream()->parallel_for(
                    sycl::nd_range<3>(cuda_gridsize(n) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=, l_w = l.w, l_h = l.h, l_c = l.c, l_batch = l.batch, l_delta_gpu = l.delta_gpu, state_delta = state.delta, l_indexes_gpu = l.indexes_gpu](sycl::nd_item<3> item_ct1) {
                            backward_maxpool_depth_layer_kernel(
                                n, l_w, l_h, l_c, l_batch, l_delta_gpu,
                                state_delta, l_indexes_gpu);
                    });
                CHECK_CUDA(cudaPeekAtLastError());
                return;
	}

	size_t n = l.h*l.w*l.c*l.batch;

        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=, l_h = l.h, l_w = l.w, l_c = l.c, l_stride_x = l.stride_x, l_stride_y = l.stride_y, l_size = l.size, l_pad = l.pad, l_delta_gpu = l.delta_gpu, state_delta = state.delta, l_indexes_gpu = l.indexes_gpu](sycl::nd_item<3> item_ct1) {
                    backward_maxpool_layer_kernel(
                        n, l_h, l_w, l_c, l_stride_x, l_stride_y, l_size, l_pad,
                        l_delta_gpu, state_delta, l_indexes_gpu);
            });
        CHECK_CUDA(cudaPeekAtLastError());

        if (l.maxpool_zero_nonmax)
	{
                get_cuda_stream()->parallel_for(
                    sycl::nd_range<3>(cuda_gridsize(n) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=, l_indexes_gpu = l.indexes_gpu, state_delta = state.delta](sycl::nd_item<3> item_ct1) {
                            backward_zero_nonmax_kernel(n, l_indexes_gpu,
                                                        state_delta);
                    });
                CHECK_CUDA(cudaPeekAtLastError());
        }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void forward_local_avgpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *input, float *output)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int h = (in_h + pad - size) / stride_y + 1;
        int w = (in_w + pad - size) / stride_x + 1;
	int c = in_c;

        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if (id >= n) return;

	int j = id % w;
	id /= w;
	int i = id % h;
	id /= h;
	int k = id % c;
	id /= c;
	int b = id;

	int w_offset = -pad / 2;
	int h_offset = -pad / 2;

	int out_index = j + w*(i + h*(k + c*b));
	float avg = 0;
	int counter = 0;
	int l, m;
	for (l = 0; l < size; ++l) {
		for (m = 0; m < size; ++m) {
			int cur_h = h_offset + i*stride_y + l;
			int cur_w = w_offset + j*stride_x + m;
			int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
			int valid = (cur_h >= 0 && cur_h < in_h &&
				cur_w >= 0 && cur_w < in_w);
			if (valid) {
				counter++;
				avg += input[index];
			}
		}
	}
	output[out_index] = avg / counter;  // as CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
}


void backward_local_avgpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride_x, int stride_y, int size, int pad, float *delta, float *prev_delta)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int h = (in_h + pad - size) / stride_y + 1;
        int w = (in_w + pad - size) / stride_x + 1;
	int c = in_c;
	int area_x = (size - 1) / stride_x;
	int area_y = (size - 1) / stride_y;

        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if (id >= n) return;

	int index = id;
	int j = id % in_w;
	id /= in_w;
	int i = id % in_h;
	id /= in_h;
	int k = id % in_c;
	id /= in_c;
	int b = id;

	int w_offset = -pad / 2;
	int h_offset = -pad / 2;

	int counter = 0;
	float d = 0;
	int l, m;
	for (l = -area_y; l < area_y + 1; ++l) {
		for (m = -area_x; m < area_x + 1; ++m) {
			int out_w = (j - w_offset) / stride_x + m;
			int out_h = (i - h_offset) / stride_y + l;
			int out_index = out_w + w*(out_h + h*(k + c*b));
			int valid = (out_w >= 0 && out_w < w && out_h >= 0 && out_h < h);
			if (valid) {
				counter++;
				d += delta[out_index];
			}
		}
	}
	if(counter > 0) prev_delta[index] += d / counter;
}



void forward_local_avgpool_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

#ifdef CUDNN_DISABLED
	if (!state.train && l.stride == l.size)
	{
		// cudnnPoolingBackward
		cudnnStatus_t maxpool_status;

		float alpha = 1, beta = 0;
		maxpool_status = cudnnPoolingForward(
			cudnn_handle(),
			l.poolingDesc,
			&alpha,
			l.srcTensorDesc,
			state.input,
			&beta,
			l.dstTensorDesc,
			l.output_gpu);

		//maxpool_status = cudnnDestroyPoolingDescriptor(poolingDesc);
		//cudnnDestroyTensorDescriptor(l.srcTensorDesc);
		//cudnnDestroyTensorDescriptor(l.dstTensorDesc);
	}
	else
#endif
	{
		int h = l.out_h;
		int w = l.out_w;
		int c = l.out_c;

		size_t n = h*w*c*l.batch;

                get_cuda_stream()->parallel_for(
                    sycl::nd_range<3>(cuda_gridsize(n) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=, l_h = l.h, l_w = l.w, l_c = l.c, l_stride_x = l.stride_x, l_stride_y = l.stride_y, l_size = l.size, l_pad = l.pad, state_input = state.input, l_output_gpu = l.output_gpu](sycl::nd_item<3> item_ct1) {
                            forward_local_avgpool_layer_kernel(
                                n, l_h, l_w, l_c, l_stride_x, l_stride_y,
                                l_size, l_pad, state_input, l_output_gpu);
                    });
                CHECK_CUDA(cudaPeekAtLastError());
        }
}

void backward_local_avgpool_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	size_t n = l.h * l.w * l.c * l.batch;

        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=, l_h = l.h, l_w = l.w, l_c = l.c, l_stride_x = l.stride_x, l_stride_y = l.stride_y, l_size = l.size, l_pad = l.pad, l_delta_gpu = l.delta_gpu, state_delta = state.delta](sycl::nd_item<3> item_ct1) {
                    backward_local_avgpool_layer_kernel(
                        n, l_h, l_w, l_c, l_stride_x, l_stride_y, l_size, l_pad,
                        l_delta_gpu, state_delta);
            });
        CHECK_CUDA(cudaPeekAtLastError());
}
