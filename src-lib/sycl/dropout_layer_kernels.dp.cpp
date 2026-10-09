/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * SYCL version of src-lib/dropout_layer_kernels.cu, generated with SYCLomatic and adapted by the DarkStudio project.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sycl/sycl.hpp>
#include "darknet_internal.hpp"
#include <cmath>

//void fill_ongpu(int N, float ALPHA, float * X, int INCX);
//int64_t get_current_iteration(Darknet::Network net);


void dropblock_fast_kernel(float *rand, float prob, int w, int h, int spatial, int filters, int batch, int block_size, float *drop_blocks_scale, float *output,
                           int &prob_block, int &index_block)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int threads = BLOCK;
        const int id = item_ct1.get_local_id(2);
        const int f = item_ct1.get_group(2) % filters;
        const int b = item_ct1.get_group(2) / filters;

        if (id == 0) {
		prob_block = 1.0 * 1000000;
		index_block = -1;
	}
        /*
	DPCT1065:279: Consider replacing sycl::nd_item::barrier() with
         * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
         * better performance if there is no access to global memory.
	*/
        item_ct1.barrier();

        int i;
	for (i = id; i < spatial; i += threads) {
		int index = b*spatial*f + f*spatial + i;

		if (rand[index] < prob) {
			//Chose with the lowest rand[i]
			int new_val = rand[index] * 1000000;
			rand[index] = 1;
                        int old_val = dn_sycl::atomic_fetch_min<
                            sycl::access::address_space::generic_space>(
                            &prob_block, new_val);
                        if (new_val < old_val)
			{
				index_block = i;
			}
		}

	}
        /*
	DPCT1065:280: Consider replacing sycl::nd_item::barrier() with
         * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
         * better performance if there is no access to global memory.
	*/
        item_ct1.barrier();
        if (index_block == -1) return;


	int b_x = index_block % w;
	int b_y = index_block / w;

	if (b_x > (w - block_size)) b_x = b_x - (w - block_size);
	if (b_y > (h - block_size)) b_y = b_y - (h - block_size);

        b_x = sycl::max(0, sycl::min(b_x, w - block_size));
        b_y = sycl::max(0, sycl::min(b_y, h - block_size));

        int block_square_size = block_size * block_size;

	for (i = id; i < block_square_size; i += threads)
	{
		int i_x = i % block_size;
		int i_y = i / block_size;

		int x = b_x + i_x;
		int y = b_y + i_y;

		if (x >= 0 && x < w && y >= 0 && y < h) {
			int new_index = b*filters*spatial + f*spatial + y*w + x;

			output[new_index] = 0;
			rand[new_index] = 0;
		}
	}

	if (id == 0 && drop_blocks_scale)
	{
                dn_sycl::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &drop_blocks_scale[b], block_square_size);
        }

}

void set_scales_dropblock_kernel(float *drop_blocks_scale, int block_size_w, int block_size_h, int outputs, int batch)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2);
        if (index >= batch) return;

	const float prob = drop_blocks_scale[index] / (float)outputs;
	const float scale = 1.0f / (1.0f - prob);
	drop_blocks_scale[index] = scale;
}

void scale_dropblock_kernel(float *output, int size, int outputs, float *drop_blocks_scale)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2);
        if (index >= size) return;

	const int b = index / outputs;
	output[index] *= drop_blocks_scale[b];
}


void backward_dropblock_kernel(float *pass, float *delta, int size)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2);
        if (index >= size) return;

	if (pass[index] == 0) delta[index] = 0;
}


void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int id = (item_ct1.get_group(2) +
                  item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                     item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
        if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(Darknet::Layer &l,
                               Darknet::NetworkState state) try {
        TAT(TATPARMS);

	if (!state.train) return;
	int iteration_num = get_current_iteration(state.net); // (*state.net.seen) / (state.net.batch*state.net.subdivisions);
	//if (iteration_num < state.net.burn_in) return;

	// We gradually increase the block size and the probability of dropout - during the first half of the training
	float multiplier = 1.0;
	if(iteration_num < (state.net.max_batches*0.85))
		multiplier = (iteration_num / (float)(state.net.max_batches*0.85));

	// dropblock
	if (l.dropblock)
	{
		//l.probability = 1 / keep_prob
		//const int max_blocks_per_channel = 10;
		const float cur_prob = l.probability * multiplier;
//		const float cur_scale = 1.f / (1.f - cur_prob);

		int block_width = l.dropblock_size_abs *multiplier;
		int block_height = l.dropblock_size_abs *multiplier;

		if (l.dropblock_size_rel) {
			block_width = l.dropblock_size_rel * l.w * multiplier;
			block_height = l.dropblock_size_rel * l.h * multiplier;
		}

		block_width		= std::clamp(block_width, 1, l.w);
		block_height	= std::clamp(block_height, 1, l.h);

		const int block_size = std::min(block_width, block_height);
		const float block_prob = cur_prob / (block_size*block_size);
		assert(block_size <= l.w && block_size <= l.h);

		const int size = l.inputs*l.batch;
		cuda_random(l.rand_gpu, size);

		fill_ongpu(l.batch, 0, l.drop_blocks_scale_gpu, 1);

		//fill_ongpu(l.outputs * l.batch, 1, state.input, 1); // remove!!!

		int num_blocks = l.batch * l.c;
                get_cuda_stream()->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<int, 0> prob_block_acc_ct1(cgh);
                        sycl::local_accessor<int, 0> index_block_acc_ct1(cgh);

                        auto l_w_l_h_ct4 = l.w * l.h;

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                                  sycl::range<3>(1, 1, BLOCK),
                                              sycl::range<3>(1, 1, BLOCK)),
                            [=, l_rand_gpu = l.rand_gpu, l_w = l.w, l_h = l.h, l_c = l.c, l_batch = l.batch, l_drop_blocks_scale_gpu = l.drop_blocks_scale_gpu, state_input = state.input](sycl::nd_item<3> item_ct1) {
                                    dropblock_fast_kernel(
                                        l_rand_gpu, block_prob, l_w, l_h,
                                        l_w_l_h_ct4, l_c, l_batch, block_size,
                                        l_drop_blocks_scale_gpu, state_input,
                                        prob_block_acc_ct1,
                                        index_block_acc_ct1);
                            });
                });
                CHECK_CUDA(cudaPeekAtLastError());

                num_blocks = get_number_of_blocks(l.batch, BLOCK);
                get_cuda_stream()->parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=, l_drop_blocks_scale_gpu = l.drop_blocks_scale_gpu, l_outputs = l.outputs, l_batch = l.batch](sycl::nd_item<3> item_ct1) {
                            set_scales_dropblock_kernel(l_drop_blocks_scale_gpu,
                                                        block_size, block_size,
                                                        l_outputs, l_batch);
                    });
                CHECK_CUDA(cudaPeekAtLastError());

                num_blocks = get_number_of_blocks(l.outputs * l.batch, BLOCK);
                get_cuda_stream()->submit([&](sycl::handler &cgh) {
                        auto l_outputs_l_batch_ct1 = l.outputs * l.batch;

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                                  sycl::range<3>(1, 1, BLOCK),
                                              sycl::range<3>(1, 1, BLOCK)),
                            [=, state_input = state.input, l_outputs = l.outputs, l_drop_blocks_scale_gpu = l.drop_blocks_scale_gpu](sycl::nd_item<3> item_ct1) {
                                    scale_dropblock_kernel(
                                        state_input, l_outputs_l_batch_ct1,
                                        l_outputs, l_drop_blocks_scale_gpu);
                            });
                });
                CHECK_CUDA(cudaPeekAtLastError());

        }
	// dropout
	else
	{
		int size = l.inputs*l.batch;
		cuda_random(l.rand_gpu, size);

                get_cuda_stream()->parallel_for(
                    sycl::nd_range<3>(cuda_gridsize(size) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=, state_input = state.input, l_rand_gpu = l.rand_gpu, l_probability = l.probability, l_scale = l.scale](sycl::nd_item<3> item_ct1) {
                            yoloswag420blazeit360noscope(
                                state_input, size, l_rand_gpu, l_probability,
                                l_scale);
                    });
                CHECK_CUDA(cudaPeekAtLastError());
        }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void backward_dropout_layer_gpu(Darknet::Layer &l,
                                Darknet::NetworkState state) try {
        TAT(TATPARMS);

	if(!state.delta) return;

	const int size = l.inputs*l.batch;

	// dropblock
	if (l.dropblock)
	{
		int iteration_num = get_current_iteration(state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);
		float multiplier = 1.0;
		if (iteration_num < (state.net.max_batches*0.85))
			multiplier = (iteration_num / (float)(state.net.max_batches*0.85));

		int block_width = l.dropblock_size_abs * multiplier;
		int block_height = l.dropblock_size_abs * multiplier;

		if (l.dropblock_size_rel)
		{
			block_width = l.dropblock_size_rel * l.w * multiplier;
			block_height = l.dropblock_size_rel * l.h * multiplier;
		}

		block_width		= std::clamp(block_width, 1, l.w);
		block_height	= std::clamp(block_height, 1, l.h);

		int num_blocks = get_number_of_blocks(l.outputs * l.batch, BLOCK);
                get_cuda_stream()->submit([&](sycl::handler &cgh) {
                        auto l_outputs_l_batch_ct2 = l.outputs * l.batch;

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                                  sycl::range<3>(1, 1, BLOCK),
                                              sycl::range<3>(1, 1, BLOCK)),
                            [=, l_rand_gpu = l.rand_gpu, state_delta = state.delta](sycl::nd_item<3> item_ct1) {
                                    backward_dropblock_kernel(
                                        l_rand_gpu, state_delta,
                                        l_outputs_l_batch_ct2);
                            });
                });
                CHECK_CUDA(cudaPeekAtLastError());

                get_cuda_stream()->submit([&](sycl::handler &cgh) {
                        auto l_outputs_l_batch_ct1 = l.outputs * l.batch;

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                                  sycl::range<3>(1, 1, BLOCK),
                                              sycl::range<3>(1, 1, BLOCK)),
                            [=, state_delta = state.delta, l_outputs = l.outputs, l_drop_blocks_scale_gpu = l.drop_blocks_scale_gpu](sycl::nd_item<3> item_ct1) {
                                    scale_dropblock_kernel(
                                        state_delta, l_outputs_l_batch_ct1,
                                        l_outputs, l_drop_blocks_scale_gpu);
                            });
                });
                CHECK_CUDA(cudaPeekAtLastError());
        }
	// dropout
	else
	{
                get_cuda_stream()->parallel_for(
                    sycl::nd_range<3>(cuda_gridsize(size) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=, state_delta = state.delta, l_rand_gpu = l.rand_gpu, l_probability = l.probability, l_scale = l.scale](sycl::nd_item<3> item_ct1) {
                            yoloswag420blazeit360noscope(
                                state_delta, size, l_rand_gpu, l_probability,
                                l_scale);
                    });
                CHECK_CUDA(cudaPeekAtLastError());
        }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
