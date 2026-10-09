/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * SYCL version of src-lib/im2col_kernels.cu, generated with SYCLomatic and adapted by the DarkStudio project.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cassert>

#include "darknet_gpu.hpp"
#include "im2col.hpp"
#include "dark_cuda.hpp"
#include "Timing.hpp"


template<typename T1, typename T2>
inline T1 __shfl_custom(T1 val, T2 lane) {
#if DN_SYCL_COMPAT_RT_VERSION >= 9000
        /*
	DPCT1121:130: Make sure that the "val" which is used in the SYCL
         * group function/algorithm is initialized.
	*/
        return dn_sycl::select_from_sub_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(), val, lane);
#else
	return __shfl(val, lane);
#endif
}

template<typename T>
inline uint32_t __ballot_custom(T val) {
#if DN_SYCL_COMPAT_RT_VERSION >= 9000
        return sycl::reduce_over_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (FULL_MASK &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                    val
                ? (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                              .get_local_linear_id())
                : 0,
            sycl::ext::oneapi::plus<>());
#else
	return __ballot(val);
#endif
}


// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE

void im2col_gpu_kernel(const int n, const float* data_im,
		const int height, const int width, const int ksize,
		const int pad,
		const int stride,
		const int height_col, const int width_col,
		float *data_col)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        for (int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2);
             index < n;
             index += item_ct1.get_local_range(2) * item_ct1.get_group_range(2))
        {
		int w_out = index % width_col;
		int h_index = index / width_col;
		int h_out = h_index % height_col;
		int channel_in = h_index / height_col;
		int channel_out = channel_in * ksize * ksize;
		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;
		float* data_col_ptr = data_col;
		data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
		const float* data_im_ptr = data_im;
		data_im_ptr += (channel_in * height + h_in) * width + w_in;
		for (int i = 0; i < ksize; ++i)
		{
			for (int j = 0; j < ksize; ++j)
			{
				int h = h_in + i;
				int w = w_in + j;

				*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;

				//data_im[(channel_in * height + h_in) * width + w_in + i * width + j];
				//(*data_col_ptr) = data_im_ptr[ii * width + jj];

				data_col_ptr += height_col * width_col;
			}
		}
	}
}

void im2col_ongpu(float *im,
		int channels, int height, int width,
		int ksize, int stride, int pad, float *data_col)
{
	TAT(TATPARMS);

	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int num_kernels = channels * height_col * width_col;
        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (num_kernels + BLOCK - 1) / BLOCK) *
                    sycl::range<3>(1, 1, BLOCK),
                sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                    im2col_gpu_kernel(num_kernels, im, height, width, ksize,
                                      pad, stride, height_col, width_col,
                                      data_col);
            });

        CHECK_CUDA(cudaPeekAtLastError());
}


// float 32
void im2col_align_gpu_kernel(const int n, const float* data_im,
	const int height, const int width, const int ksize,
	const int pad,
	const int stride,
	const int height_col, const int width_col,
	float *data_col, const int bit_align)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        for (int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2);
             index < n;
             index += item_ct1.get_local_range(2) * item_ct1.get_group_range(2))
        {
		int w_out = index % width_col;
		int h_index = index / width_col;
		int h_out = h_index % height_col;
		int channel_in = h_index / height_col;
		int channel_out = channel_in * ksize * ksize;
		int h_in = h_out * stride - pad;
		int w_in = w_out * stride - pad;
		//float* data_col_ptr = data_col;
		//float* data_col_ptr_32 = data_col + (channel_out * bit_align + h_out * width_col + w_out) / 32;
		//data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
		//data_col_ptr += channel_out * bit_align + h_out * width_col + w_out;
		float* data_col_ptr = &data_col[channel_out * bit_align + h_out * width_col + w_out];
		const float* data_im_ptr = data_im;
		data_im_ptr += (channel_in * height + h_in) * width + w_in;
		for (int i = 0; i < ksize; ++i)
		{
			for (int j = 0; j < ksize; ++j)
			{
				int h = h_in + i;
				int w = w_in + j;

				float val = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0.0f;

				int pre_out_index = index % (width_col*height_col);
				int out_index = (channel_out + i*ksize + j) * bit_align + pre_out_index;// h_out * width_col + w_out;
				data_col[out_index] = val;

				/// @todo V3 should this be "*data_col_ptr" instead?  Otherwise, what exactly are we incrementing here?  (answer: the pointer...but why?)
				data_col_ptr += bit_align;
			}
		}
	}
}

void im2col_align_ongpu(float *im,
	int channels, int height, int width,
	int ksize, int stride, int pad, float *data_col, int bit_align)
{
	TAT(TATPARMS);

	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int num_kernels = channels * height_col * width_col;
        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, (num_kernels + BLOCK - 1) / BLOCK) *
                    sycl::range<3>(1, 1, BLOCK),
                sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                    im2col_align_gpu_kernel(num_kernels, im, height, width,
                                            ksize, pad, stride, height_col,
                                            width_col, data_col, bit_align);
            });

        CHECK_CUDA(cudaPeekAtLastError());
}


// binary im2col - stride=1
/*
DPCT1110:133: The total declared local variable size in device function
 * im2col_align_bin_gpu_kernel exceeds 128 bytes and may cause high register
 * pressure. Consult with your hardware vendor to find the total register size
 * available and adjust the code, or use smaller sub-group size to avoid high
 * register pressure.
*/
void im2col_align_bin_gpu_kernel(const int n, const float *data_im,
                                 const int height, const int width,
                                 const int ksize, const int channels,
                                 const int pad, const int stride,
                                 const int height_col, const int width_col,
                                 float *data_col, const int bit_align)
{
	//__shared__ float tmp_s[1];
	//__shared__ ulonglong4 tmp256_s[1];


	//#define SHRED_VALS ((BLOCK / 169) * )
	//__shared__ float dst_s[1024];
	//__shared__ float dst_s[1024];
	//__shared__ uint32_t bit_s[32];
	//__shared__ uint8_t bit_s[128];

        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
        //for (; index < n; index += blockDim.x*gridDim.x)
	{
		int c_index = index;
		int channel_in = c_index % channels;

		//int h_out = index % height_col;
		//int c_index = index / height_col;
		//int channel_in = c_index % channels;

		int channel_out = channel_in * ksize * ksize;

		int j_index = c_index / channels;
		int j = j_index % ksize;
		int i = j_index / ksize;

		int pre_out_index = (channel_out + i*ksize + j) * bit_align;
		int j_pad = (j - pad);
		int i_pad = (i - pad);

		for(int wh_index = 0; wh_index < (height_col*width_col); wh_index += 32)
		//for (int h_out = 0; h_out < height_col; ++h_out)
		{

			// the end of padding
			//if(0)
			//for (int w_out = 0; w_out < (width_col); w_out += 32)
			{
				const int w_out = wh_index % width_col;
				const int h_out = wh_index / width_col;

				const int w = w_out + j_pad;
				const int h = h_out + i_pad;

				int pre_in_index = channel_in * height * width;
				int pre_in_wh_index = h * width + w;

				int send_wh_index = wh_index;
				if (i >= ksize) send_wh_index = height_col*width_col;

				#pragma unroll
				for (int t = 0; t < WARP_SIZE; ++t)
				{
                                        const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

                                        const int cur_wh_index = __shfl_custom(send_wh_index, t) + lane_id;

					if (cur_wh_index < (width_col*height_col))// && (cur_i_pad+pad) < ksize)
					{
						const int cur_pre_out_index = __shfl_custom(pre_out_index, t);

						const int cur_pre_in_index = __shfl_custom(pre_in_index, t);
						const int cur_pre_in_wh_index = __shfl_custom(pre_in_wh_index, t) + lane_id;

						int w = cur_pre_in_wh_index % width;
						int h = cur_pre_in_wh_index / width;
						int in_index = cur_pre_in_index + cur_pre_in_wh_index;

						int out_index = cur_pre_out_index + cur_wh_index;

						float val = (w >= 0 && w < width && h >= 0 && h < height) ?
							data_im[in_index] : float();

						//data_col[out_index] = val;
						//tmp_s[0] = val;

						uint32_t bit_mask = __ballot_custom(val > 0);
						if (lane_id == 0) {
							uint8_t *bit8_ptr = &(((uint8_t *)data_col)[out_index / 8]);
							uint32_t *bit32_ptr = (uint32_t *)bit8_ptr;
							*bit32_ptr = bit_mask;
						}
					}


				}

			}// w_out

		}
	}
}


void im2col_align_bin_ongpu(float *im,
	int channels, int height, int width,
	int ksize, int stride, int pad, float *data_col, int bit_align)
{
	TAT(TATPARMS);

	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	//int num_kernels = channels * height_col * width_col * ksize * ksize;
	//int num_kernels = channels * ksize * ksize * height_col;
	int num_kernels = channels * ksize * ksize;
	int num_blocks = num_kernels / BLOCK + 1;

	//im2col_align_bin_gpu_kernel <<<(num_kernels + BLOCK - 1) / BLOCK,
        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                    im2col_align_bin_gpu_kernel(
                        num_kernels, im, height, width, ksize, channels, pad,
                        stride, height_col, width_col, data_col, bit_align);
            });

        CHECK_CUDA(cudaPeekAtLastError());
}


void float_to_bit_gpu_kernel(float *src, unsigned char *dst, size_t size,
                             uint32_t *tmp)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

        int index = 32 * item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
        float src_val;
	uint32_t *dst32_ptr = ((unsigned int*)dst);

	for (int i = 0; i < 32; ++i)
	{
		if ((index + i * 1024) < size) src_val = src[index + i*1024];
		else src_val = 0;
		//unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
		//const int num_of_warps = blockDim.x / WARP_SIZE;
                const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
                const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

                uint32_t bit_mask = __ballot_custom(src_val > 0);
		if (lane_id == 0) tmp[i * 32 + warp_id] = bit_mask;
	}
        item_ct1.barrier(sycl::access::fence_space::local_space);
        dst32_ptr[item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2)] = tmp[item_ct1.get_local_id(2)];
}


void float_to_bit_gpu(float *src, unsigned char *dst, size_t size)
{
	TAT(TATPARMS);

	const int num_blocks = get_number_of_blocks(size, 32 * 1024);
        get_cuda_stream()->submit([&](sycl::handler &cgh) {
                /*
	    DPCT1101:320: 'WARP_SIZE*32' expression was replaced
                 * with a value. Modify the code to use the original expression,
                 * provided in comments, if it is correct.
	    */
                sycl::local_accessor<uint32_t, 1> tmp_acc_ct1(
                    sycl::range<1>(1024 /*WARP_SIZE*32*/), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                          sycl::range<3>(1, 1, 1024),
                                      sycl::range<3>(1, 1, 1024)),
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                                float_to_bit_gpu_kernel(
                                    src, dst, size,
                                    tmp_acc_ct1
                                        .get_multi_ptr<
                                            sycl::access::decorated::no>()
                                        .get());
                        });
        });
        CHECK_CUDA(cudaPeekAtLastError());
}


static inline unsigned char get_bit(unsigned char const*const src, size_t index)
{
	size_t src_i = index / 8;
	int src_shift = index % 8;
	unsigned char val = (src[src_i] & (1 << src_shift)) > 0;
	//unsigned char val = (src[src_i] & (1 << (8 - src_shift))) > 0;
	return val;
}

// Intel CPUs and nVidia CUDA GPU are little endian
unsigned char reverse_byte(unsigned char a)
{
	return ((a & 0x1) << 7) | ((a & 0x2) << 5) |
		((a & 0x4) << 3) | ((a & 0x8) << 1) |
		((a & 0x10) >> 1) | ((a & 0x20) >> 3) |
		((a & 0x40) >> 5) | ((a & 0x80) >> 7);
}

unsigned char reverse_byte_2(unsigned char a)
{
	return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

unsigned char reverse_byte_CUDA(unsigned char a)
{
        uint32_t tmp = dn_sycl::reverse_bits<unsigned int>(a);
        return tmp >> 24;
}

void transpose8rS32_reversed_diagonale(unsigned char* A, unsigned char* B, int m, int n)
{
	unsigned x, y, t;

	// Load the array and pack it into x and y.
	x = (A[0] << 24) | (A[m] << 16) | (A[2 * m] << 8) | A[3 * m];
	y = (A[4 * m] << 24) | (A[5 * m] << 16) | (A[6 * m] << 8) | A[7 * m];

	t = (x ^ (x >> 7)) & 0x00AA00AA;  x = x ^ t ^ (t << 7);
	t = (y ^ (y >> 7)) & 0x00AA00AA;  y = y ^ t ^ (t << 7);

	t = (x ^ (x >> 14)) & 0x0000CCCC;  x = x ^ t ^ (t << 14);
	t = (y ^ (y >> 14)) & 0x0000CCCC;  y = y ^ t ^ (t << 14);

	t = (x & 0xF0F0F0F0) | ((y >> 4) & 0x0F0F0F0F);
	y = ((x << 4) & 0xF0F0F0F0) | (y & 0x0F0F0F0F);
	x = t;

	B[7 * n] = reverse_byte_CUDA(x >> 24);  B[6 * n] = reverse_byte_CUDA(x >> 16);  B[5 * n] = reverse_byte_CUDA(x >> 8);  B[4 * n] = reverse_byte_CUDA(x);
	B[3 * n] = reverse_byte_CUDA(y >> 24);  B[2 * n] = reverse_byte_CUDA(y >> 16);  B[1 * n] = reverse_byte_CUDA(y >> 8);  B[0 * n] = reverse_byte_CUDA(y);

	//__device__ ​ unsigned int 	__brev(unsigned int  x)
	//Reverse the bit order of a 32 bit unsigned integer.
	// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html
}


// transpose 8x8 bit
void transpose_bin_gpu_kernel(unsigned char *A, unsigned char *B, const int n, const int m,
	const int lda, const int ldb, const int block_size)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int i;
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        //for (i = 0; i < n; i += 8)
	{
		i = (index*8) % n;
		int j;
		//for (j = 0; j < m - 8; j += 8)
		{
			j = ((index * 8) / n) * 8;
			if (j < m) {
				int a_index = i*lda + j;
				int b_index = j*ldb + i;
				transpose8rS32_reversed_diagonale(&A[a_index / 8], &B[b_index / 8], lda / 8, ldb / 8);
			}
			//else if (j < m) {
			//    for (; j < m; ++j) {
			//        if (get_bit(A, i*lda + j)) set_bit(B, j*ldb + i);
			//        else remove_bit(B, j*ldb + i);
			//    }
			//}
		}
	}
}



uint8_t reverse_8_bit(uint8_t a) {
	return ((a * 0x0802LU & 0x22110LU) | (a * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16;
}

uint32_t reverse_32_bit(uint32_t a)
{
	// __device__ ​ unsigned int __brev(unsigned int  x) // CUDA
	// unsigned int __rbit(unsigned int val) // for ARM    //__asm__("rbit %0, %1\n" : "=r"(output) : "r"(input));
        return dn_sycl::reverse_bits<unsigned int>(a);
        //return (reverse_8_bit(a >> 24) << 0) |
	//    (reverse_8_bit(a >> 16) << 8) |
	//    (reverse_8_bit(a >> 8) << 16) |
	//    (reverse_8_bit(a >> 0) << 24);
}

#define swap(a0, a1, j, m) t = (a0 ^ (a1 >>j)) & m; a0 = a0 ^ t; a1 = a1 ^ (t << j);

void transpose32_optimized(uint32_t A[32])
{
	int j, k;
	unsigned m, t;

	//m = 0x0000FFFF;
	//for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
	//    for (k = 0; k < 32; k = (k + j + 1) & ~j) {
	//        t = (A[k] ^ (A[k + j] >> j)) & m;
	//        A[k] = A[k] ^ t;
	//        A[k + j] = A[k + j] ^ (t << j);
	//    }
	//}

	j = 16;
	m = 0x0000FFFF;
	for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

	j = 8;
	m = 0x00ff00ff;
	for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

	j = 4;
	m = 0x0f0f0f0f;
	for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

	j = 2;
	m = 0x33333333;
	for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

	j = 1;
	m = 0x55555555;
	for (k = 0; k < 32; k = (k + j + 1) & ~j) { swap(A[k], A[k + j], j, m); }

	// reverse Y
	for (j = 0; j < 16; ++j) {
		uint32_t tmp = A[j];
		A[j] = reverse_32_bit(A[31 - j]);
		A[31 - j] = reverse_32_bit(tmp);
	}
}

void transpose_32x32_bits_reversed_diagonale(uint32_t *A, uint32_t *B, int m, int n,
                                             uint32_t *A_shared)
{
	//unsigned A_tmp[32];
	//int i;
	//#pragma unroll
	//for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
	//transpose32_optimized(A_tmp);
	//#pragma unroll
	//for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];

        uint32_t *A_tmp =
            &A_shared[32 * sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                               .get_local_id(2)];

        int i;
	#pragma unroll 32
	for (i = 0; i < 32; ++i) A_tmp[i] = A[i * m];
	transpose32_optimized(A_tmp);
	#pragma unroll 32
	for (i = 0; i < 32; ++i) B[i*n] = A_tmp[i];
}

// transpose 32x32 bit
void transpose_bin_gpu_kernel_32(uint32_t *A, uint32_t *B, const int n, const int m,
	const int lda, const int ldb, const int block_size, uint32_t *A_shared)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int i;
        int index = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_id(2)) *
                    32;

        //for (i = 0; i < n; i += 8)
	{
		i = index % n;
		int j;
		//for (j = 0; j < m - 8; j += 8)
		{
			j = (index / n) * 32;
			if (j < m) {
				int a_index = i*lda + j;
				int b_index = j*ldb + i;
                                transpose_32x32_bits_reversed_diagonale(
                                    &A[a_index / 32], &B[b_index / 32],
                                    lda / 32, ldb / 32, A_shared);
                        }
		}
	}
}

void transpose_bin_gpu(unsigned char *A, unsigned char *B, const int n, const int m,
	const int lda, const int ldb, const int block_size)
{
	//int size = n*m/ (8*8) + 1;
	int size32 = n*m / (32*32) + 1;
	//const int num_blocks = size / BLOCK + 1;
	const int num_blocks32 = size32 / BLOCK_TRANSPOSE32 + 1;
        get_cuda_stream()->submit([&](sycl::handler &cgh) {
                /*
	    DPCT1101:321: '32 * BLOCK_TRANSPOSE32' expression
                 * was replaced with a value. Modify the code to use the
                 * original expression, provided in comments, if it is correct.

                 */
                sycl::local_accessor<uint32_t, 1> A_shared_acc_ct1(
                    sycl::range<1>(8192 /*32 * BLOCK_TRANSPOSE32*/), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, num_blocks32) *
                            sycl::range<3>(1, 1, BLOCK_TRANSPOSE32),
                        sycl::range<3>(1, 1, BLOCK_TRANSPOSE32)),
                    [=](sycl::nd_item<3> item_ct1) {
                            transpose_bin_gpu_kernel_32(
                                (uint32_t *)A, (uint32_t *)B, n, m, lda, ldb,
                                block_size,
                                A_shared_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get());
                    });
        });
        //transpose_bin_gpu_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>>(A, B, n, m, lda, ldb, block_size);
        CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------

void transpose_uint32_kernel(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
	//l.bit_align - algined (n) by 32
	//new_ldb - aligned (k) by 256
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        //for (i = 0; i < src_h; i += 1)
	int i = index % src_h;  // l.size*l.size*l.c;
	{
		//for (j = 0; j < src_w; j += 1)
		int j = index / src_h;  // out_h*out_w;
		if(j < src_w)
		{
			((uint32_t *)dst)[j*dst_align / 32 + i] = ((uint32_t *)src)[i*src_align + j];
		}
	}
}

void transpose_uint32_gpu(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
	TAT(TATPARMS);

	int size = src_w * src_h;
	const int num_blocks = size / BLOCK + 1;
        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                    transpose_uint32_kernel(src, dst, src_h, src_w, src_align,
                                            dst_align);
            });
        CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------

//#define TRANS_LOOP 10

void transpose_uint32_kernel_2(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align,
                               uint32_t *tmp)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
   // misaligned_array[32x32]
        const int w_align = 33;
	//const int shared_size = w_align * 32;

	//l.bit_align - algined (n) by 32
	//new_ldb - aligned (k) by 256

	const int src_w_align = src_w + (32 - src_w % 32);
	//const int src_h_align = src_h + (32 - src_h % 32);

	const int warps_in_width = src_w_align / 32;
	//const int warps_in_height = src_h_align / 32;

        const int local_x = item_ct1.get_local_id(2) % 32;       // index % 32;
        const int local_x_index = item_ct1.get_local_id(2) / 32; // index / 32;
        const int local_y = local_x_index % 32;

//#pragma unroll TRANS_LOOP
	//for (int i = 0; i < TRANS_LOOP; ++i)
	{
                const int global_index = item_ct1.get_group(
                    2); // blockIdx.x*TRANS_LOOP + i;// local_x_index / 32;
                const int global_x_index = global_index % warps_in_width;
		const int global_y_index = global_index / warps_in_width;

		const int global_x = global_x_index * 32 + local_x;
		const int global_y = global_y_index * 32 + local_y;

		uint32_t val = 0;
		if (global_x < src_w && global_y < src_h) {
			val = src[global_y * src_align + global_x];
		}
		//dst[global_x * dst_align / 32 + global_y] = val;
		//tmp[local_y * 32 + local_x] = val;

		tmp[local_x * w_align + local_y] = val;
                item_ct1.barrier(sycl::access::fence_space::local_space);
                val = tmp[local_y * w_align + local_x];

		const int new_global_x = global_y_index * 32 + local_x;
		const int new_global_y = global_x_index * 32 + local_y;

		if (new_global_x < src_h && new_global_y < src_w) {
			dst[new_global_y * (dst_align / 32) + new_global_x] = val;
		}
	}
}


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
void repack_input_kernel(float *input, float *re_packed_input, int w, int h, int c)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        const int items_per_channel = w * h;

	int c_pack = index % 32;
	int chan_index = index / 32;
	int chan = (chan_index * 32) % c;
	int i = (chan_index * 32) / c;

	//for (chan = 0; chan < c; chan += 32)
	{
		//for (i = 0; i < items_per_channel; ++i)
		if(i < items_per_channel)
		{
			//for (c_pack = 0; c_pack < 32; ++c_pack)
			{
				float src = input[(chan + c_pack)*items_per_channel + i];

				re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
			}
		}
	}
}

void repack_input_gpu(float *input, float *re_packed_input, int w, int h, int c)
{
	TAT(TATPARMS);

	int size = w * h * c;
	const int num_blocks = size / BLOCK + 1;
        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                    repack_input_kernel(input, re_packed_input, w, h, c);
            });
        CHECK_CUDA(cudaPeekAtLastError());
}
// --------------------------------


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
void repack_input_kernel_2(float *input, float *re_packed_input, int w, int h, int c)
{
	//__shared__ uint32_t tmp[33 * 32];  // 33x32 is misaligned 32 x 32 to avoid bank conflicts

        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        const int items_per_channel = w * h;

	int c_pack = index % 32;
	int chan_index = index / 32;
	int chan = (chan_index * 32) % c;
	int i = (chan_index * 32) / c;

	//for (chan = 0; chan < c; chan += 32)
	{
		//for (i = 0; i < items_per_channel; ++i)
		if (i < items_per_channel)
		{
			//for (c_pack = 0; c_pack < 32; ++c_pack)
			{
				float src = input[(chan + c_pack)*items_per_channel + i];

				re_packed_input[chan*items_per_channel + i * 32 + c_pack] = src;
			}
		}
	}
}


// 32 channels -> 1 channel (with 32 floats)
// 256 channels -> 8 channels (with 32 floats)
void repack_input_kernel_bin(float *input, uint32_t *re_packed_input_bin, int w, int h, int c)
{
	//__shared__ uint32_t tmp[32];
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        const int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                          item_ct1.get_local_id(2);

        const int global_warp_id = index / WARP_SIZE;
        const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

        const int items_per_channel = w * h;
	const int items_per_channel_aligned = items_per_channel + WARP_SIZE - (items_per_channel % WARP_SIZE);

	int i = 32 * (global_warp_id % (items_per_channel_aligned / WARP_SIZE));
	int chan = 32 * (global_warp_id / (items_per_channel_aligned / WARP_SIZE));

	if (chan < c)
	{
		uint32_t result_bits = 0;

		for (int c_pack = 0; c_pack < 32; ++c_pack)
		{
			float src = 0;
			if ((i + lane_id) < items_per_channel) {
				src = input[(chan + c_pack)*items_per_channel + (i + lane_id)];
			}
			uint32_t bit_mask = __ballot_custom(src > 0);

			uint32_t cur_bit = (bit_mask >> lane_id) & uint32_t(1);

			result_bits |= (cur_bit << c_pack);
		}
		if ((i + lane_id) < items_per_channel) {
			re_packed_input_bin[chan*items_per_channel / 32 + (i + lane_id)] = result_bits;
		}
	}
}

void repack_input_gpu_bin(float *input, uint32_t *re_packed_input_bin, int w, int h, int c)
{
	TAT(TATPARMS);

	int size = (w * h * c) / 32 + 1;
	const int block_size = BLOCK;
	const int num_blocks = get_number_of_blocks(size, block_size);
        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, block_size),
                              sycl::range<3>(1, 1, block_size)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                    repack_input_kernel_bin(input, re_packed_input_bin, w, h,
                                            c);
            });
        CHECK_CUDA(cudaPeekAtLastError());
}


void fill_int8_gpu_kernel(unsigned char *src, unsigned char val, size_t size)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
        if(index < size) src[index] = 0;
}

void fill_int8_gpu(unsigned char *src, unsigned char val, size_t size)
{
	TAT(TATPARMS);

	const int num_blocks = size / BLOCK + 1;
        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                    fill_int8_gpu_kernel(src, val, size);
            });
        CHECK_CUDA(cudaPeekAtLastError());
}


static inline uint8_t xnor_bit1(uint8_t a, uint8_t b)
{
	return ~(a^b) & 0b1;
}

static inline uint32_t xor_int32(uint32_t a, uint32_t b)
{
	return (a^b);
}

static inline uint64_t xor_int64(uint64_t a, uint64_t b)
{
	return (a^b);
}

static inline sycl::ulong4 xor_int256(sycl::ulong4 a, sycl::ulong4 b)
{
        sycl::ulong4 res;
        res.w() = (a.w() ^ b.w());
        res.x() = (a.x() ^ b.x());
        res.y() = (a.y() ^ b.y());
        res.z() = (a.z() ^ b.z());
        return res;
}


__inline__ int warpAllReduceSum(int val)
{
	for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
#if DN_SYCL_COMPAT_RT_VERSION >= 9000
                val += dn_sycl::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), val,
                    mask);
#else
		val += __shfl_xor(val, mask);
#endif

	return val;
}

// Tensor Cores binary (CC >= 7.3 && CUDA >= 10.0) - __CUDA_SUBBYTE_IMMA__
#if DN_SYCL_COMPAT_RT_VERSION >= 10000

#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 128
#define WMMA_K32 (WMMA_K/32)

#define WMMA_Nx2 (WMMA_N*2)

// Tensor Cores are used for XOR-GEMM
/*
DPCT1110:140: The total declared local variable size in device function
 * gemm_nn_custom_bin_mean_transposed_tensor_kernel exceeds 128 bytes and may
 * cause high register pressure. Consult with your hardware vendor to find the
 * total register size available and adjust the code, or use smaller sub-group
 * size to avoid high register pressure.
*/
void gemm_nn_custom_bin_mean_transposed_tensor_kernel(
    int M, int N, int K, unsigned char *A, int lda, unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias_arr, int leaky_activation,
    float *shortcut_in_gpu, float *shortcut_out_gpu, int *C_s)
{
	// total 57%
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

            // 2 * 8 KB - Temprorary result of GEMM WMMA for 32 warps

        const int lane_id = item_ct1.get_local_id(2) % 32;
        const int warp_id = item_ct1.get_local_id(2) / 32;
        const int global_warp_id = index / 32;

	const int N_aligned = N + WMMA_Nx2 - (N % WMMA_Nx2);

	int i, j, k;//, h;
	// 47% = 29 + 10 + 8
	j = global_warp_id % (N_aligned / WMMA_Nx2);
	j = j * WMMA_Nx2;
	{    // out_h*out_w - one channel output size [169 - 173056]
		i = global_warp_id / (N_aligned / WMMA_Nx2);
		i = i * WMMA_M;

		//int count = 0;
		k = 0;

		if (i < M)  //if (i < M)  // l.n - filters [16 - 55 - 1024]
		{
			if (j + WMMA_Nx2 > N) j = N - WMMA_Nx2;   // must be: j+7 < N
			if (i + WMMA_M > M) i = M - WMMA_M;   // must be: i+7 < M

#if DN_SYCL_CUDA_ARCH >= 730
                        // Tensor Cores

                        /*
			DPCT1082:143: Migration of
                         * nvcuda::wmma::fragment<wmma::matrix_a, 8, 8, 128,
                         * wmma::experimental::precision::b1, wmma::row_major>
                         * type is not supported.
			*/
                        /*
			DPCT1082:144: Migration of
                         * nvcuda::wmma::matrix_a type is not supported.

                         */
                        /*
			DPCT1082:145: Migration of
                         * nvcuda::wmma::row_major type is not supported.

                         */
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                       wmma::experimental::precision::b1,
                                       wmma::row_major>
                            a_frag;
                        /*
			DPCT1082:146: Migration of
                         * nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 128,
                         * wmma::experimental::precision::b1, wmma::col_major>
                         * type is not supported.
			*/
                        /*
			DPCT1082:147: Migration of
                         * nvcuda::wmma::matrix_b type is not supported.

                         */
                        /*
			DPCT1082:148: Migration of
                         * nvcuda::wmma::col_major type is not supported.

                         */
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                       wmma::experimental::precision::b1,
                                       wmma::col_major>
                            b_frag;
                        /*
			DPCT1082:149: Migration of
                         * nvcuda::wmma::fragment<wmma::accumulator, 8, 8, 128,
                         * int> type is not supported.
			*/
                        /*
			DPCT1082:150: Migration of
                         * nvcuda::wmma::accumulator type is not supported.

                         */
                        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N,
                                       WMMA_K, int>
                            c1_frag, c2_frag;
                        /*
			DPCT1007:151: Migration of
                         * nvcuda::wmma::fill_fragment is not supported.

                         */
                        wmma::fill_fragment(c1_frag, 0); // !!!! XOR isn't XNOR !!!!!!!!!!
                        /*
			DPCT1007:152: Migration of
                         * nvcuda::wmma::fill_fragment is not supported.

                         */
                        wmma::fill_fragment(c2_frag, 0); // !!!! XOR isn't XNOR !!!!!!!!!!

                        // 8 x 8 x 4 (uint32_t, 4 * 32 = 128 bit)
			for (; k < K; k += 128)  // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
			{
				int64_t A_cur_index = (i*lda + k) / 8;  // index in bits
				int64_t B1_cur_index = (j*ldb + k) / 8;  // index in bits
				int64_t B2_cur_index = ((j + 8)*ldb + k) / 8;  // index in bits

				// try to use A that is cached in shared memory - poor performance
				//if (i == start_i) wmma::load_matrix_sync(a_frag, &A_s[k / 32], (512 * 32));   // lda = (128*32) bits
				//else wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda);   // lda = M

				// lda, ldb - are in bits
                                /*
				DPCT1007:153:
                                 * Migration of nvcuda::wmma::load_matrix_sync
                                 * is not supported.
 */
                                wmma::load_matrix_sync(
                                    a_frag, (uint32_t *)(A + A_cur_index),
                                    lda); // lda = M

                                /*
				DPCT1007:154:
                                 * Migration of nvcuda::wmma::load_matrix_sync
                                 * is not supported.
 */
                                wmma::load_matrix_sync(
                                    b_frag, (uint32_t *)(B + B1_cur_index),
                                    ldb); // ldb = K
                                /*
				DPCT1007:297:
                                 * Migration of nvcuda::wmma::bmma_sync is not
                                 * supported.
				*/
                                wmma::bmma_sync(c1_frag, a_frag, b_frag, c1_frag); // XOR-GEMM

                                /*
				DPCT1007:155:
                                 * Migration of nvcuda::wmma::load_matrix_sync
                                 * is not supported.
 */
                                wmma::load_matrix_sync(
                                    b_frag, (uint32_t *)(B + B2_cur_index),
                                    ldb); // ldb = K
                                /*
				DPCT1007:298:
                                 * Migration of nvcuda::wmma::bmma_sync is not
                                 * supported.
				*/
                                wmma::bmma_sync(c2_frag, a_frag, b_frag, c2_frag); // XOR-GEMM
                        }
			// C[i*ldc + j]
                        /*
			DPCT1007:141: Migration of
                         * nvcuda::wmma::store_matrix_sync is not supported.

                         */
                        wmma::store_matrix_sync(&C_s[warp_id * WMMA_M * WMMA_N],
                                                c1_frag, WMMA_N,
                                                wmma::mem_row_major);
                        /*
			DPCT1007:142: Migration of
                         * nvcuda::wmma::store_matrix_sync is not supported.

                         */
                        wmma::store_matrix_sync(&C_s[warp_id * WMMA_M * WMMA_N +
                                                     WMMA_M * WMMA_N * 32],
                                                c2_frag, WMMA_N,
                                                wmma::mem_row_major);
#else // __CUDA_ARCH__ >= 730

			// Custom XOR-GEMM
			int k_d = lane_id % 4;
			int i_d = lane_id / 4;
			//int j_d = lane_id / 4;

			int32_t accum_c_val[8*2]; // wmma::fill_fragment(c_frag, 0);
			for (int local_j = 0; local_j < 8*2; ++local_j) {
				accum_c_val[local_j] = 0;
			}

			// 8 x 8 x 4 (uint32_t, 4 * 32 = 128 bit)
			for (; k < K; k += 128)  // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
			{
				//int64_t A_cur_index = (i*lda + k) / 8;
				//int64_t A_cur_index = (local_i*lda + k) / 8;
				//int64_t B_cur_index = (j*ldb + k) / 8;

				// lda, ldb - are in bits
				// 8*4 = 32
				// 8*8 = 64
				int k_d = lane_id % 4;
				int i_d = lane_id / 4;
				int j_d = lane_id / 4;
				uint32_t a_val = *(uint32_t *)(A + ((i + i_d)*lda + (k + k_d*32)) / 8); // wmma::load_matrix_sync(a_frag, (uint32_t *)(A + A_cur_index), lda);

				for (int c_x = 0; c_x < 2; c_x++)
				{
					uint32_t b_val = *(uint32_t *)(B + ((c_x * 8 + j + j_d)*ldb + (k + k_d * 32)) / 8); // wmma::load_matrix_sync(b_frag, (uint32_t *)(B + B_cur_index), ldb);

					// wmma::bmma_sync(c_frag, a_frag, b_frag, c_frag);
					int32_t c_val[8];  // 8 x 32 threads = 256
					#pragma unroll
					for (int local_j = 0; local_j < 8; ++local_j)
					{
						uint32_t b_val_cur = __shfl_custom(b_val, local_j * 4 + k_d);
						c_val[local_j] = sycl::popcount(xor_int32(a_val, b_val_cur));
					}

					#pragma unroll
					for (int local_j = 0; local_j < 8; ++local_j)
					{
						#pragma unroll
						for (int local_k = 0; local_k < 4; ++local_k) {
							accum_c_val[local_j + c_x*8] += __shfl_custom(c_val[local_j], i_d * 4 + local_k);
						}
					}
				}
			}

			// only the first 8 threads (i) contain 8 good values each, in c_val[8] (j) = 8 x 8 =64
			// wmma::store_matrix_sync(&C_s[warp_id*WMMA_M*WMMA_N], c_frag, WMMA_N, wmma::mem_row_major);
			if (k_d == 0) {
				for (int c_x = 0; c_x < 2; c_x++)
				{
					for (int local_j = 0; local_j < 8; ++local_j)
					{
						C_s[warp_id*WMMA_M*WMMA_N + i_d*WMMA_N + local_j + WMMA_M*WMMA_N*32 * c_x] = accum_c_val[local_j + c_x*8];
					}
				}
			}
#endif // __CUDA_ARCH__ >= 730

			for(int c_x = 0; c_x < 2; c_x++)
			{
				int j_d = lane_id % WMMA_N;
				{
					#pragma unroll
					for (int i_d = lane_id / WMMA_N; i_d < WMMA_M; i_d += WMMA_M / 2)
					{
						int count = C_s[warp_id*WMMA_M*WMMA_N + i_d*WMMA_N + j_d + WMMA_M*WMMA_N*32*c_x];

						const int bit_step = 128;
						int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
						count = count - f1;    // remove extra bits (from empty space for align only)

						count = (2 * count - K);

						float mean_val = mean_arr[i + i_d];
						float bias_val = bias_arr[i + i_d];
						float dst_val = count *mean_val + bias_val;
						if (leaky_activation)
							dst_val = (dst_val >= 0) ? (dst_val) : (0.1f*dst_val);    // Leaky activation

						size_t out_index = (i + i_d)*ldc + (c_x * 8 + j + j_d);
						C[out_index] = dst_val;

						if (shortcut_out_gpu) {
							shortcut_out_gpu[out_index] = shortcut_in_gpu[out_index] + dst_val;
						}
					}

				}
			}
		}
	}
}
#endif  // CUDART_VERSION >= 10000


// Coalescing
// A (weights) in the shared_memory - GOOD
/*
DPCT1110:156: The total declared local variable size in device function
 * gemm_nn_custom_bin_mean_transposed_gpu_kernel exceeds 128 bytes and may cause
 * high register pressure. Consult with your hardware vendor to find the total
 * register size available and adjust the code, or use smaller sub-group size to
 * avoid high register pressure.
*/
void gemm_nn_custom_bin_mean_transposed_gpu_kernel(
    int M, int N, int K, unsigned char *A, int lda, unsigned char *B, int ldb,
    float *C, int ldc, float *mean_arr, float *bias_arr, int leaky_activation,
    float *shortcut_in_gpu, float *shortcut_out_gpu, uint8_t *A_s)
{
	// total 57%
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        //__shared__ uint64_t A_s[6144];  // 48 KB // [lda x M`]
	//__shared__ uint8_t A_s[6144*8];  // 48 KB // [lda x M`]

        int start_i = item_ct1.get_group(2) * item_ct1.get_local_range(2) / N;
        int end_i = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                     item_ct1.get_local_range(2)) /
                        N +
                    1;

        size_t shared_size = lda * (end_i - start_i);

	int i_cur = index / N;
	int local_i = i_cur - start_i;
	// ~10%
        for (int k = item_ct1.get_local_id(2) * 64; k < shared_size;
             k += item_ct1.get_local_range(2) * 64) {
                int x = start_i*lda + k;
		if (x < (M*lda)) *((uint64_t *)(A_s + k / 8)) = *((uint64_t *)(A + x / 8));
	}
        item_ct1.barrier(sycl::access::fence_space::local_space);

        int i, j, k; //, h;
	// 47% = 29 + 10 + 8
	j = index % N;
	{    // out_h*out_w - one channel output size [169 - 173056]
		i = index / N;
		//if (i < M)  // l.n - filters [16 - 55 - 1024]
		{
			int count = 0;
			k = 0;

#ifdef NOT_USED
			// 32 thread X 256 bit = 8192 bit
			for (; k < (K - 8192); k += 8192) {   // l.size*l.size*l.c - one filter size [27 - 9216]
				ulonglong4 c_bit256;

				//int64_t A_cur_index = (i*lda + k) / 8;
				int64_t A_cur_index = (local_i*lda + k) / 8;
				int64_t B_cur_index = (j*ldb + k) / 8;
				if (i >= M) A_cur_index = 0;

#pragma unroll
				for (int t = 0; t < WARP_SIZE; ++t) {
					const int lane_id = threadIdx.x % WARP_SIZE;

					const int64_t A_i = __shfl_custom(A_cur_index, t) + 32 * lane_id;
					const int64_t B_i = __shfl_custom(B_cur_index, t) + 32 * lane_id;

					{
						//ulonglong4 a_bit256 = *((ulonglong4 *)(A + A_i));    // weights
						ulonglong4 a_bit256 = *((ulonglong4 *)(A_s + A_i));    // weights
						ulonglong4 b_bit256 = *((ulonglong4 *)(B + B_i));    // input
						c_bit256 = xor_int256(a_bit256, b_bit256);
						int tmp_count = __popcll(c_bit256.w) + __popcll(c_bit256.x) +
							__popcll(c_bit256.y) + __popcll(c_bit256.z);

						int sum_count = warpAllReduceSum(tmp_count);
						if (lane_id == t) count += sum_count;
					}
				}
			}
#endif


//#ifdef NOT_USED
			// 32 thread X 64 bit = 2048 bit // 29%
			for (; k < (K - 2048); k += 2048) {   // l.size*l.size*l.c - one filter size [27 - 9216]
				uint64_t c_bit64;

				//int64_t A_cur_index = (i*lda + k) / 8;
				int64_t A_cur_index = (local_i*lda + k) / 8;
				int64_t B_cur_index = (j*ldb + k) / 8;
				if (i >= M) A_cur_index = 0;

				#pragma unroll
				for (int t = 0; t < WARP_SIZE; ++t) {
                                        const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

                                        const int64_t A_i = __shfl_custom(A_cur_index, t) + 8 * lane_id;
					const int64_t B_i = __shfl_custom(B_cur_index, t) + 8 * lane_id;

					{
						//uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
						uint64_t a_bit64 = *((uint64_t *)(A_s + A_i));    // weights
						uint64_t b_bit64 = *((uint64_t *)(B + B_i));    // input
						c_bit64 = xor_int64(a_bit64, b_bit64);
                                                int tmp_count = sycl::popcount(c_bit64);

                                                int sum_count = warpAllReduceSum(tmp_count);
						if (lane_id == t) count += sum_count;
					}
				}
			}
//#endif

//#ifdef NOT_USED
			// 32 thread X 32 bit = 1024 bit // 10%
			for (; k < (K - 1024); k += 1024) {   // l.size*l.size*l.c - one filter size [27 - 9216]

				//int64_t A_cur_index = (i*lda + k) / 8;
				int64_t A_cur_index = (local_i*lda + k) / 8;
				int64_t B_cur_index = (j*ldb + k) / 8;
				if (i >= M) A_cur_index = 0;

				#pragma unroll
				for (int t = 0; t < WARP_SIZE; ++t) {
                                        const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

                                        const int64_t A_i = __shfl_custom(A_cur_index, t) + 4 * lane_id;
					const int64_t B_i = __shfl_custom(B_cur_index, t) + 4 * lane_id;

					{
						//uint64_t a_bit64 = *((uint64_t *)(A + A_i));    // weights
						uint32_t a_bit32 = *((uint32_t *)(A_s + A_i));    // weights
						uint32_t b_bit32 = *((uint32_t *)(B + B_i));    // input
						uint32_t c_bit32 = xor_int32(a_bit32, b_bit32);
                                                int tmp_count = sycl::popcount(c_bit32);

                                                int sum_count = warpAllReduceSum(tmp_count);
						if (lane_id == t) count += sum_count;
					}
				}
			}
//#endif

			if (i < M)
			{
				float mean_val = mean_arr[i];
				float bias_val = bias_arr[i];

//#ifdef NOT_USED
				// 8%
				for (; k < K; k += 256) {   // l.size*l.size*l.c - one filter size [27 - 144 - 9216]
					//ulonglong4 a_bit256 = *((ulonglong4 *)(A + (i*lda + k) / 8));    // weights
                                        sycl::ulong4 a_bit256 = *(
                                            (sycl::ulong4 *)(A_s +
                                                             (local_i * lda +
                                                              k) /
                                                                 8)); // weights
                                        sycl::ulong4 b_bit256 =
                                            *((sycl::ulong4 *)(B +
                                                               (j * ldb + k) /
                                                                   8)); // input
                                        sycl::ulong4 c_bit256 = xor_int256(a_bit256, b_bit256);

                                        count += sycl::popcount(c_bit256.w()) +
                                                 sycl::popcount(c_bit256.x()) +
                                                 sycl::popcount(c_bit256.y()) +
                                                 sycl::popcount(c_bit256.z());
                                }
//#endif

#ifdef NOT_USED
				for (; k < K; k += 64) {   // l.size*l.size*l.c - one filter size [27 - 9216]
					//uint64_t a_bit64 = *((uint64_t *)(A + (i*lda + k) / 8));    // weights
					uint64_t a_bit64 = *((uint64_t *)(A_s + (local_i*lda + k) / 8));    // weights
					uint64_t b_bit64 = *((uint64_t *)(B + (j*ldb + k) / 8));            // input
					uint64_t c_bit64 = xor_int64(a_bit64, b_bit64);

					count += __popcll(c_bit64);
				}
#endif

				const int bit_step = 256;
				int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
				count = count - f1;    // remove extra bits (from empty space for align only)
				float dst_val = (2 * count - K) *mean_val + bias_val;
				if(leaky_activation)
					dst_val = (dst_val >= 0) ? (dst_val) : (0.1f*dst_val);    // Leaky activation
				size_t out_index = i*ldc + j;
				C[out_index] = dst_val;

				if (shortcut_out_gpu) {
					shortcut_out_gpu[out_index] = shortcut_in_gpu[out_index] + dst_val;
				}
			}
		}
	}
}


// further optimization - use WMMA GEMM for using Tensor Cores
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/posts/tensor-cores/simpleTensorCoreGEMM.cu
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/cudaTensorCoreGemm/cudaTensorCoreGemm.cu
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-subbyte
// nvcuda::wmma::col_major ->  cutlass::MatrixLayout::kColumnMajor (matrix is not transposed)

// Matrix A	Matrix B	Accumulator	Matrix Size (m-n-k)
// precision::b1	precision::b1	int	8x8x128

// The only dimensions currently supported by WMMA for XNOR
// const int WMMA_M = 8;
// const int WMMA_N = 8;
// const int WMMA_K = 128;


// GOOD
void gemm_nn_custom_bin_mean_transposed_gpu(int M, int N, int K,
	unsigned char *A, int lda,
	unsigned char *B, int ldb,
	float *C, int ldc, float *mean_arr, float *bias, int leaky_activation,
	float *shortcut_in_gpu, float *shortcut_out_gpu)
{
#if DN_SYCL_COMPAT_RT_VERSION >= 10000
        if (1)
	{
		const int M_aligned = M + (8 - (M % 8));
		const int N_aligned = N + (16 - (N % 16));
		int size = (M_aligned / 8)*(N_aligned / 16)*WARP_SIZE;
		const int num_blocks = get_number_of_blocks(size, BLOCK);

                get_cuda_stream()->submit([&](sycl::handler &cgh) {
                        /*
		    DPCT1101:322: 'WMMA_N * WMMA_M * 32
                         * * 2' expression was replaced with a value. Modify the
                         * code to use the original expression, provided in
                         * comments, if it is correct.
		    */
                        sycl::local_accessor<int, 1> C_s_acc_ct1(
                            sycl::range<1>(4096 /*WMMA_N * WMMA_M * 32 * 2*/),
                            cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                                  sycl::range<3>(1, 1, BLOCK),
                                              sycl::range<3>(1, 1, BLOCK)),
                            [=](sycl::nd_item<3> item_ct1) {
                                    gemm_nn_custom_bin_mean_transposed_tensor_kernel(
                                        M, N, K, A, lda, B, ldb, C, ldc,
                                        mean_arr, bias, leaky_activation,
                                        shortcut_in_gpu, shortcut_out_gpu,
                                        C_s_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                            });
                });
        }
	else
#endif  //# CUDART_VERSION >= 10000
	{
		int size = M*N;
		const int num_blocks = get_number_of_blocks(size, BLOCK);

                get_cuda_stream()->submit([&](sycl::handler &cgh) {
                        sycl::local_accessor<uint8_t, 1> A_s_acc_ct1(
                            sycl::range<1>(6144 * 8 / 4), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                                  sycl::range<3>(1, 1, BLOCK),
                                              sycl::range<3>(1, 1, BLOCK)),
                            [=](sycl::nd_item<3>
                                    item_ct1) [[sycl::reqd_sub_group_size(
                                32)]] {
                                    gemm_nn_custom_bin_mean_transposed_gpu_kernel(
                                        M, N, K, A, lda, B, ldb, C, ldc,
                                        mean_arr, bias, leaky_activation,
                                        shortcut_in_gpu, shortcut_out_gpu,
                                        A_s_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                            });
                });
        }
        CHECK_CUDA(cudaPeekAtLastError());
}


void convolve_gpu_kernel(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n, int size, int pad)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        int fil;
	// filter index
	//for (fil = 0; fil < n; ++fil)
	int chan, y, x, f_y, f_x;
	// channel index
	//for (chan = 0; chan < in_c; ++chan)
	// input - y
	//for (y = 0; y < in_h; ++y)
	// input - x
	//for (x = 0; x < in_w; ++x)
	x = index % in_w;
	int index2 = index / in_w;
	y = index2 % in_h;
	fil = index2 / in_h;
	if (fil < n)
	{

		int const output_index = fil*in_w*in_h + y*in_w + x;
		float sum = 0;

		for (chan = 0; chan < in_c; ++chan)
		{
			int const weights_pre_index = fil*in_c*size*size + chan*size*size;
			int const input_pre_index = chan*in_w*in_h;

			// filter - y
			for (f_y = 0; f_y < size; ++f_y)
			{
				int input_y = y + f_y - pad;
				// filter - x
				for (f_x = 0; f_x < size; ++f_x)
				{
					int input_x = x + f_x - pad;
					if (input_y < 0 || input_x < 0 || input_y >= in_h || input_x >= in_w) continue;

					int input_index = input_pre_index + input_y*in_w + input_x;
					int weights_index = weights_pre_index + f_y*size + f_x;

					sum += input[input_index] * weights[weights_index];

				}
			}
		}
		output[output_index] = sum;
	}
}

void convolve_gpu(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n, int size, int pad)
{
	TAT(TATPARMS);

	int array_size = in_w*in_h*n;    // width X height X filters
	const int num_blocks = array_size / BLOCK + 1;

        get_cuda_stream()->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                    convolve_gpu_kernel(input, weights, output, in_w, in_h,
                                        in_c, n, size, pad);
            });
        CHECK_CUDA(cudaPeekAtLastError());
}


void convolve_bin_gpu_kernel(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n,
	int size, int pad, int new_lda, float *mean_arr_gpu, uint32_t *weights_shared,
	uint32_t *input_shared)
{
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);

        int fil;
	// filter index
	int chan, y, x, f_y, f_x;
	// channel index
	x = index % in_w;
	int index2 = index / in_w;
	y = index2 % in_h;
	fil = index2 / in_h;
	{
		int const output_index = fil*in_w*in_h + y*in_w + x;
		int sum = 0;
		int good_val = 0;

                int min_index = item_ct1.get_group(2) * item_ct1.get_local_range(2);
                int min_fil = (min_index / in_w) / in_h;
                int max_index = (item_ct1.get_group(2) + 1) * item_ct1.get_local_range(2) - 1;
                int max_fil = (max_index / in_w) / in_h;

                  // 7 KB (6 filters) - use (new_lda) for size calculation
                const int weights_size = size*size*in_c / 32 + 1;

		for (int tmp_fil = min_fil; tmp_fil <= max_fil; tmp_fil++) {
                        for (int s = item_ct1.get_local_id(2); s < weights_size;
                             s += item_ct1.get_local_range(2)) {
                                //weights_shared[s + (tmp_fil - min_fil)*new_lda / 8] = ((uint8_t *)weights)[tmp_fil*new_lda / 8 + s];
				weights_shared[s + (tmp_fil - min_fil)*new_lda/32] = ((uint32_t *)weights)[tmp_fil*new_lda / 32 + s];
			}
		}
                /*
		DPCT1065:301: Consider replacing
                 * sycl::nd_item::barrier() with
                 * sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                 * for better performance if there is no access to global
                 * memory.
		*/
                item_ct1.barrier();

                for (chan = 0; chan < in_c; ++chan)
		{
			int const input_pre_index = chan*in_w*in_h;

                           // 21.2 KB bytes (for input size 832x832)
                        const int input_shared_size = in_w*in_h / 32 + 1;
			const int add_input_index = input_pre_index % 32;
                        /*
			DPCT1118:160: SYCL group
                         * functions and algorithms must be encountered in
                         * converged control flow. You may need to adjust the
                         * code.
			*/
                        /*
			DPCT1065:302: Consider replacing
                         * sycl::nd_item::barrier() with
                         * sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                         * for better performance if there is no access to
                         * global memory.
			*/
                        item_ct1.barrier(); // why??? but is required

                        for (int s = item_ct1.get_local_id(2);
                             s < input_shared_size;
                             s += item_ct1.get_local_range(2)) {
                                input_shared[s] = ((uint32_t *)input)[input_pre_index / 32 + s];
			}
                        /*
			DPCT1118:161: SYCL group
                         * functions and algorithms must be encountered in
                         * converged control flow. You may need to adjust the
                         * code.
			*/
                        /*
			DPCT1065:303: Consider replacing
                         * sycl::nd_item::barrier() with
                         * sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                         * for better performance if there is no access to
                         * global memory.
			*/
                        item_ct1.barrier();

                        if (fil < n)    // (1-6 for one BLOCK)
			{
				// filter - y
				for (f_y = 0; f_y < size; ++f_y)
				{
					int input_y = y + f_y - pad;
					// filter - x
					for (f_x = 0; f_x < size; ++f_x)
					{
						int input_x = x + f_x - pad;
						if (input_y < 0 || input_x < 0 || input_y >= in_h || input_x >= in_w) continue;

						int weights_shared_index = (fil - min_fil)*new_lda + chan*size*size + f_y*size + f_x;
						uint8_t w_bit = get_bit((uint8_t *)weights_shared, weights_shared_index);

						int input_shared_index = /*input_pre_index +*/ input_y*in_w + input_x + add_input_index;
						uint8_t in_bit = get_bit((uint8_t *)input_shared, input_shared_index);

						int res = xnor_bit1(in_bit, w_bit);
						sum += res;
						good_val++;
					}
				}
			}
		}
		sum = sum - (good_val - sum);
                dn_sycl::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    &output[output_index], sum * mean_arr_gpu[fil]);
        }

}

void convolve_bin_gpu(float *input, float *weights, float *output, int in_w, int in_h, int in_c, int n,
	int size, int pad, int new_lda, float *mean_arr_gpu)
{
	TAT(TATPARMS);

	int array_size = in_w*in_h*n;    // width X height X filters
	const int num_blocks = array_size / BLOCK + 1;

        get_cuda_stream()->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint32_t, 1> weights_shared_acc_ct1(
                    sycl::range<1>(3 * 3 * 1024 * 6 / 32 + 1), cgh);
                sycl::local_accessor<uint32_t, 1> input_shared_acc_ct1(
                    sycl::range<1>(416 * 416 / 32 + 1), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                          sycl::range<3>(1, 1, BLOCK),
                                      sycl::range<3>(1, 1, BLOCK)),
                    [=](sycl::nd_item<3> item_ct1) {
                            convolve_bin_gpu_kernel(
                                input, weights, output, in_w, in_h, in_c, n,
                                size, pad, new_lda, mean_arr_gpu,
                                weights_shared_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get(),
                                input_shared_acc_ct1
                                    .get_multi_ptr<
                                        sycl::access::decorated::no>()
                                    .get());
                    });
        });
        CHECK_CUDA(cudaPeekAtLastError());
}

// --------------------------------

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
	return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
 for (int i =                                                                  \
          sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *   \
              sycl::ext::oneapi::this_work_item::get_nd_item<3>()              \
                  .get_local_range(2) +                                        \
          sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2); \
      i < (n);                                                                 \
      i +=                                                                     \
      sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_range(2) * \
      sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group_range(2))

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu
void im2col_gpu_kernel_ext(const int n, const float* data_im,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int height_col, const int width_col,
	float* data_col)
{
	/// @todo V3 what "index" is this referencing?  How does this compile?

	// CUDA: grid stride looping
	CUDA_KERNEL_LOOP(index, n)
	{
		const int h_index = index / width_col;
		const int h_col = h_index % height_col;
		const int w_col = index % width_col;
		const int c_im = h_index / height_col;
		const int c_col = c_im * kernel_h * kernel_w;
		const int h_offset = h_col * stride_h - pad_h;
		const int w_offset = w_col * stride_w - pad_w;
		float* data_col_ptr = data_col;
		data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
		const float* data_im_ptr = data_im;
		data_im_ptr += (c_im * height + h_offset) * width + w_offset;
		for (int i = 0; i < kernel_h; ++i)
		{
			for (int j = 0; j < kernel_w; ++j)
			{
				int h_im = h_offset + i * dilation_h;
				int w_im = w_offset + j * dilation_w;
				*data_col_ptr =
					(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
					data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
				data_col_ptr += height_col * width_col;
			}
		}
	}
}


void im2col_gpu_ext(const float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	float* data_col)
{
	TAT(TATPARMS);

	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	int width_col = (width + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int num_kernels = channels * height_col * width_col;
	// NOLINT_NEXT_LINE(whitespace/operators)
        dn_sycl::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, CAFFE_GET_BLOCKS(num_kernels)) *
                    sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS),
                sycl::range<3>(1, 1, CAFFE_CUDA_NUM_THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
                    im2col_gpu_kernel_ext(
                        num_kernels, data_im, height, width, kernel_h, kernel_w,
                        pad_h, pad_w, stride_h, stride_w, dilation_h,
                        dilation_w, height_col, width_col, data_col);
            });

        CHECK_CUDA(cudaPeekAtLastError());
}
