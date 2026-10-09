/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * SYCL backend (Intel GPUs via oneAPI) contributed by the DarkStudio project.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @file
 * A small CUDA-on-SYCL compatibility layer used when Darknet is built with @p DARKNET_GPU_SYCL.
 *
 * Darknet's host code (dark_cuda.cpp, the layers, gemm.cpp, ...) is written against the CUDA runtime, cuBLAS and
 * cuRAND APIs.  Instead of rewriting that code, this file provides the small subset of those APIs that Darknet uses,
 * implemented with SYCL, oneMKL BLAS and oneMKL RNG.  This is the same approach the AMD ROCm build takes with the
 * macros in darknet_gpu.hpp, except SYCL is not source-compatible with CUDA so real functions are needed.
 *
 * The CUDA kernels themselves (the .cu files) were converted to SYCL with SYCLomatic and live in src-lib/sycl/.
 * The helpers in the @p dn_sycl namespace replace the few SYCLomatic "dpct" helpers those files use, so there is
 * no dependency on the dpct headers.
 *
 * Semantics worth knowing:
 *
 * - A @p cudaStream_t is an in-order @p sycl::queue.  All queues of a device share one SYCL context.
 * - Device memory is SYCL USM device memory, and "pinned" host memory is USM host memory.
 * - CUDA copies to or from regular (pageable) host memory return only once the host buffer can be reused.  SYCL
 *   does not guarantee that, so such copies wait for completion.  Copies between device and USM host memory stay
 *   asynchronous.
 * - SYCL reports errors with exceptions.  They are caught here and turned into a @p cudaError_t which is then
 *   available through @p cudaGetLastError(), so Darknet's existing @p CHECK_CUDA() error handling keeps working.
 */

#ifdef DARKNET_GPU_SYCL

#include <sycl/sycl.hpp>

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>


// =====================
// === CUDA runtime  ===
// =====================

enum cudaError
{
	cudaSuccess						= 0,
	cudaErrorInvalidValue			= 1,
	cudaErrorMemoryAllocation		= 2,
	cudaErrorInitializationError	= 3,
	cudaErrorInsufficientDriver		= 35,
	cudaErrorNotSupported			= 801,
	cudaErrorNoDevice				= 100,
	cudaErrorInvalidDevice			= 101,
	cudaErrorUnknown				= 999,
};
typedef cudaError cudaError_t;

enum cudaMemcpyKind
{
	cudaMemcpyHostToHost		= 0,
	cudaMemcpyHostToDevice		= 1,
	cudaMemcpyDeviceToHost		= 2,
	cudaMemcpyDeviceToDevice	= 3,
	cudaMemcpyDefault			= 4,
};

#define cudaHostAllocDefault			0x00
#define cudaHostRegisterMapped			0x02
#define cudaStreamDefault				0x00
#define cudaStreamNonBlocking			0x01
#define cudaEventDefault				0x00
#define cudaEventDisableTiming			0x02
#define cudaEventWaitDefault			0x00
#define cudaDeviceScheduleBlockingSync	0x04

/// A CUDA stream is an in-order SYCL queue.
typedef sycl::queue * cudaStream_t;

/// A CUDA event is a SYCL event (a barrier submitted to a queue).
struct dn_sycl_event { sycl::event event; };
typedef dn_sycl_event * cudaEvent_t;

/// Only the fields Darknet uses.  @p major and @p minor are 0 so code paths for specific NVIDIA architectures
/// (e.g. tensor cores) are never selected.
struct cudaDeviceProp
{
	char name[256];
	int major;
	int minor;
	size_t totalGlobalMem;
};

/// CUDA's 3D launch configuration.  Converts to the equivalent SYCL range, which lists dimensions as (z, y, x).
struct dim3
{
	unsigned int x;
	unsigned int y;
	unsigned int z;
	constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
	operator sycl::range<3>() const { return sycl::range<3>(z, y, x); }
};

// driver API, only used to print the current context (CUcontext itself is declared by the SYCL headers for CUDA interop)
typedef int CUresult;
#define CUDA_SUCCESS 0

cudaError_t cudaGetDeviceCount		(int * count);
cudaError_t cudaSetDevice			(int device);
cudaError_t cudaGetDevice			(int * device);
cudaError_t cudaSetDeviceFlags		(unsigned int flags);
cudaError_t cudaGetDeviceProperties	(cudaDeviceProp * prop, int device);
cudaError_t cudaDeviceSynchronize	();
cudaError_t cudaRuntimeGetVersion	(int * version);
cudaError_t cudaDriverGetVersion	(int * version);
cudaError_t cudaGetLastError		();
cudaError_t cudaPeekAtLastError		();
const char * cudaGetErrorName		(cudaError_t error);
const char * cudaGetErrorString		(cudaError_t error);
CUresult cuCtxGetCurrent			(CUcontext * ctx);

cudaError_t cudaMalloc				(void ** ptr, size_t size);
cudaError_t cudaFree				(void * ptr);
cudaError_t cudaHostAlloc			(void ** ptr, size_t size, unsigned int flags);
cudaError_t cudaFreeHost			(void * ptr);
cudaError_t cudaMemGetInfo			(size_t * free_bytes, size_t * total_bytes);
cudaError_t cudaMemcpy				(void * dst, const void * src, size_t size, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync			(void * dst, const void * src, size_t size, cudaMemcpyKind kind, cudaStream_t stream = nullptr);
cudaError_t cudaMemset				(void * ptr, int value, size_t size);

cudaError_t cudaStreamCreate		(cudaStream_t * stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t * stream, unsigned int flags);
cudaError_t cudaStreamSynchronize	(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent		(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);

cudaError_t cudaEventCreate			(cudaEvent_t * event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags);
cudaError_t cudaEventRecord			(cudaEvent_t event, cudaStream_t stream = nullptr);
cudaError_t cudaEventSynchronize	(cudaEvent_t event);
cudaError_t cudaEventElapsedTime	(float * ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy		(cudaEvent_t event);

// CUDA graphs are not supported by the SYCL backend (yet); these always fail so Darknet falls back to normal launches.
typedef void * cudaGraph_t;
typedef void * cudaGraphExec_t;
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0 };
cudaError_t cudaStreamBeginCapture	(cudaStream_t stream, cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture	(cudaStream_t stream, cudaGraph_t * graph);
cudaError_t cudaGraphInstantiate	(cudaGraphExec_t * exec, cudaGraph_t graph, void * a = nullptr, void * b = nullptr, size_t c = 0);
cudaError_t cudaGraphLaunch			(cudaGraphExec_t exec, cudaStream_t stream);


// ===============================
// === cuBLAS (oneMKL BLAS)    ===
// ===============================

enum cublasStatus_t
{
	CUBLAS_STATUS_SUCCESS			= 0,
	CUBLAS_STATUS_NOT_INITIALIZED	= 1,
	CUBLAS_STATUS_EXECUTION_FAILED	= 13,
};

enum cublasOperation_t
{
	CUBLAS_OP_N = 0,
	CUBLAS_OP_T = 1,
};

struct dn_sycl_blas_handle { sycl::queue * queue; };
typedef dn_sycl_blas_handle * cublasHandle_t;

cublasStatus_t cublasCreate		(cublasHandle_t * handle);
cublasStatus_t cublasSetStream	(cublasHandle_t handle, cudaStream_t stream);

/// Column-major SGEMM, same as cuBLAS.  @p alpha and @p beta are host pointers.
cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
		int m, int n, int k, const float * alpha, const float * A, int lda, const float * B, int ldb,
		const float * beta, float * C, int ldc);


// ===============================
// === cuRAND (oneMKL RNG)     ===
// ===============================

enum curandStatus_t
{
	CURAND_STATUS_SUCCESS			= 0,
	CURAND_STATUS_LAUNCH_FAILURE	= 201,
};

enum curandRngType_t
{
	CURAND_RNG_PSEUDO_DEFAULT = 100,
};

struct dn_sycl_rng;
typedef dn_sycl_rng * curandGenerator_t;

curandStatus_t curandCreateGenerator				(curandGenerator_t * generator, curandRngType_t type);
curandStatus_t curandSetPseudoRandomGeneratorSeed	(curandGenerator_t generator, unsigned long long seed);
curandStatus_t curandGenerateUniform				(curandGenerator_t generator, float * output, size_t n);


// ==================================================
// === Helpers used by the converted kernel files ===
// ==================================================

/// CUDA runtime version the kernels were converted from (SYCLomatic + CUDA 12.9 headers).
#define DN_SYCL_COMPAT_RT_VERSION 12090

/// Stands in for @p __CUDA_ARCH__.  Zero, so architecture-specific CUDA code (e.g. tensor cores) takes the generic path.
#define DN_SYCL_CUDA_ARCH 0

namespace dn_sycl
{
	/// Remember an error so @p cudaGetLastError() reports it.
	void set_last_error(cudaError_t error, const char * what = nullptr);

	/// Run @p fn and convert any SYCL or oneMKL exception into a @p cudaError_t.
	cudaError_t check_error(const std::function<void()> & fn);

	/// The queue for the current device (same as @p get_cuda_stream()).
	sycl::queue & get_in_order_queue();

	/// Minimal stand-in for the dpct device object used by the converted code.
	struct device_ext
	{
		sycl::queue & in_order_queue();
		sycl::queue & default_queue();
		void queues_wait_and_throw();
	};
	device_ext & get_current_device();

	/// Throws if the device lacks one of the aspects (e.g. fp16).
	void has_capability_or_fail(const sycl::device & dev, const std::initializer_list<sycl::aspect> & aspects);

	typedef sycl::queue * queue_ptr;

	/// Single-precision pow.  An integral exponent uses @p sycl::pown, everything else @p sycl::pow.  Always float,
	/// since many Intel GPUs (e.g. Iris Xe) have no FP64.
	template <typename T, typename U>
	inline float pow(const T base, const U exponent)
	{
		if constexpr (std::is_integral_v<U>)
		{
			return sycl::pown(static_cast<float>(base), static_cast<int>(exponent));
		}
		else
		{
			return sycl::pow(static_cast<float>(base), static_cast<float>(exponent));
		}
	}

	template <typename T>
	inline T permute_sub_group_by_xor(const sycl::sub_group & sg, const T value, const unsigned int mask)
	{
		return sycl::permute_group_by_xor(sg, value, mask);
	}

	template <typename T>
	inline T select_from_sub_group(const sycl::sub_group & sg, const T value, const int index)
	{
		return sycl::select_from_group(sg, value, index);
	}

	template <sycl::access::address_space Space = sycl::access::address_space::generic_space, typename T, typename U>
	inline T atomic_fetch_add(T * address, const U operand)
	{
		sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, Space> ref(*address);
		return ref.fetch_add(static_cast<T>(operand));
	}

	template <sycl::access::address_space Space = sycl::access::address_space::generic_space, typename T, typename U>
	inline T atomic_fetch_min(T * address, const U operand)
	{
		sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device, Space> ref(*address);
		return ref.fetch_min(static_cast<T>(operand));
	}

	/// Same as CUDA's @p __brev().
	template <typename T>
	inline T reverse_bits(T value)
	{
		static_assert(std::is_unsigned_v<T>, "reverse_bits() requires an unsigned type");
		T result = 0;
		for (unsigned int i = 0; i < sizeof(T) * 8; i ++)
		{
			result = (result << 1) | (value & 1);
			value >>= 1;
		}
		return result;
	}
}

/// Replaces SYCLomatic's DPCT_CHECK_ERROR(): evaluate the expression and return a @p cudaError_t instead of throwing.
#define DN_SYCL_CHECK_ERROR(expr) dn_sycl::check_error([&]() { expr; })

namespace Darknet
{
	/// Display the SYCL runtime and the GPU(s) Darknet can use.  The SYCL equivalent of @p show_rocm_info().
	void show_sycl_info();
}

#endif // DARKNET_GPU_SYCL
