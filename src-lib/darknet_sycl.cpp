/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * SYCL backend (Intel GPUs via oneAPI) contributed by the DarkStudio project.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "darknet_internal.hpp"

#ifdef DARKNET_GPU_SYCL

#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/rng.hpp>

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	/// Everything we keep per SYCL device.  All queues of a device share one context so USM pointers work everywhere.
	struct DeviceState
	{
		sycl::device device;
		sycl::context context;
		sycl::queue * main_queue = nullptr;			///< used for allocations, synchronous copies, and as the default stream
		std::vector<sycl::queue *> queues;			///< every queue created on this device, including @p main_queue
		std::unordered_map<void *, size_t> allocations;
		size_t allocated_bytes = 0;
	};

	std::mutex & state_mutex()
	{
		static std::mutex m;
		return m;
	}

	thread_local int current_device = 0;
	thread_local cudaError_t last_error = cudaSuccess;
	thread_local std::string last_error_message;

	/// Errors reported asynchronously by SYCL queues (kernel failures discovered at wait time).
	std::atomic<int> async_error(cudaSuccess);
	std::mutex async_message_mutex;
	std::string async_error_message;

	void async_handler(sycl::exception_list exceptions)
	{
		for (const auto & ptr : exceptions)
		{
			try
			{
				std::rethrow_exception(ptr);
			}
			catch (const std::exception & e)
			{
				async_error = cudaErrorUnknown;
				std::scoped_lock lock(async_message_mutex);
				async_error_message = e.what();
			}
		}
	}

	/// The GPUs Darknet may use.  Prefer Level Zero; fall back to OpenCL GPUs.  Both usually list the same Intel GPU,
	/// so only one backend is used to avoid seeing every GPU twice.  ONEAPI_DEVICE_SELECTOR is honoured by SYCL itself.
	std::vector<sycl::device> & gpu_devices()
	{
		static std::vector<sycl::device> devices = []
		{
			std::vector<sycl::device> level_zero;
			std::vector<sycl::device> others;
			try
			{
				for (const auto & dev : sycl::device::get_devices(sycl::info::device_type::gpu))
				{
					(dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? level_zero : others).push_back(dev);
				}
			}
			catch (const std::exception &)
			{
				// no SYCL runtime or no GPU driver
			}
			return level_zero.empty() ? others : level_zero;
		}();
		return devices;
	}

	DeviceState & device_state(const int idx)
	{
		static std::vector<std::unique_ptr<DeviceState>> states(16);

		std::scoped_lock lock(state_mutex());
		auto & state = states.at(idx);
		if (not state)
		{
			auto & devices = gpu_devices();
			if (idx < 0 or static_cast<size_t>(idx) >= devices.size())
			{
				throw std::invalid_argument("invalid SYCL GPU index #" + std::to_string(idx));
			}

			state = std::make_unique<DeviceState>();
			state->device		= devices[idx];
			state->context		= sycl::context(state->device);
			state->main_queue	= new sycl::queue(state->context, state->device, async_handler, sycl::property::queue::in_order());
			state->queues.push_back(state->main_queue);
		}
		return *state;
	}

	DeviceState & current_state()
	{
		return device_state(current_device);
	}

	sycl::queue & queue_or_main(cudaStream_t stream)
	{
		return stream ? *stream : *current_state().main_queue;
	}

	bool is_usm(const void * ptr, const sycl::context & ctx)
	{
		return sycl::get_pointer_type(ptr, ctx) != sycl::usm::alloc::unknown;
	}

	cudaError_t fail(const cudaError_t error, const std::string & message)
	{
		dn_sycl::set_last_error(error, message.c_str());
		return error;
	}
}


// ===========================
// === dn_sycl:: helpers   ===
// ===========================

void dn_sycl::set_last_error(cudaError_t error, const char * what)
{
	last_error = error;
	last_error_message = what ? what : "";
}


cudaError_t dn_sycl::check_error(const std::function<void()> & fn)
{
	try
	{
		fn();
		return cudaSuccess;
	}
	catch (const sycl::exception & e)
	{
		return fail(e.code() == sycl::errc::memory_allocation ? cudaErrorMemoryAllocation : cudaErrorUnknown, e.what());
	}
	catch (const std::exception & e)
	{
		return fail(cudaErrorUnknown, e.what());
	}
}


sycl::queue & dn_sycl::get_in_order_queue()
{
	return *get_cuda_stream();
}


sycl::queue & dn_sycl::device_ext::in_order_queue()
{
	return dn_sycl::get_in_order_queue();
}


sycl::queue & dn_sycl::device_ext::default_queue()
{
	return dn_sycl::get_in_order_queue();
}


void dn_sycl::device_ext::queues_wait_and_throw()
{
	for (auto * q : current_state().queues)
	{
		q->wait_and_throw();
	}
}


dn_sycl::device_ext & dn_sycl::get_current_device()
{
	static device_ext ext;
	return ext;
}


void dn_sycl::has_capability_or_fail(const sycl::device & dev, const std::initializer_list<sycl::aspect> & aspects)
{
	for (const auto aspect : aspects)
	{
		if (not dev.has(aspect))
		{
			throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
				"the SYCL device \"" + dev.get_info<sycl::info::device::name>() + "\" does not support a required aspect (e.g. fp16)");
		}
	}
}


// ====================
// === CUDA runtime ===
// ====================

cudaError_t cudaGetDeviceCount(int * count)
{
	*count = static_cast<int>(gpu_devices().size());
	return *count > 0 ? cudaSuccess : fail(cudaErrorNoDevice, "no SYCL GPU found (is the Intel GPU driver installed?)");
}


cudaError_t cudaSetDevice(int device)
{
	if (device < 0 or static_cast<size_t>(device) >= gpu_devices().size())
	{
		return fail(cudaErrorInvalidDevice, "invalid SYCL GPU index #" + std::to_string(device));
	}
	current_device = device;
	return cudaSuccess;
}


cudaError_t cudaGetDevice(int * device)
{
	*device = current_device;
	return cudaSuccess;
}


cudaError_t cudaSetDeviceFlags(unsigned int)
{
	return cudaSuccess;
}


cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device)
{
	return dn_sycl::check_error([&]()
	{
		const auto & dev = device_state(device).device;
		std::memset(prop, 0, sizeof(*prop));
		const std::string name = dev.get_info<sycl::info::device::name>();
		std::strncpy(prop->name, name.c_str(), sizeof(prop->name) - 1);
		prop->major				= 0;
		prop->minor				= 0;
		prop->totalGlobalMem	= dev.get_info<sycl::info::device::global_mem_size>();
	});
}


cudaError_t cudaDeviceSynchronize()
{
	return dn_sycl::check_error([&]()
	{
		dn_sycl::get_current_device().queues_wait_and_throw();
	});
}


cudaError_t cudaRuntimeGetVersion(int * version)
{
	*version = __LIBSYCL_MAJOR_VERSION * 1000 + __LIBSYCL_MINOR_VERSION * 10;
	return cudaSuccess;
}


cudaError_t cudaDriverGetVersion(int * version)
{
	*version = 0;
	return cudaSuccess;
}


cudaError_t cudaPeekAtLastError()
{
	if (last_error != cudaSuccess)
	{
		return last_error;
	}
	return static_cast<cudaError_t>(async_error.load());
}


cudaError_t cudaGetLastError()
{
	cudaError_t error = cudaPeekAtLastError();
	if (last_error == cudaSuccess and async_error != cudaSuccess)
	{
		std::scoped_lock lock(async_message_mutex);
		last_error_message = async_error_message;
	}
	last_error = cudaSuccess;
	async_error = cudaSuccess;
	return error;
}


const char * cudaGetErrorName(cudaError_t error)
{
	switch (error)
	{
		case cudaSuccess:					return "cudaSuccess";
		case cudaErrorInvalidValue:			return "cudaErrorInvalidValue";
		case cudaErrorMemoryAllocation:		return "cudaErrorMemoryAllocation";
		case cudaErrorInitializationError:	return "cudaErrorInitializationError";
		case cudaErrorInsufficientDriver:	return "cudaErrorInsufficientDriver";
		case cudaErrorNotSupported:			return "cudaErrorNotSupported";
		case cudaErrorNoDevice:				return "cudaErrorNoDevice";
		case cudaErrorInvalidDevice:		return "cudaErrorInvalidDevice";
		default:							return "cudaErrorUnknown";
	}
}


const char * cudaGetErrorString(cudaError_t error)
{
	if (error != cudaSuccess and not last_error_message.empty())
	{
		return last_error_message.c_str();
	}
	return error == cudaSuccess ? "no error" : "SYCL error";
}


CUresult cuCtxGetCurrent(CUcontext * ctx)
{
	*ctx = reinterpret_cast<CUcontext>(&current_state().context);
	return CUDA_SUCCESS;
}


cudaError_t cudaMalloc(void ** ptr, size_t size)
{
	*ptr = nullptr;
	return dn_sycl::check_error([&]()
	{
		auto & state = current_state();
		*ptr = sycl::malloc_device(size, *state.main_queue);
		if (*ptr == nullptr)
		{
			throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
				"failed to allocate " + Darknet::size_to_IEC_string(size) + " of SYCL device memory");
		}
		std::scoped_lock lock(state_mutex());
		state.allocations[*ptr] = size;
		state.allocated_bytes += size;
	});
}


cudaError_t cudaFree(void * ptr)
{
	if (ptr == nullptr)
	{
		return cudaSuccess;
	}
	return dn_sycl::check_error([&]()
	{
		auto & state = current_state();
		sycl::free(ptr, state.context);
		std::scoped_lock lock(state_mutex());
		auto iter = state.allocations.find(ptr);
		if (iter != state.allocations.end())
		{
			state.allocated_bytes -= iter->second;
			state.allocations.erase(iter);
		}
	});
}


cudaError_t cudaHostAlloc(void ** ptr, size_t size, unsigned int)
{
	*ptr = nullptr;
	return dn_sycl::check_error([&]()
	{
		*ptr = sycl::malloc_host(size, current_state().context);
		if (*ptr == nullptr)
		{
			throw sycl::exception(sycl::make_error_code(sycl::errc::memory_allocation),
				"failed to allocate " + Darknet::size_to_IEC_string(size) + " of SYCL host memory");
		}
	});
}


cudaError_t cudaFreeHost(void * ptr)
{
	if (ptr == nullptr)
	{
		return cudaSuccess;
	}
	return dn_sycl::check_error([&]()
	{
		sycl::free(ptr, current_state().context);
	});
}


cudaError_t cudaMemGetInfo(size_t * free_bytes, size_t * total_bytes)
{
	return dn_sycl::check_error([&]()
	{
		auto & state = current_state();
		*total_bytes = state.device.get_info<sycl::info::device::global_mem_size>();
		// integrated GPUs share system memory and SYCL cannot always report free memory, so track our own allocations
		std::scoped_lock lock(state_mutex());
		*free_bytes = *total_bytes > state.allocated_bytes ? *total_bytes - state.allocated_bytes : 0;
	});
}


cudaError_t cudaMemcpy(void * dst, const void * src, size_t size, cudaMemcpyKind)
{
	return dn_sycl::check_error([&]()
	{
		current_state().main_queue->memcpy(dst, src, size).wait_and_throw();
	});
}


cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t size, cudaMemcpyKind, cudaStream_t stream)
{
	return dn_sycl::check_error([&]()
	{
		auto & q = queue_or_main(stream);
		auto event = q.memcpy(dst, src, size);

		// CUDA treats copies involving pageable host memory as synchronous with respect to the host buffer;
		// SYCL does not, so wait unless both sides are USM allocations
		const auto & ctx = current_state().context;
		if (not is_usm(dst, ctx) or not is_usm(src, ctx))
		{
			event.wait_and_throw();
		}
	});
}


cudaError_t cudaMemset(void * ptr, int value, size_t size)
{
	return dn_sycl::check_error([&]()
	{
		current_state().main_queue->memset(ptr, value, size).wait_and_throw();
	});
}


cudaError_t cudaStreamCreate(cudaStream_t * stream)
{
	return cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}


cudaError_t cudaStreamCreateWithFlags(cudaStream_t * stream, unsigned int)
{
	*stream = nullptr;
	return dn_sycl::check_error([&]()
	{
		auto & state = current_state();
		*stream = new sycl::queue(state.context, state.device, async_handler, sycl::property::queue::in_order());
		std::scoped_lock lock(state_mutex());
		state.queues.push_back(*stream);
	});
}


cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	return dn_sycl::check_error([&]()
	{
		queue_or_main(stream).wait_and_throw();
	});
}


cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int)
{
	return dn_sycl::check_error([&]()
	{
		queue_or_main(stream).ext_oneapi_submit_barrier({event->event});
	});
}


cudaError_t cudaEventCreate(cudaEvent_t * event)
{
	*event = new dn_sycl_event;
	return cudaSuccess;
}


cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int)
{
	return cudaEventCreate(event);
}


cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
	return dn_sycl::check_error([&]()
	{
		event->event = queue_or_main(stream).ext_oneapi_submit_barrier();
	});
}


cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
	return dn_sycl::check_error([&]()
	{
		event->event.wait_and_throw();
	});
}


cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t, cudaEvent_t)
{
	*ms = 0.0f;
	return fail(cudaErrorNotSupported, "cudaEventElapsedTime() is not supported by the SYCL backend");
}


cudaError_t cudaEventDestroy(cudaEvent_t event)
{
	delete event;
	return cudaSuccess;
}


cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode)
{
	return fail(cudaErrorNotSupported, "CUDA graphs are not supported by the SYCL backend");
}


cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t * graph)
{
	*graph = nullptr;
	return fail(cudaErrorNotSupported, "CUDA graphs are not supported by the SYCL backend");
}


cudaError_t cudaGraphInstantiate(cudaGraphExec_t * exec, cudaGraph_t, void *, void *, size_t)
{
	*exec = nullptr;
	return fail(cudaErrorNotSupported, "CUDA graphs are not supported by the SYCL backend");
}


cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t)
{
	return fail(cudaErrorNotSupported, "CUDA graphs are not supported by the SYCL backend");
}


// ===============================
// === cuBLAS (oneMKL BLAS)    ===
// ===============================

cublasStatus_t cublasCreate(cublasHandle_t * handle)
{
	*handle = new dn_sycl_blas_handle{&dn_sycl::get_in_order_queue()};
	return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream)
{
	if (handle == nullptr)
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}
	handle->queue = stream ? stream : current_state().main_queue;
	return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
		int m, int n, int k, const float * alpha, const float * A, int lda, const float * B, int ldb,
		const float * beta, float * C, int ldc)
{
	if (handle == nullptr)
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	using oneapi::mkl::transpose;
	const auto status = dn_sycl::check_error([&]()
	{
		oneapi::mkl::blas::column_major::gemm(*handle->queue,
			transa == CUBLAS_OP_T ? transpose::trans : transpose::nontrans,
			transb == CUBLAS_OP_T ? transpose::trans : transpose::nontrans,
			m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
	});

	return status == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_EXECUTION_FAILED;
}


// ===============================
// === cuRAND (oneMKL RNG)     ===
// ===============================

struct dn_sycl_rng
{
	unsigned long long seed = 0;
	std::unique_ptr<oneapi::mkl::rng::philox4x32x10> engine;
};


curandStatus_t curandCreateGenerator(curandGenerator_t * generator, curandRngType_t)
{
	*generator = new dn_sycl_rng;
	return CURAND_STATUS_SUCCESS;
}


curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed)
{
	generator->seed = seed;
	generator->engine.reset();
	return CURAND_STATUS_SUCCESS;
}


curandStatus_t curandGenerateUniform(curandGenerator_t generator, float * output, size_t n)
{
	const auto status = dn_sycl::check_error([&]()
	{
		auto & q = dn_sycl::get_in_order_queue();
		if (not generator->engine)
		{
			generator->engine = std::make_unique<oneapi::mkl::rng::philox4x32x10>(q, generator->seed);
		}
		oneapi::mkl::rng::generate(oneapi::mkl::rng::uniform<float>(0.0f, 1.0f), *generator->engine, static_cast<std::int64_t>(n), output);
	});
	return status == cudaSuccess ? CURAND_STATUS_SUCCESS : CURAND_STATUS_LAUNCH_FAILURE;
}


// ===========================
// === version information ===
// ===========================

void Darknet::show_sycl_info()
{
	TAT(TATPARMS);

	*cfg_and_state.output
		<< "SYCL " << Darknet::in_colour(Darknet::EColour::kBrightWhite,
			std::to_string(__LIBSYCL_MAJOR_VERSION) + "." + std::to_string(__LIBSYCL_MINOR_VERSION) + "." + std::to_string(__LIBSYCL_PATCH_VERSION))
		<< " (Intel oneAPI), oneMKL BLAS and RNG, cuDNN is " << Darknet::in_colour(Darknet::EColour::kBrightRed, "not used") << std::endl;

	const auto & devices = gpu_devices();
	if (devices.empty())
	{
		*cfg_and_state.output << "=> " << Darknet::in_colour(Darknet::EColour::kBrightRed, "no SYCL GPU detected") << std::endl;
	}

	for (size_t idx = 0; idx < devices.size(); idx ++)
	{
		const auto & dev = devices[idx];
		const std::string backend = dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? "Level Zero" : "OpenCL";
		*cfg_and_state.output
			<< "=> " << idx << ": " << Darknet::in_colour(Darknet::EColour::kBrightGreen, dev.get_info<sycl::info::device::name>())
			<< " [" << backend << ", " << dev.get_info<sycl::info::device::max_compute_units>() << " compute units"
			<< (dev.has(sycl::aspect::fp16) ? ", fp16" : "") << (dev.has(sycl::aspect::fp64) ? ", fp64" : "") << "]"
			<< ", " << Darknet::in_colour(Darknet::EColour::kYellow, Darknet::size_to_IEC_string(dev.get_info<sycl::info::device::global_mem_size>()))
			<< std::endl;
	}
}

#endif // DARKNET_GPU_SYCL
