# SYCL kernels (Intel GPUs)

These files are the SYCL versions of Darknet's CUDA kernels (`src-lib/*.cu`). They're compiled only when Darknet is configured with `-DDARKNET_TRY_SYCL=ON` using the Intel oneAPI DPC++/C++ compiler. This backend was contributed by the DarkStudio project.

| File | Generated from |
|---|---|
| `activation_kernels.dp.cpp` | `activation_kernels.cu` |
| `avgpool_layer_kernels.dp.cpp` | `avgpool_layer_kernels.cu` |
| `blas_kernels.dp.cpp` | `blas_kernels.cu` |
| `col2im_kernels.dp.cpp` | `col2im_kernels.cu` |
| `convolutional_kernels.dp.cpp` | `convolutional_kernels.cu` |
| `crop_layer_kernels.dp.cpp` | `crop_layer_kernels.cu` |
| `dropout_layer_kernels.dp.cpp` | `dropout_layer_kernels.cu` |
| `im2col_kernels.dp.cpp` | `im2col_kernels.cu` |
| `maxpool_layer_kernels.dp.cpp` | `maxpool_layer_kernels.cu` |
| `network_kernels.dp.cpp` | `network_kernels.cu` |

## How they were made

1. [SYCLomatic](https://github.com/oneapi-src/SYCLomatic) (Intel's open-source CUDA→SYCL tool) converted the `.cu` files, configured like the ROCm build: GPU code, CUDA runtime API, **no cuDNN**.
2. An import step adapted the output for Darknet:
   - SYCLomatic's `dpct` helpers are replaced by the small `dn_sycl::` helpers in `darknet_sycl.hpp`, so there's no dependency on the dpct headers.
   - `CHECK_CUDA(0)` (SYCLomatic's replacement for `cudaPeekAtLastError()`) is turned back into `CHECK_CUDA(cudaPeekAtLastError())`, which the compatibility layer supports.
   - Kernel lambdas capture only the `Darknet::Layer` / `NetworkState` fields they use. Capturing the whole struct exceeds the 2 KiB kernel-argument limit of Intel GPUs.

The scripts are in the DarkStudio repository (`tools/sycl-migrate/`). After the initial import, these files are maintained by hand.

## How the rest of Darknet uses them

The host code (`dark_cuda.cpp`, the layers, `gemm.cpp`, ...) isn't duplicated. `darknet_sycl.hpp` / `darknet_sycl.cpp` provide the CUDA runtime, cuBLAS and cuRAND calls Darknet uses, implemented with SYCL and oneMKL:

| CUDA | SYCL backend |
|---|---|
| `cudaStream_t` | in-order `sycl::queue *` (one context per device) |
| `cudaMalloc` / `cudaHostAlloc` | USM device / host memory |
| `cublasSgemm` | `oneapi::mkl::blas::column_major::gemm` |
| `curandGenerateUniform` | oneMKL RNG (philox4x32x10) |
| error codes | SYCL exceptions caught and returned as `cudaError_t` |
| CUDA graphs | not supported, so Darknet uses normal launches |
| tensor cores / `wmma` (XNOR nets) | not used. `DN_SYCL_CUDA_ARCH` is 0, so the generic code runs |

## Known limitations

- No cuDNN equivalent yet, so convolutions run as im2col + oneMKL SGEMM (like the ROCm build). oneDNN is a possible later step.
- Many Intel GPUs (e.g. Iris Xe) have no FP64. Kernels must stay in single precision, which means `float` literals (`0.5f`, not `0.5`).
- The kernels are JIT-compiled at the first launch. Use `-DDARKNET_SYCL_TARGETS=...` for ahead-of-time compilation.
