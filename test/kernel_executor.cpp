#include "kernel_executor.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include "cuda.h"

static void checkCudaErrors(CUresult err)
{
    assert(err == CUDA_SUCCESS);
}

/// main - Program entry point
void execute_cuda_kernel(const KernelInfo& kernel_info)
{
    CUdevice device;
    CUmodule cudaModule;
    CUcontext context;
    CUfunction function;
    CUlinkState linker;
    int devCount;

    // CUDA initialization
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGetCount(&devCount));
    checkCudaErrors(cuDeviceGet(&device, 0));

    char name[128];
    checkCudaErrors(cuDeviceGetName(name, 128, device));
    std::cout << "Using CUDA Device [0]: " << name << "\n";

    auto kernel_path = kernel_info.kernel_file_path;
    std::ifstream t(kernel_path);
    if (!t.is_open())
    {
        std::cerr << kernel_path << " not found\n";
        return;
    }
    std::string str((std::istreambuf_iterator<char>(t)),
                    std::istreambuf_iterator<char>());

    // Create driver context
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    // Create module for object
    checkCudaErrors(cuModuleLoadDataEx(&cudaModule, str.c_str(), 0, 0, 0));

    // Get kernel function
    checkCudaErrors(cuModuleGetFunction(&function, cudaModule, kernel_info.kernel_name.c_str()));
    
    // create argument params for the kernel
    std::vector<CUdeviceptr> device_ptrs;  // the variables should live until the end
    std::vector<char*> kernel_params;

    // Device data
    for (auto& var : kernel_info.arguments)
    {
        if (var->vtype == VariableType::TENSOR)
        {
            auto tvar = std::static_pointer_cast<Tensor>(var);
            size_t data_size = tvar->calculate_required_memory();

            CUdeviceptr temp;
            device_ptrs.push_back(temp);
            auto& devBufferVar = device_ptrs[device_ptrs.size() - 1];
            checkCudaErrors(cuMemAlloc(&devBufferVar, data_size));

            if (tvar->is_input)
            {
                checkCudaErrors(cuMemcpyHtoD(devBufferVar, tvar->tensor_data.data(), data_size));
            }

            kernel_params.push_back(reinterpret_cast<char*>(&devBufferVar));
        }
        else  // it is a scalar
        {
            auto svar = std::static_pointer_cast<Scalar>(var);

            if (svar->dtype == DataType::FLOAT32)
            {
                kernel_params.push_back(reinterpret_cast<char*>(&svar->value_f32));
            }
            else
            {
                std::cerr << "Unsupported data type \n";
            }
        }
    }

    unsigned blockSizeX = kernel_info.num_threads;
    unsigned blockSizeY = 1;
    unsigned blockSizeZ = 1;
    unsigned gridSizeX = 1;
    unsigned gridSizeY = 1;
    unsigned gridSizeZ = 1;

    std::cout << "Launching kernel\n";

    // Kernel launch
    checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                                   blockSizeX, blockSizeY, blockSizeZ,
                                   0, NULL, (void**)kernel_params.data(), NULL));

    // Retrieve device data
    int kidx = 0;
    for (auto& var : kernel_info.arguments)
    {
        if (var->vtype == VariableType::TENSOR)
        {
            auto tvar = std::static_pointer_cast<Tensor>(var);
            size_t data_size = tvar->calculate_required_memory();
            if (!tvar->is_input)
            {
                checkCudaErrors(cuMemcpyDtoH(tvar->tensor_data.data(), *reinterpret_cast<CUdeviceptr*>(kernel_params[kidx]), data_size));
            }

            kidx++;
        }
    }

    // Clean-up
    for (auto& dev_ptr : device_ptrs)
    {
        checkCudaErrors(cuMemFree(dev_ptr));
    }
    checkCudaErrors(cuModuleUnload(cudaModule));
    checkCudaErrors(cuCtxDestroy(context));
}
