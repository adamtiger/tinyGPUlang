#include "kernel_executor.hpp"


int main()
{
    auto a = std::make_shared<Tensor>(DataType::FLOAT32, std::vector<int>{2, 3}, true);
    auto b = std::make_shared<Tensor>(DataType::FLOAT32, std::vector<int>{2, 3}, true);
    auto c = std::make_shared<Tensor>(DataType::FLOAT32, std::vector<int>{2, 3}, true);
    auto d = std::make_shared<Tensor>(DataType::FLOAT32, std::vector<int>{2, 3}, false);

    std::vector<float> a_data = {2.f, 1.f, 1.5f, 4.f, 5.f, 8.f};
    std::vector<float> b_data = {1.f, 3.f, 4.5f, 2.5f, 6.f, 9.f};
    std::vector<float> c_data = {1.5f, 1.f, 0.f, 1.f, 1.f, 0.5f};

    memcpy(a->tensor_data.data(), a_data.data(), 6 * sizeof(float));
    memcpy(b->tensor_data.data(), b_data.data(), 6 * sizeof(float));
    memcpy(c->tensor_data.data(), c_data.data(), 6 * sizeof(float));

    KernelInfo kernel_info = {};
    kernel_info.num_threads = 6;
    
    kernel_info.arguments.push_back(a);
    kernel_info.arguments.push_back(b);
    kernel_info.arguments.push_back(c);
    kernel_info.arguments.push_back(d);

    kernel_info.kernel_name = "calc_complex";
    kernel_info.kernel_file_path = "C:\\Data\\AI\\works\\tinyGPUlang\\artifacts\\calc_complex.ptx";

    execute_cuda_kernel(kernel_info);

    // print output
    d->print();

    return 0;
}
