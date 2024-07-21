#pragma once

#include <memory>
#include <vector>
#include <string>
#include <iostream>

enum class VariableType
{
    SCALAR,
    TENSOR
};

enum class DataType
{
    FLOAT32
};

struct Variable 
{
    VariableType vtype;
    DataType dtype;
};

struct Scalar : public Variable
{
    union {
        float value_f32;
    };

    explicit Scalar(const DataType dtype, float value)
    {
        vtype = VariableType::SCALAR;
        this->dtype = dtype;
        this->value_f32 = value;
    }
};

struct Tensor : public Variable
{
    bool is_input;
    std::vector<int> shape;
    std::vector<char> tensor_data;  // has to be empty for output

    explicit Tensor(
        const DataType dtype, const std::vector<int>& shape, const bool is_input=true
        ) : shape(shape), is_input(is_input)
    {
        vtype = VariableType::TENSOR;
        this->dtype = dtype;

        int tensor_size = calculate_required_memory();
        tensor_data.resize(tensor_size);
    }

    int get_num_elements() const
    {
        int length = 1;
        for (int d : shape)
        {
            length *= d;
        }
        return length;
    }

    int calculate_required_memory() const
    {
        int data_type_size = (dtype == DataType::FLOAT32 ? 4 : 2);
        int length = get_num_elements();
        return length * data_type_size;
    }

    void print() const
    {
        int length = get_num_elements();

        std::cout << "[";
        for (int ix = 0; ix < length; ++ix)
        {
            std::cout << reinterpret_cast<const float*>(tensor_data.data())[ix] << ", ";
        }
        std::cout << "] \n";
    }
};

using VariablePtr = std::shared_ptr<Variable>;

struct KernelInfo
{
    int num_threads;  // assume 1d block, and only 1 block
    std::string kernel_name;
    std::string kernel_file_path;
    std::vector<VariablePtr> arguments;
};

void execute_cuda_kernel(const KernelInfo& kernel_info);
