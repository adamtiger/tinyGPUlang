#include "parser.hpp"
#include <fstream>
#include <iostream>

// static class variables
std::unordered_set<char> TGLparser::ignored_chars = 
{
    ' ',   // space
    '\t',  // tab
    '\n'   // new line
};

std::unordered_set<char> TGLparser::bracket_chars = 
{
    '(',  // for functions or arithmetic expressions
    ')',  // --
    '[',  // only for tensor variable type def.
    ']',  // --
    '{',  // only for function body test
    '}'   // --
};

std::unordered_set<char> TGLparser::comma_chars =
{
    ',',
    ';'
};

std::unordered_set<char> TGLparser::arithmetic_chars =
{
    '*',
    '/',
    '+',
    '-',
    '='
};

std::unordered_set<std::string> TGLparser::builtin_kernel_names = 
{
    "sqrt",
    "exp2",
    "log2",
    "abs",
    "sin"
};

// class functions

TGLparser::TGLparser(const std::string& path_to_tgl)
{
    read_all_lines_from_tgl(path_to_tgl);

    for (auto& line : all_lines)
    {
        int start_pos = 0;
        std::string ntoken = "start";
        while (ntoken != "")
        {
            start_pos = parse_next_token(ntoken, line, start_pos);
            std::cout << ntoken << "\n";
        }
    }
    
    int next_line;
    int next_pos;
    parse_next_kernel(0, 0, next_line, next_pos);

    auto kernel = defined_global_kernels["calc_mse"];

    std::cout << kernel->name << std::endl;
    std::cout << (int)kernel->scope << std::endl;

    for (auto v : kernel->arguments)
    {
        std::cout << v->name << std::endl;
        std::cout << (int)v->dtype << std::endl;
        std::cout << (int)v->vtype << std::endl;
        
        if (v->vtype == VariableType::TENSOR)
        {
            auto shape = std::static_pointer_cast<TensorNode>(v)->shape;

            for (int s : shape)
            {
                std::cout << s << " ";
            }
            std::cout << std::endl;
        }
        
    }
}

std::vector<KernelNodePtr> TGLparser::get_all_global_kernel() const
{
    return {};
}

KernelNodePtr TGLparser::get_global_kernel(const std::string& kernel_name) const
{
    return nullptr;
}

// helper functions
void TGLparser::read_all_lines_from_tgl(const std::string& path_to_tgl)
{
    std::ifstream tgl_file(path_to_tgl);

    if (tgl_file)
    {
        std::cout << "Source file was opened successfully " << path_to_tgl << std::endl;

        while(!tgl_file.eof())
        {
            std::string line;
            std::getline(tgl_file, line);
            all_lines.push_back(line);

            //std::cout << line << "\n";
        }

        std::cout << "Source file was processed successfully " << path_to_tgl << std::endl;
    }
    else
    {
        std::cout << "Error while opening file " << path_to_tgl << std::endl;
    }
}

int TGLparser::parse_next_token(std::string& next_token, const std::string& line, const int start_pos)
{
    next_token = "";

    int current_pos = start_pos;
    if (static_cast<size_t>(start_pos) < line.size())
    {
        char c = line[current_pos];

        // go until characters can be ignored
        while (ignored_chars.contains(c))
        {
            current_pos += 1;
            c = line[current_pos];
        }

        // ignore comments
        if (c == '#')
        {
            next_token = "";
            return current_pos;
        }

        // return immediately if the character is special or arithmetic
        if (bracket_chars.contains(c) || arithmetic_chars.contains(c) || comma_chars.contains(c))
        {
            next_token += c;
            return current_pos + 1;
        }

        // can be the start of keyword or variable (func) name
        // TODO: simplify
        int first_pos = current_pos;
        bool proceed = true;
        while (proceed)
        {
            current_pos += 1;

            proceed = static_cast<size_t>(current_pos) < line.size();
            if (proceed)
            {
                c = line[current_pos];
                proceed = proceed && !(bracket_chars.contains(c) || arithmetic_chars.contains(c) || ignored_chars.contains(c) || comma_chars.contains(c));
            }
        }

        // cut the token
        int last_pos = current_pos;
        next_token = line.substr(first_pos, last_pos - first_pos);
    }
    return current_pos;
}

void TGLparser::parse_next_kernel(const int start_line, const int start_pos, int& next_line, int& next_pos)
{
    int current_line = start_line;
    int current_pos = start_pos;
    std::string next_token;
    
    // find the line with the next function header
    bool found_func = false;
    while (!found_func)
    {
        if (current_line >= all_lines.size())
        {
            break;
        }

        auto& cline = all_lines[current_line];
        
        bool cont = true;
        while (cont)
        {
            current_pos = parse_next_token(next_token, cline, current_pos);
            found_func = next_token == "func";
            cont = !(next_token == "" || found_func);
        }

        if (!found_func)
            current_line += 1;
    }

    // parse kernel header and save it
    KernelNodePtr kernel = parse_kernel_header(current_line, current_pos, current_pos);
    if (kernel == nullptr)  // no more function can be found
    {
        next_line = static_cast<int>(all_lines.size());
        next_pos = 0;
        return;
    }

    if (kernel->scope == KernelScope::GLOBAL)
    {
        if (defined_global_kernels.contains(kernel->name))
        {
            std::cout << "Global kernel already defined " << kernel->name << "\n";
        }

        defined_global_kernels.insert({kernel->name, kernel});
    }
    else  // device
    {
        if (defined_global_kernels.contains(kernel->name))
        {
            std::cout << "Device kernel already defined " << kernel->name << "\n";
        }

        defined_device_kernels.insert({kernel->name, kernel});
    }

    // return
    next_line = current_line;
    next_pos = current_pos;
}

KernelNodePtr TGLparser::parse_kernel_header(const int start_line, const int start_pos, int& next_pos)
{
    int current_line = start_line;
    int current_pos = start_pos;
    std::string next_token;

    // parse the function type
    auto& cline = all_lines[current_line];
    current_pos = parse_next_token(next_token, cline, current_pos);
    
    KernelScope kernel_scope;
    if (next_token == "global")
    {
        kernel_scope = KernelScope::GLOBAL;
    }
    else if (next_token == "device")
    {
        kernel_scope = KernelScope::DEVICE;
    }
    else
    {
        std::cout << "Wrong scope for function " << next_token << "\n";
        return nullptr;
    }

    // parse the return type
    current_pos = parse_next_token(next_token, cline, current_pos);

    VariableNodePtr return_var_type = nullptr; 
    if (next_token != "void")
    {
        if (next_token != "f32" || next_token != "f16")
        {
            std::cout << "Wrong variable type " << next_token << "\n";
        }

        return_var_type = parse_variable_type(current_line, current_pos - 3, current_pos);  // f32 or f16;
    }

    // parse the function name
    std::string kernel_name;
    current_pos = parse_next_token(kernel_name, cline, current_pos);

    // parse argument var types
    std::vector<VariableNodePtr> args;
    
    std::string start_parantheses;
    current_pos = parse_next_token(start_parantheses, cline, current_pos);

    if (start_parantheses != "(")
    {
        std::cout << "Expected a ( character instead of " << next_token << "\n";
    }

    while (next_token != ")")
    {
        auto var_type = parse_variable_type(current_line, current_pos, current_pos);
        current_pos = parse_next_token(next_token, cline, current_pos);
        args.push_back(var_type);
    }

    // return values
    next_pos = current_pos;
    return std::make_shared<KernelNode>(kernel_name, kernel_scope, args, return_var_type);
}


VariableNodePtr TGLparser::parse_variable_type(const int start_line, const int start_pos, int& next_pos)
{
    VariableNodePtr var = nullptr;
    int current_pos = start_pos;
    
    std::string next_token;
    auto& cline = all_lines[start_line];  // variable type def has to be in one line

    // read the data type
    DataType dtype;
    current_pos = parse_next_token(next_token, cline, current_pos);
    if (next_token == "f32")
    {
        dtype = DataType::FLOAT32;
    }
    else if (next_token == "f16")
    {
        dtype = DataType::FLOAT16;
    }
    else
    {
        std::cout << "Expected a f32 or f16, but got instead " << next_token << "\n";
    }

    // decide variable type
    VariableType vtype;
    current_pos = parse_next_token(next_token, cline, current_pos);
    if (next_token == "[")  // this has to be a tensor
    {
        vtype = VariableType::TENSOR;
    }
    else  // has to be a scalar
    {
        vtype = VariableType::SCALAR;
    }

    // if tensor read the shape and the name
    if (vtype == VariableType::TENSOR)
    {
        // read the shape
        std::vector<int> shape;
        while (next_token != "]")
        {
            current_pos = parse_next_token(next_token, cline, current_pos);
            if (next_token != "]" && next_token != ",")
            {
                int s = atoi(next_token.c_str());
                shape.push_back(s);
            }
        }

        // read the var name
        current_pos = parse_next_token(next_token, cline, current_pos);
        std::string var_name = next_token;

        var = std::make_shared<TensorNode>(vtype, dtype, var_name, shape);
    }

    // if scalar (return the type)
    if (vtype == VariableType::SCALAR)
    {
        auto var_name = next_token;  // for scalar, the last read is the var. name
        var = std::make_shared<ScalarNode>(vtype, dtype, var_name);
    }
    
    next_pos = current_pos;
    return var;
}
