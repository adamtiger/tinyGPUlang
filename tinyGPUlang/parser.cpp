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

std::unordered_map<char, int> TGLparser::arithmetic_precedences = 
{
    {'*', 2},
    {'/', 2},
    {'+', 1},
    {'-', 1}
};

// class functions

TGLparser::TGLparser(const std::string& path_to_tgl)
{
    read_all_lines_from_tgl(path_to_tgl);
    
    int current_line = 0;
    int current_pos = 0;
    while (current_line < all_lines.size())
    {
        parse_next_kernel(current_line, current_pos, current_line, current_pos);
    }

    std::cout << "Source file was parsed successfully " << path_to_tgl << "\n";
    std::cout << "Parsed " << defined_global_kernels.size() << " global kernels" << std::endl;
}

std::vector<KernelNodePtr> TGLparser::get_all_global_kernel() const
{
    std::vector<KernelNodePtr> kernels;
    for (auto& n_k : defined_global_kernels)
    {
        kernels.push_back(n_k.second);
    }

    return kernels;
}

KernelNodePtr TGLparser::get_global_kernel(const std::string& kernel_name) const
{
    return defined_global_kernels.at(kernel_name);
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
        }
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
        {
            current_line += 1;
            current_pos = 0;
        }
    }

    if (!found_func)  // no more function can be found
    {
        next_line = static_cast<int>(all_lines.size());
        next_pos = 0;
        return;
    }

    // parse kernel header and save it
    KernelNodePtr kernel = parse_kernel_header(current_line, current_pos, current_pos);
    if (kernel == nullptr)  // error during kernel parsing
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

    // parse kernel body
    parse_kernel_body(kernel, current_line, current_pos, current_line, current_pos);

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
    int prev_pos = current_pos;
    current_pos = parse_next_token(next_token, cline, current_pos);

    VariableNodePtr return_var_type = nullptr; 
    if (next_token != "void")
    {
        if (next_token != "f32" && next_token != "f16")
        {
            std::cout << "Wrong variable type " << next_token << "\n";
        }

        return_var_type = parse_variable_type(current_line, prev_pos, current_pos);  // f32 or f16;
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
        std::cout << "Expected a ( character instead of " << start_parantheses << "\n";
    }

    while (next_token != ")")
    {
        auto var = parse_variable_type(current_line, current_pos, current_pos);
        current_pos = parse_next_token(next_token, cline, current_pos);  // read the var. name
        var->name = next_token;
        current_pos = parse_next_token(next_token, cline, current_pos);  // read delimiter
        args.push_back(var);
        defined_nodes.insert({var->name, var});
    }

    // return values
    next_pos = current_pos;
    auto node = std::make_shared<KernelNode>(kernel_name, kernel_scope, args, return_var_type);
    defined_nodes.insert({kernel_name, node});
    return node;
}

void TGLparser::parse_kernel_body(KernelNodePtr kernel, const int start_line, const int start_pos, int& next_line, int& next_pos)
{
    std::string next_token;
    int current_line = start_line;
    int current_pos = start_pos;

    // search for the { to know the start of the kernel body
    while (next_token != "{")
    {
        if (current_line >= all_lines.size())
        {
            std::cout << "Expected a { character for starting the kernel body \n";
            return;
        }

        auto& cline = all_lines[current_line];
        current_pos = parse_next_token(next_token, cline, current_pos);

        if (next_token == "")  // line ended
        {
            current_line += 1;
            current_pos = 0;
        }
    }

    while (next_token != "}")  // until the end of the body
    {
        // next token can be either of the following:
        //    - var v = arithmetic_expression;
        //    - function_call(args);
        //    - v = arithmetic_expression;
        //    - return v;
        // each situation requires different handling

        // get next token
        int line_expression_start_pos = current_pos;
        auto& cline = all_lines[current_line];
        current_pos = parse_next_token(next_token, cline, current_pos);
        
        // handle if next token is empty (no more expressions in current line)
        if (next_token == "")
        {
            current_line += 1;
            current_pos = 0;
        }
        // handle if next token is var (it is not ambigous)
        else if (next_token == "var")
        {
            auto node = parse_alias_node(cline, current_pos, current_pos);
            kernel->body.push_back(node);
        }
        // handle if next token is return
        else if (next_token == "return")
        {
            auto node = parse_return_node(cline, current_pos, current_pos);
            kernel->body.push_back(node);
        }
        // handle the function call or the assignment case
        else if (next_token != "}")
        {
            std::string first_token = next_token;
            current_pos = parse_next_token(next_token, cline, current_pos);
            
            // handle the assignment case
            if (next_token == "=")
            {
                auto node = parse_assignment_node(first_token, cline, current_pos, current_pos);
                kernel->body.push_back(node);
            }
            // handle the potential function call case
            else if (next_token == "(")
            {
                auto node = parse_arithmetic_node(cline, line_expression_start_pos, current_pos);
                kernel->body.push_back(node);
            }
            // unexpected case
            else
            {
                std::cout << "Unexpected expression, starts with " << first_token << std::endl;
            }
        }
    }
    
    // return
    next_line = current_line;
    next_pos = current_pos;
}


VariableNodePtr TGLparser::parse_variable_type(
    const int start_line, 
    const int start_pos, 
    int& next_pos)
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
    int prev_pos = current_pos;
    current_pos = parse_next_token(next_token, cline, current_pos);
    if (next_token == "[")  // this has to be a tensor
    {
        vtype = VariableType::TENSOR;
    }
    else  // has to be a scalar
    {
        vtype = VariableType::SCALAR;
        current_pos = prev_pos;  // restore position
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

        var = std::make_shared<TensorNode>(vtype, dtype, "", shape);
    }

    // if scalar (return the type)
    if (vtype == VariableType::SCALAR)
    {
        var = std::make_shared<ScalarNode>(vtype, dtype, "");
    }
    
    next_pos = current_pos;
    return var;
}

AliasNodePtr TGLparser::parse_alias_node(  // var d = arithmetic_node;
    const std::string& line, 
    const int start_pos, 
    int& next_pos)
{
    int current_pos = start_pos;
    std::string next_token;

    // var keyword is already consumed by the caller
    // reading var name
    current_pos = parse_next_token(next_token, line, current_pos);
    std::string var_name = next_token;
    if (defined_nodes.contains(var_name))
    {
        std::cout << "Alias variable is already defined (duplication not allowed): " << var_name << std::endl;
    }

    // check the equation sign
    current_pos = parse_next_token(next_token, line, current_pos);
    if (next_token != "=")
    {
        std::cout << "Expected an = but instead got: " << next_token << std::endl;
    }

    // process the arithmetic node (function calls also handled by it)
    auto arithm_node = parse_arithmetic_node(line, current_pos, current_pos);

    // build the alias node
    auto node = std::make_shared<AliasNode>(var_name, arithm_node);
    defined_nodes.insert({var_name, node});

    // return
    next_pos = current_pos;
    return node;
}

ReturnNodePtr TGLparser::parse_return_node( 
    const std::string& line, 
    const int start_pos, 
    int& next_pos)
{
    int current_pos = start_pos;

    // return keyword is already consumed by the caller
    // process the arithmetic node (function calls also handled by it)
    auto arithm_node = parse_arithmetic_node(line, current_pos, current_pos);

    // build return node
    auto node = std::make_shared<ReturnNode>(arithm_node);

    // return
    next_pos = current_pos;
    return node;
}

AssignmentNodePtr TGLparser::parse_assignment_node(
    const std::string& var_name, 
    const std::string& line, 
    const int start_pos, 
    int& next_pos)
{
    int current_pos = start_pos;
    std::string next_token;

    // var name is already consumed by the caller
    // = sign is also consumed by the caller
    
    // getting node for var name
    if (!defined_nodes.contains(var_name))
    {
        std::cout << "Assigning to undefined variable: " << var_name << std::endl;
    }

    auto var_node = defined_nodes.at(var_name);

    // process the arithmetic node (function calls also handled by it)
    auto arithm_node = parse_arithmetic_node(line, current_pos, current_pos);

    // build the alias node
    auto node = std::make_shared<AssignmentNode>(var_node, arithm_node);

    // return
    next_pos = current_pos;
    return node;
}

ASTNodePtr TGLparser::parse_kernel_call_node(
    const std::string& kernel_name, 
    const std::string& line, 
    const int start_pos, 
    int& next_pos)
{
    int current_pos = start_pos;
    std::string next_token;

    ASTNodePtr node = nullptr;

    // kernel_name is already consumed by the caller
    // '(' paranthesis is also consumed by the caller
    
    // getting node for var name
    if (defined_nodes.contains(kernel_name))
    {
        auto kernel_node = std::dynamic_pointer_cast<KernelNode>(defined_nodes.at(kernel_name));

        if (!kernel_node)
        {
            std::cout << "Expected a kernel node " << kernel_name << std::endl;
            exit(1);
        }

        // process the arguments
        std::vector<VariableNodePtr> arguments;
        while (next_token != ")")
        {
            current_pos = parse_next_token(next_token, line, current_pos);
            if (next_token != ")" && next_token != ",")
            {
                std::string var_name = next_token;

                if (!defined_nodes.contains(var_name))
                {
                    std::cout << "Undefined variable in call arguments: " << var_name << std::endl;
                    exit(1);
                }

                auto node = defined_nodes.at(var_name);
                arguments.push_back(std::static_pointer_cast<VariableNode>(node));
            }
        }

        // build the alias node
        node = std::make_shared<KernelCallNode>(kernel_node, arguments);
    }
    else if (builtin_kernel_names.contains(kernel_name))  // check for builtin functions
    {
        current_pos = parse_next_token(next_token, line, current_pos);
        std::string var_name = next_token;

        if (!defined_nodes.contains(var_name))
        {
            std::cout << "Undefined variable in call arguments: " << var_name << std::endl;
            exit(1);
        }

        auto var_node = defined_nodes.at(var_name);

        if (kernel_name == "sqrt")
        {
            node = SqrtNode::create_sqrt_node(var_node);
        }
        else if (kernel_name == "exp2")
        {
            node = Exp2Node::create_exp2_node(var_node);
        }
        else if (kernel_name == "log2")
        {
            node = Log2Node::create_log2_node(var_node);
        }
        else
        {
            std::cout << "Undefined variable in call arguments: " << var_name << std::endl;
            exit(1);
        }

        // consume the ')' paranthesis
        current_pos = parse_next_token(next_token, line, current_pos);
    }
    else
    {
        std::cout << "Undefined function can not be called: " << kernel_name << std::endl;
        exit(1);
    }

    // return
    next_pos = current_pos;
    return node;
}

ASTNodePtr TGLparser::parse_arithmetic_node( 
    const std::string& line, 
    const int start_pos, 
    int& next_pos)
{
    // 3 types of arithmetic operands
    // - function call with immediate return
    // - variable
    // - another complex arithm expression (paranthesis signals it)
    
    int current_pos = start_pos;
    std::string next_token;

    // helper structures
    std::vector<char> operators;
    std::vector<ASTNodePtr> ast_nodes;
    
    current_pos = parse_next_token(next_token, line, current_pos);

    while (next_token != ";" && next_token != ")" && next_token != "")
    {
        // select arithmetc operand type
        if (next_token == "(")  // complex arithmetic expression
        {
            auto sub_expression = parse_arithmetic_node(line, current_pos, current_pos);
            ast_nodes.push_back(sub_expression);
        }
        else if (arithmetic_chars.contains(next_token[0]))  // arithmetic operator sign, e.g. +
        {
            operators.push_back(next_token[0]);
        }
        else  // variable or function call
        {
            std::string expr_name = next_token;

            int next_pos = parse_next_token(next_token, line, current_pos);

            if (next_token == "(")  // has to be a function
            {
                auto node = parse_kernel_call_node(expr_name, line, next_pos, current_pos);
                ast_nodes.push_back(node);
            }
            else  // has to be a variable (or an alias)
            {
                auto node = defined_nodes.at(expr_name);
                ast_nodes.push_back(node);
            }
        }

        current_pos = parse_next_token(next_token, line, current_pos);
    }
    
    // process the arithmetic expressions
    ASTNodePtr node = nullptr;
    while (operators.size() > 0)
    {
        // find the operator with highest precedence (first occurence)
        int best_op_idx = 0;
        int highest_prec = arithmetic_precedences.at(operators[best_op_idx]);
        for (int idx = 1; idx < operators.size(); ++idx)
        {
            int tentative_prec = arithmetic_precedences.at(operators[idx]);
            if (highest_prec < tentative_prec)
            {
                best_op_idx = idx;
                highest_prec = tentative_prec;
            }
        }

        char best_op = operators[best_op_idx];

        // find the left and right arguments
        auto lhs = ast_nodes[best_op_idx];
        auto rhs = ast_nodes[best_op_idx + 1];

        // build the right binary operator
        ASTNodePtr subnode = nullptr;
        if (best_op == '*')
        {
            subnode = MulNode::create_mul_node(lhs, rhs);
        }
        else if (best_op == '/')
        {
            subnode = DivNode::create_div_node(lhs, rhs);
        }
        else if (best_op == '+')
        {
            subnode = AddNode::create_add_node(lhs, rhs);
        }
        else if (best_op == '-')
        {
            subnode = SubNode::create_sub_node(lhs, rhs);
        }

        // change the helper data
        operators.erase(operators.begin() + best_op_idx);
        ast_nodes.erase(ast_nodes.begin() + best_op_idx + 1);
        ast_nodes[best_op_idx] = subnode;
    }
    
    if (ast_nodes.size() > 0)  // otherwise it is a void return
        node = ast_nodes[0];   // last remaining node is the root

    // return
    next_pos = current_pos;
    return node;
}
