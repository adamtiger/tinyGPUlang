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
