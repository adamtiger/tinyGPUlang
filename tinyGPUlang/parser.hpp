#pragma once

#include "ast.hpp"
#include <unordered_set>

class TGLparser
{
public:
     
    explicit TGLparser(const std::string& path_to_tgl);
    std::vector<KernelNodePtr> get_all_global_kernel() const;
    KernelNodePtr get_global_kernel(const std::string& kernel_name) const;

protected:
    std::vector<std::string> all_lines;  // all of the lines from the source file

    static std::unordered_set<char> ignored_chars;
    static std::unordered_set<char> bracket_chars;
    static std::unordered_set<char> comma_chars;
    static std::unordered_set<char> arithmetic_chars;
    static std::unordered_set<std::string> builtin_kernel_names;
    std::unordered_set<std::string> defined_kernel_names;

    /**
     * Reads all the text from the source file.
     * Breaks the text into lines.
     * @param path_to_tgl string representing the full path to the soorce file (*.tgl)
     */
    void read_all_lines_from_tgl(const std::string& path_to_tgl);

    /**
     * Searches and cuts the next token from a line.
     * @param next_token the next token will be in this variable, empty string shows no more token
     * @param line the line to search for the next token
     * @param start_pos the search start at this position, inclusive
     * @return the position where the search ended (next search should start here)
     */
    int parse_next_token(std::string& next_token, const std::string& line, const int start_pos);

};
