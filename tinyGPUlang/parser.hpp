#pragma once

#include "ast.hpp"
#include "core.hpp"

class TGLparser
{
public:
     
    explicit TGLparser(const std::string& path_to_tgl);
    std::vector<KernelNodePtr> get_all_kernels() const;
    KernelNodePtr get_global_kernel(const std::string& kernel_name) const;

protected:
    std::vector<std::string> all_lines;  // all of the lines from the source file

    static std::unordered_set<char> ignored_chars;
    static std::unordered_set<char> bracket_chars;
    static std::unordered_set<char> comma_chars;
    static std::unordered_set<char> arithmetic_chars;
    static std::unordered_set<std::string> builtin_kernel_names;
    static std::unordered_map<char, int> arithmetic_precedences;
    std::unordered_map<std::string, KernelNodePtr> defined_global_kernels;
    std::unordered_map<std::string, KernelNodePtr> defined_device_kernels;
    std::unordered_map<std::string, ASTNodePtr> defined_nodes;
    std::vector<KernelNodePtr> defined_kernels;  // kernels defined in order

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

    /**
     * Parse the next function from the source file.
     * @param start_line the first line to look for the next kernel def
     * @param start_pos first position within the line
     * @param next_line the line where the search ended (next search should start here)
     * @param next_pos the position where the search ended (next search should start here)
     */
    void parse_next_kernel(const int start_line, const int start_pos, int& next_line, int& next_pos);

    /**
     * Reads the kernel header (gives the definition of the kernel).
     * The kernel header should be in a single line.
     */ 
    KernelNodePtr parse_kernel_header(const int start_line, const int start_pos, int& next_pos);
    
    /**
     * Reads the body of the kernel, each line (expression) will
     * be an ASTNode. The kernel body is among curly brackets.
     */
    void parse_kernel_body(KernelNodePtr kernel, const int start_line, const int start_pos, int& next_line, int& next_pos);

    /**
     * Checks for missing paranthesis in an expression, line.
     * Stops the process if error found.
     */
    void check_paranthesis_in_line(const int start_line);
    
    /**
     * Reads the type of a variable. Type definition can happen in
     * a kernel header. The name will remain empty.
     */
    VariableNodePtr parse_variable_type(const int start_line, const int start_pos, int& next_pos);

    /**
     * Reads a constant value in the code (a value given inside the code).
     * Only floats are supported.
     */
    ConstantNodePtr parse_constant_scalar(
        const std::string& value_as_string, 
        const int start_line, 
        const int start_pos);
    
    /**
     * Reads the var t = arithm expr.; like code pieces.
     * The alias will be for the ASTNode built from the arithmetic expression.
     * @param start_pos shows the position at the beginning of the alias name.
     */
    AliasNodePtr parse_alias_node( 
        const int start_line, 
        const int start_pos, 
        int& next_pos);

    /**
     * Reads the return arithm expr.; like code pieces.
     * @param start_pos shows the position at the beginning of the variable name.
     */
    ReturnNodePtr parse_return_node( 
        const int start_line, 
        const int start_pos, 
        int& next_pos);

    /**
     * Reads the d = arithm expr.; like code pieces.
     * @param start_pos shows the position at the beginning of the arithmetic ops.
     */
    AssignmentNodePtr parse_assignment_node(
        const std::string& var_name, 
        const int start_line, 
        const int start_pos, 
        int& next_pos);

    /**
     * Reads a function call. The call is to a device function
     * defined in the same tgl file, somewhere earlier.
     * Builtin function calls are handled as arithmetic expressions.
     * @param start_pos shows the position at the beginning of the args.
     */
    ASTNodePtr parse_kernel_call_node(
        const std::string& kernel_name, 
        const int start_line, 
        const int start_pos, 
        int& next_pos);

    /**
     * Can be a function call (newly defined are delegated), and regular
     * mathematical expressions. (E.g.: a + (b * c) - abs(d) + ((a / b) + d + 2.0); )
     * @param start_pos shows the position at the beginning of the expression.
     */
    ASTNodePtr parse_arithmetic_node( 
        const int start_line, 
        const int start_pos, 
        int& next_pos);
};
