#pragma once

// includes
#include <memory>
#include <mutex>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#include <vector>
#include <unordered_set>
#include <unordered_map>


/*
    Target types.
*/
enum class Target
{
    NVIDIA_GPU
};


/*
    Thread-safe uuid generator.
    This is global. Creates an
    id unique overall the system
    and the execution time.

    Typically used for get the ID
    for the operators and variables.
*/
class GlobalUUIDGenerator
{
public:
    static int generate_uuid();
    static void reset();

private:
    static int next_id;
    static std::mutex lock_obj;
};

/**
 * Checks if a string can be a float.
 */
bool is_float_number(const std::string& value_as_str);

/**
    Exchanges the path extension to another one.
    It starts with string and returns a string.
    @param path_to_file file path to be modified
    @param new_extension name of new extension (e.g. tgl)
    @return a string with replaced extension
*/
std::string replace_extension(
    const std::string& path_to_file,
    const std::string& new_extension
);

/**
    Exchanges the folder path the file name will be the same
    @param path_to_file file path to be modified
    @param new_folder_path name of new folder to save with same file name
    @return a string with replaced extension
*/
std::string replace_folder_path(
    const std::string& path_to_file,
    const std::string& new_folder_path
);

/**
    Writes error message to the screen and halts the
    program execution.
*/
void emit_error(const std::string& error_msg, const int line=-1, const int pos=-1);
