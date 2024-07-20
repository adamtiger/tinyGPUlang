#pragma once

#include <mutex>
#include <string>

/*
    Target types.
*/
enum class Target
{
    NVIDIA_GPU,
    AMD_GPU
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