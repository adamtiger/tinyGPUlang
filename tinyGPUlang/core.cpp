#include "core.hpp"
#include <filesystem>
#include <iostream>

int GlobalUUIDGenerator::generate_uuid()
{
    int id = 0;
    lock_obj.lock();
    id = next_id++;
    lock_obj.unlock();
    return id;
}

void GlobalUUIDGenerator::reset()
{
    next_id = 0;
}

// statics

int GlobalUUIDGenerator::next_id = 0;
std::mutex GlobalUUIDGenerator::lock_obj;


std::string replace_extension(
    const std::string& path_to_file,
    const std::string& new_extension)
{
    auto rpl_file_path = std::filesystem::path(path_to_file).replace_extension(new_extension);
    return rpl_file_path.string();
}


std::string replace_folder_path(
    const std::string& path_to_file,
    const std::string& new_folder_path)
{
    auto new_folder = std::filesystem::path(new_folder_path);
    auto orig_file_name = std::filesystem::path(path_to_file).filename();
    auto new_file_path = new_folder.append(orig_file_name.string());
    return new_file_path.string();
}


void emit_error(const std::string& error_msg, const int line, const int pos)
{
    if (line > -1 && pos > -1)
    {
        std::cerr << "Line[" << line + 1 << "] " << "Col[" << pos + 1 << "]: ";
    } 

    std::cerr << error_msg << "\n"; 
    exit(1);
}