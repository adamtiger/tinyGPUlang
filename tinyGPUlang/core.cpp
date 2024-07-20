#include "core.hpp"
#include <filesystem>

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
