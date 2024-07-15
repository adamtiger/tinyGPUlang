#include "core.hpp"

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