#pragma once

#include <mutex>

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
