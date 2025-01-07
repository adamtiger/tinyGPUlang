# finding the source files
set(TEST_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(TEST_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})

set (SOURCES
    ${TEST_ROOT}/kernel_executor.cpp
	${TEST_ROOT}/main.cpp
)

set (HEADERS
	${TEST_ROOT}/kernel_executor.hpp
)

# compiler settings
if (UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
endif()

if (UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIC")
endif()

add_executable(test ${SOURCES} ${HEADERS})
target_include_directories(test PRIVATE ${TEST_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
# we need to link the cuda device library, assume they are in the same folder
# CUDA_LIBRARIES: path to cudart library
get_filename_component(CUDA_LIB_PATH "${CUDA_LIBRARIES}" DIRECTORY)
if (UNIX) 
    target_link_libraries(test PRIVATE cuda cudart)
elseif (WIN32) 
    target_link_libraries(test PRIVATE ${CUDA_LIB_PATH}/cuda.lib)
endif()


if (UNIX)
    target_compile_options(test PRIVATE -fPIC -fopenmp)
endif()

if (MSVC)
    target_link_options(test PUBLIC "$<$<CONFIG:Release>:/NODEFAULTLIB:libcmt.lib>") 
    target_compile_options(test PRIVATE "/fp:fast" "/FA")
    set_target_properties(test PROPERTIES LINK_FLAGS "/PROFILE")
endif()

if (WIN32)
	#required to move the dependecies to the exe on windows
	# link: https://cmake.org/cmake/help/git-stage/manual/cmake-generator-expressions.7.html#genex:TARGET_RUNTIME_DLLS
    add_custom_command(TARGET test POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:test> $<TARGET_RUNTIME_DLLS:test>
       COMMAND_EXPAND_LISTS
    )
endif()
