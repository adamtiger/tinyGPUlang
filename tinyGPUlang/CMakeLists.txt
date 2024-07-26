# LLVM related settings
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

separate_arguments(LLVM_DEFINITIONS)

message(STATUS "LLVM libraries: ${LLVM_AVAILABLE_LIBS}")
message(STATUS "LLVM includes: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM definitions: ${LLVM_DEFINITIONS}")

# finding the source files
set(TGLC_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(TGLC_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})

set (SOURCES
    ${TGLC_ROOT}/compiler.cpp
    ${TGLC_ROOT}/core.cpp
	${TGLC_ROOT}/parser.cpp
	${TGLC_ROOT}/ast.cpp
	${TGLC_ROOT}/codegen.cpp
)

set (HEADERS
	${TGLC_ROOT}/core.hpp
	${TGLC_ROOT}/parser.hpp
	${TGLC_ROOT}/ast.hpp
	${TGLC_ROOT}/codegen.hpp
)

# compiler settings
if (UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIC")
endif()

add_executable(tglc ${SOURCES} ${HEADERS})
target_include_directories(tglc PRIVATE ${TGLC_INCLUDE_DIR} ${LLVM_INCLUDE_DIRS})

if (UNIX)
    target_compile_options(tglc PRIVATE -fPIC)
endif()

if (MSVC)
    target_link_options(tglc PUBLIC "$<$<CONFIG:Release>:/NODEFAULTLIB:libcmt.lib>") 
    target_compile_options(tglc PRIVATE "/fp:fast")
endif()

target_link_libraries(tglc PRIVATE ${LLVM_AVAILABLE_LIBS})

if (WIN32)
	#required to move the dependecies to the exe on windows
	# link: https://cmake.org/cmake/help/git-stage/manual/cmake-generator-expressions.7.html#genex:TARGET_RUNTIME_DLLS
    add_custom_command(TARGET tglc POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:tglc> $<TARGET_RUNTIME_DLLS:tglc>
       COMMAND_EXPAND_LISTS
    )
endif()
