# LLVM related settings
set(LLVM_DIR "C:\\progs\\llvm\\lib\\cmake\\llvm")
find_package(LLVM REQUIRED CONFIG)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})
llvm_map_components_to_libnames(llvm_libs
    Analysis
    Core
    ExecutionEngine
    InstCombine
    Object
    OrcJIT
    RuntimeDyld
    ScalarOpts
    Support
    TransformUtils
    native
    irreader
)

set(LLVM_INSTALL_DIR_MN "C:\\progs\\llvm")
set(LLVM_LIB_DIR_MN ${LLVM_INSTALL_DIR_MN}/lib)
file(GLOB LLVM_LIBRARIES_MN 
	${LLVM_LIB_DIR_MN}/LLVM*${CMAKE_STATIC_LIBRARY_SUFFIX})

# finding the source files
set(TGLC_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(TGLC_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR})

set (SOURCES
    ${TGLC_ROOT}/compiler.cpp
    ${TGLC_ROOT}/core.cpp
	${TGLC_ROOT}/parser.cpp
	${TGLC_ROOT}/ir.cpp
	${TGLC_ROOT}/ptxgen.cpp
)

set (HEADERS
	${TGLC_ROOT}/core.hpp
	${TGLC_ROOT}/parser.hpp
	${TGLC_ROOT}/ir.hpp
	${TGLC_ROOT}/ptxgen.hpp
)

# compiler settings
if (UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
endif()

if (UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIC")
endif()

add_executable(tglc ${SOURCES} ${HEADERS})
target_include_directories(tglc PRIVATE ${TGLC_INCLUDE_DIR})

if (UNIX)
    target_compile_options(tglc PRIVATE -fPIC -fopenmp)
endif()

if (MSVC)
    target_link_options(tglc PUBLIC "$<$<CONFIG:Release>:/NODEFAULTLIB:libcmt.lib>") 
    target_compile_options(tglc PRIVATE "/fp:fast" "/FA")
    set_target_properties(tglc PROPERTIES LINK_FLAGS "/PROFILE")
endif()

target_link_libraries(tglc PRIVATE ${llvm_libs} ${LLVM_LIBRARIES_MN})

if (WIN32)
	#required to move the dependecies to the exe on windows
	# link: https://cmake.org/cmake/help/git-stage/manual/cmake-generator-expressions.7.html#genex:TARGET_RUNTIME_DLLS
    add_custom_command(TARGET tglc POST_BUILD
       COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:tglc> $<TARGET_RUNTIME_DLLS:tglc>
       COMMAND_EXPAND_LISTS
    )
endif()
