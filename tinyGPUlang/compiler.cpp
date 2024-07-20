#include <iostream>
#include "parser.hpp"
#include "codegen.hpp"

int main(int argc, char** argv)
{
    std::cout << "TinyGPUlang compiler \n";

    std::string file_name = "call_vec";

    std::string path_tgl = "C:\\Data\\AI\\works\\tinyGPUlang\\artifacts\\" + file_name + ".tgl";
    TGLparser parser(path_tgl);
    auto kernels = parser.get_all_kernels();

    auto printer = std::make_shared<ASTPrinter>();
    for (auto kernel : kernels)
    {
        kernel->accept(*printer);
    }
    printer->save_into_file("C:\\Data\\AI\\works\\tinyGPUlang\\artifacts\\" + file_name + ".ast");

    PTXGenerator ptx_generator;
    ptx_generator.build_ir_from_kernel(kernels[0]);
    ptx_generator.build_ir_from_kernel(kernels[1]);
    ptx_generator.generate_ptx("C:\\Data\\AI\\works\\tinyGPUlang\\artifacts\\" + file_name + ".ptx");

    return 0;
}
