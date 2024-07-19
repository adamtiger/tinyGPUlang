#include <iostream>
#include "parser.hpp"
#include "codegen.hpp"

int main(int argc, char** argv)
{
    std::cout << "TinyGPUlang compiler \n";

    std::string path_tgl = "C:\\Data\\AI\\works\\tinyGPUlang\\artifacts\\add_vec.tgl";
    TGLparser parser(path_tgl);
    auto kernels = parser.get_all_global_kernel();

    auto printer = std::make_shared<ASTPrinter>();
    for (auto kernel : kernels)
    {
        kernel->accept(*printer);
    }
    printer->save_into_file("ast.txt");


    PTXGenerator ptx_generator;
    ptx_generator.build_ir_from_kernel(kernels[0]);
    ptx_generator.generate_ptx("add_vec.ptx");

    return 0;
}
