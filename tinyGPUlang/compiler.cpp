#include <iostream>
#include "parser.hpp"

int main(int argc, char** argv)
{
    std::cout << "TinyGPUlang compiler \n";

    std::string path_tgl = "C:\\Data\\AI\\works\\tinyGPUlang\\artifacts\\complex.tgl";
    TGLparser parser(path_tgl);
    auto kernels = parser.get_all_global_kernel();

    auto printer = std::make_shared<ASTPrinter>();
    for (auto kernel : kernels)
    {
        kernel->accept(*printer);
    }
    printer->save_into_file("ast.txt");

    return 0;
}
