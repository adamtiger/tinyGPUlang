#include <iostream>
#include "parser.hpp"

int main(int argc, char** argv)
{

    std::cout << "TinyGPUlang compiler \n";

    std::string path_tgl = "C:\\Data\\AI\\works\\tinyGPUlang\\artifacts\\add_tensors.tgl";
    TGLparser parser(path_tgl);
    

    return 0;
}
