#include <iostream>
#include <filesystem>
#include "core.hpp"
#include "parser.hpp"
#include "codegen.hpp"

static void print_version_info();

static void print_help_info();

static void compile_source_file(
    const std::string& tgl_path, 
    const Target target, 
    const bool save_temps);

int main(int argc, char** argv)
{
    // first argument is the name of the application
    // process the rest
    
    // second argument can contain version or helper or something else
    // check for version and helper
    if (argc <= 1)
    {
        std::cout << "No option is supported. See --help for details! \n";
        exit(1);
    }

    std::string arg_str = argv[1];
    if (arg_str == "--version")
    {
        print_version_info();
    }
    else if (arg_str == "--help")
    {
        print_help_info();
    }
    else
    {
        std::string path_to_tgl = "";
        Target target = Target::NVIDIA_GPU;
        bool save_temps = false;

        int arg_ix = 1;
        while (arg_ix < argc)
        {
            arg_str = argv[arg_ix];

            if (arg_str == "--src")
            {
                path_to_tgl = argv[arg_ix + 1];
                arg_ix += 2;

                std::string extension = std::filesystem::path(path_to_tgl).extension().string();
                if (extension != ".tgl")
                {
                    std::cout << "Expected a source file with tgl extension. Instead got " << extension << " See --help for details! \n";
                    exit(1);
                }
            }
            else if (arg_str == "--target")
            {
                std::string trg_str = argv[arg_ix + 1];
                
                if (trg_str == "nvidia")
                {
                    target = Target::NVIDIA_GPU;
                }
                else if (trg_str == "amd")
                {
                    target = Target::AMD_GPU;
                }
                else
                {
                    std::cout << "Unknown target name " << trg_str << " . See --help for details! \n";
                    exit(1);
                }

                arg_ix += 2;
            }
            else if (arg_str == "--save-temps")
            {
                save_temps = true;
                arg_ix += 1;
            }
            else
            {
                std::cout << "Unknown command line option " << arg_str << " . See --help for details! \n";
                exit(1);
            }
        }

        if (path_to_tgl != "")
        {
            compile_source_file(path_to_tgl, target, save_temps);
        }
        else
        {
            std::cout << "No path to a source is supported. Error during compilation. See --help for details! \n";
        }
    }

    return 0;
}

void print_version_info()
{
    std::cout << "Tiny GPU language compiler (TGLC) - v1.0.0 \n"; 
}

void print_help_info()
{
    std::stringstream ss;
    ss << "\n";
    ss << "TGLC compiler, manual \n";
    ss << "The following command line options are available: \n";
    ss << "    --version     : prints the version of the tglc compiler \n";
    ss << "    --help        : prints the user manual \n";
    ss << "\n";
    ss << "    --src         : path to the tgl file (only one file at once) \n";
    ss << "    --target      : one of nvidia or amd (defaults to nvidia) \n";
    ss << "    --save-temps  : if present, saves the ll and ast files (defaults to false) \n";
    ss << "\n";

    std::cout << ss.str();
}

void compile_source_file(
    const std::string& tgl_path, 
    const Target target, 
    const bool save_temps)
{
    std::cout << "TinyGPUlang compiler \n";

    TGLparser parser(tgl_path);
    auto kernels = parser.get_all_kernels();
    
    if (save_temps)
    {
        auto printer = std::make_shared<ASTPrinter>();
        for (auto kernel : kernels)
        {
            kernel->accept(*printer);
        }
        printer->save_into_file(replace_extension(tgl_path, "ast"));
    }
    
    PTXGenerator ptx_generator;
    for (auto kernel : kernels)
    {
        ptx_generator.build_ir_from_kernel(kernel);
    }
    ptx_generator.generate_ptx(replace_extension(tgl_path, "ptx"), save_temps);
}
