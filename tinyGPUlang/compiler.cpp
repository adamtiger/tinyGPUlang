#include "core.hpp"
#include "parser.hpp"
#include "codegen.hpp"

static void print_version_info();

static void print_help_info();

static void compile_source_file(
    const std::string& tgl_path, 
    const Target target, 
    const bool save_temps,
    const std::string& out_folder_path,
    const std::string& sm_xx);

int main(int argc, char** argv)
{
    // first argument is the name of the application
    // process the rest
    
    // second argument can contain version or helper or something else
    // check for version and helper
    if (argc <= 1)
    {
        emit_error("No option is supported. See --help for details!");
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
        std::string out_folder_path = "";
        std::string sm_xx = "";

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
                    std::stringstream ss;
                    ss << "Expected a source file with tgl extension. Instead got ";
                    ss << extension;
                    ss << ". See --help for details!";
                    emit_error(ss.str());
                }
            }
            else if (arg_str == "--target")
            {
                std::string trg_str = argv[arg_ix + 1];
                
                if (trg_str == "nvidia")
                {
                    target = Target::NVIDIA_GPU;
                }
                else
                {
                    std::stringstream ss;
                    ss << "Unknown target name ";
                    ss << trg_str;
                    ss << ". See --help for details!";
                    emit_error(ss.str());
                }

                arg_ix += 2;
            }
            else if (arg_str == "--save-temps")
            {
                save_temps = true;
                arg_ix += 1;
            }
            else if (arg_str == "--out")
            {
                out_folder_path = argv[arg_ix + 1];
                arg_ix += 2;
            }
            else if (arg_str == "--sm")
            {
                sm_xx = "sm_" + std::string(argv[arg_ix + 1]);
                arg_ix += 2;
            }
            else
            {
                std::stringstream ss;
                ss << "Unknown command line option ";
                ss << arg_str;
                ss << ". See --help for details!";
                emit_error(ss.str());
            }
        }

        if (path_to_tgl != "")
        {
            compile_source_file(path_to_tgl, target, save_temps, out_folder_path, sm_xx);
        }
        else
        {
            std::stringstream ss;
            ss << "No path to a source is supported. ";
            ss << "Error during compilation. ";
            ss << ". See --help for details!";
            emit_error(ss.str());
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
    ss << "    --target      : currently only nvidia is supported (defaults to 'nvidia') \n";
    ss << "    --save-temps  : if present, saves the ll and ast files (defaults to false) \n";
    ss << "    --out         : if present, it has to be a folder path for saving files \n";
    ss << "    --sm          : if present, it will set the .target directive in the output ptx (default is given by llvm, regularly sm_30) \n";
    ss << "\n";

    std::cout << ss.str();
}

void compile_source_file(
    const std::string& tgl_path, 
    const Target target, 
    const bool save_temps,
    const std::string& out_folder_path,
    const std::string& sm_xx)
{
    std::cout << "TinyGPUlang compiler \n";
    
    std::string temp_path = tgl_path;
    if (out_folder_path != "")
    {
        temp_path = replace_folder_path(tgl_path, out_folder_path);
    }

    std::string ast_file_path = replace_extension(temp_path, "ast");;
    std::string ptx_file_path = replace_extension(temp_path, "ptx");

    TGLparser parser(tgl_path);
    auto kernels = parser.get_all_kernels();
    
    if (save_temps)
    {
        auto printer = std::make_shared<ASTPrinter>();
        for (auto kernel : kernels)
        {
            kernel->accept(*printer);
        }
        printer->save_into_file(ast_file_path);
    }
    
    PTXGenerator ptx_generator;
    for (auto kernel : kernels)
    {
        ptx_generator.build_ir_from_kernel(kernel);
    }
    ptx_generator.generate_ptx(ptx_file_path, sm_xx, save_temps);
}
