# Overview

The tutorial gives a demonstration on how to generate ptx file from llvm ir.

## Compiler architecture

The compiler has 3 components:
- parser: reads the tgl file and produces an internal abstract syntax tree
- AST: the implementation of the abstract syntax tree
- code generator: produces LLVM IR from the AST objects, it can turn the LLVM IR into ptx

All components are discussed at later sections of the tutorial.

## How to use the compiler?

The compiler is an executable (named tglc) with command-line options.

### Examples for usage (e.g on windows):
```
tglc.exe --src tgl_code_file_path.tgl 
```
Will create ptx file with the same name and in the same folder but with ptx extension.

```
tglc.exe --src tgl_code_file_path.tgl --save-temps
```
Beside ptx generation, it will output the .ll with llvm ir and a .ast file, with the printed AST. This helps to understand better what is happening at each step.

```
tglc.exe --src tgl_code_file_path.tgl --out output_folder_path
```
Output folder can be specified to save the created files.

```
tglc.exe --src tgl_code_file_path.tgl --sm 80
```
Sets the target architecture to sm_80.

The usage on linux is very similar (from build/tinyGPUlang):
```
./tglc --version
```

## Test the ptx

The test folder contains an example execution of the ptx with cuda driver api.
The kernel name and ptx name should be changed accordingly.
There is no command-line options for test.

## Next

[The TGL language](s2_tgl_language.md)
