# tinyGPUlang

Tutorial on building a gpu compiler backend in LLVM

## Goals

The goal of this tutorial is to show a *simple example* on how to generate ptx from the llvm ir and how to write the IR itself to access cuda features.

For the sake of demonstration a language frontend is also provided. The main idea of the language is to support pointwise (aka elementwise) operations with gpu acceleration.

If you are just curios about the code generation backend, you can jump directly to [The code generator for NVPTX backend](docs/s4_codegen.md) part.

## What is inside the repo?

- tinyGPUlang: the compiler, creates ptx from tgl (the example language file)
- test: a cuda driver api based test for the generated ptx
- examples: example tgl files
- docs: documentation for the tutorial 

## Tutorial content

1. [Overview](docs/s1_overview.md)
2. [The TGL language](docs/s2_tgl_language.md)
3. [Abstract Syntax Tree](docs/s3_ast.md)
4. [The code generator for NVPTX backend](docs/s4_codegen.md)
5. [Short overview of the parser](docs/s5_parser.md)
6. [How to build the project?](docs/s6_build_proj.md)

## References

- [LLVM documentation for NVPTX backend](https://llvm.org/docs/NVPTXUsage.html)
- [annotation for global kernel](https://stackoverflow.com/questions/19743861/what-is-llvm-metadata)
- [TVM NVPTX codegen](https://github.com/apache/tvm/blob/main/src/target/llvm/codegen_nvptx.cc)

## TODO

- [ ] test on linux
- [ ] build system cleaning
- [ ] document (codegen, build)
