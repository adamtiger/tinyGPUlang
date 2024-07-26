# Building TinyGPUlang

Two platforms were tested for building the compiler:
* windows (11)
* linux (ubuntu24)

The compiler has the following dependencies:
* llvm (with header and libs), tested: llvm-16
* cuda, tested: v12.0

## Linux

### Building LLVM

You may skip this part if you already have a built llvm on your machine.

The following steps were performed on an aws ec2 instance (with ubuntu24), started from scratch.
Preparation includes installing the following packages:
```
sudo apt install clang

sudo apt install cmake

sudo apt install ninja-build
```

Download the llvm source code from the releases. In this tutorial, [this version (download link)](https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.7.tar.gz) was used.

```
wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.7.tar.gz

tar -xf llvmorg-18.1.7.tar.gz
```

Then navigate to the llvm subfolder and create the build folder:
```
cd llvm-project-llvmorg-18.1.7/llvm

mkdir build

cd build
```

Generate ninja build files with cmake:
```
cmake -G Ninja -DLLVM_HOST_TRIPLE=x86_64 -DCMAKE_BUILD_TYPE=Release  -S ..
```

Then build the project, this can take a while (even an hour) to finish:
```
cmake --build .
```

Install the libraries:
```
cmake --install .
```

