# Building TinyGPUlang

Two platforms were tested for building the compiler:
* windows (11)
* linux (ubuntu24)

The compiler has the following dependencies:
* llvm (with header and libs), tested: llvm-18.1.7
* cuda, tested: v12.0

Installing cuda is out of the scope of this tutorial.

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

Then build the project, this can take a while (more than an hour) to finish:
```
cmake --build .
```

Install the libraries (sudo is needed to have permission for copy):
```
sudo cmake --install .
```

### Building TinyGPUlang

The mechanism is very similar. After the build directory is created:

```
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release  -S ..

cmake --build .
```

## Windows

### Building LLVM

It is assumed the msvc and its accompanying build tools for c++ is already installed.
It is recommended to use msvc on windows to avoid problems with accessing windows system libraries by clang.

Download the zip file from [here (download link)](https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.7.zip)

Unzip and create a build directory in the llvm folder.
Start a command prompt in the build directory.

Generate visual studio project files with cmake:
```
cmake -G "Visual Studio 17 2022" -DLLVM_HOST_TRIPLE=x86_64  -S ..
```

Then build the project, this can take a while (more than an hour) to finish:
```
cmake --build . --config Release
```

Install the libraries (this requires administrator mode):
```
cmake --install .
```

### Building TinyGPUlang

The mechanism is very similar. After the build directory is created:

```
cmake -G "Visual Studio 17 2022" -S ..

cmake --build . --config Release
```
