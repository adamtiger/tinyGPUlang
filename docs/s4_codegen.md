# Code generator for NVPTX backend

This is the most relevant part of the tutorial.
The code generator transforms the AST into LLVM IR (with nvidia specific instructions) then generates
ptx code from the IR.

## Main differences compared to CPU backends

* gpus have different type of memories explicitly available for programmers (e.g. shared memory, constant memory etc.): these has to be specified in llvm ir with the help of address spaces
* there are global and device kernels: differentiated with the help of annotation
* output will be a ptx file: can be handled as an assembly, when outputting the file with target machine
* we have to access the nvptx intrinsics to use gpu specific builtin functions and features

### Memory type selection

To ensure the pointer is for the right type of memory address (global, shared etc.) the address space should be set properly.
For example a float32 pointer to a global memory address can be created as:

```
llvm::Type* glob_f32 = llvm::PointerType::get(llvm::Type::getFloatTy(*ctx), 1U);  // from codegen.cpp
```

The address space can be set in the second argument of PointerType::get. 1 refers to global memory.

### Specifying a kernel as global

This requires to set the nvvm.annotations metadata object properly:

```
std::vector<llvm::Metadata*> metadata_fields =
{
    llvm::ValueAsMetadata::get(kernel_llvm_fn),  // kernel_llvm_fn is a llvm::Function* 
    llvm::MDString::get(*ctx, "kernel"),         // kernel string in metadata (see below)
    llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), 1))
};

llvm::NamedMDNode* nnvm_meta_node = lmod->getOrInsertNamedMetadata("nvvm.annotations");
nnvm_meta_node->addOperand(llvm::MDNode::get(*ctx, metadata_fields));
```

This will create the following metadata in IR:

```
!nvvm.annotations = !{!1}
!1 = !{ptr @glob_kernel, !"kernel", i32 1}
```

Here glob_kernel refers to the kernel in kernel_llvm_fn. 
For device functions, no annotation is required because each function is a device function by default.

### Producing ptx files from the IR

The code generation requires the instantiation of the right target machine.
The following target triple is needed:
```
auto target_triple = "nvptx64-nvidia-cuda";
```

Then create a target with target lookup, after that target machine can be also created:
```
std::string Error;
auto target = llvm::TargetRegistry::lookupTarget(target_triple, Error);

auto CPU = "sm_80"; 
auto nv_target_machine = target->createTargetMachine(target_triple, CPU, ...);
```
The CPU parameter can set the target compute capability for ptx.

The data layout can be simply created from the target machine:
```
compiler_state->gmodule->setDataLayout(nv_target_machine->createDataLayout());
```

Finally, in order to print ptx, the file type should be *assembly*:
```
llvm::legacy::PassManager pass;
auto FileType = llvm::CodeGenFileType::AssemblyFile;
nv_target_machine->addPassesToEmitFile(pass, dest, nullptr, FileType);
```

### GPU specific intrinsics

Several basic operation can be used the same way as for any other cpu device.
For instance, to add two numbers:

```
auto* ret = irb->CreateFAdd(lhs_val, rhs_val);
```

For calculating the address at an index and loading the element is also done with GEP:
```
llvm::Value* ptr_element = irb->CreateGEP(llvm::Type::getFloatTy(*ctx), ptr, idx, "ptr");
```

However there are instrinsics which requires special care. First the following header needs to be included:
```
#include "llvm/IR/IntrinsicsNVPTX.h"
```

Then, using sqrt is possible in the following way:
```
auto* ret = irb->CreateIntrinsic(x_val->getType(), llvm::Intrinsic::nvvm_sqrt_f, x_val);
```

The list of the *available intrinsics can be seen in the IntrinsicsNVPTX.h header*.

We can also access important data about the thread id, cluster id etc.:
```
tid = irb->CreateIntrinsic(
    llvm::Type::getInt32Ty(*ctx), 
    llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, 
    {}
);
```
The intrinsic has no input value, this function just queries the thread index.

For more examples, see the codegen.cpp file in the tutorial.

## Next

[Short overview of the parser](s5_parser.md)
