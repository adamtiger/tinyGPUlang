#pragma once

#include "ast.hpp"
#include "core.hpp"

/*
    LLVM requires a large number of
      headers to be included. This
      header collects them.
*/

#include "llvm/IR/Type.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Mangler.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"

#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"


// helpers
struct LLVMState
{
    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::Module> gmodule;
    std::unique_ptr<llvm::IRBuilder<>> ir_builder;
};


/**
 * Generates the ptx code from
 * the LLVM IR.
 */
class PTXGenerator
{
public:
    explicit PTXGenerator();
    
    void build_ir_from_kernel(const KernelNodePtr kernel);
    
    void generate_ptx(
        const std::string& ptx_file, 
        const std::string& sm_xx, 
        const bool save_temps);

private:
    std::shared_ptr<LLVMState> compiler_state;
    std::unordered_map<std::string, llvm::Function*> defined_functions;
};


/**
 * Generates LLVM IR with ptx instrinsics
 * from the AST nodes.
 */
class NVIRBuilder : public ASTVisitor
{
public:
    explicit NVIRBuilder(
        std::shared_ptr<LLVMState> compiler_state,
        const std::unordered_map<std::string, llvm::Function*>& defined_functions,
        std::unordered_map<int, llvm::Value*>& values
    );

    virtual void apply(KernelNode& node);
    virtual void apply(KernelCallNode& node);
    
    virtual void apply(ConstantNode& node);
    virtual void apply(ScalarNode& node);
    virtual void apply(TensorNode& node);

    virtual void apply(AddNode& node);
    virtual void apply(SubNode& node);
    virtual void apply(MulNode& node);
    virtual void apply(DivNode& node);

    virtual void apply(AbsNode& node);
    virtual void apply(SqrtNode& node);
    virtual void apply(Log2Node& node);
    virtual void apply(Exp2Node& node);

    virtual void apply(AssignmentNode& node);
    virtual void apply(AliasNode& node);
    virtual void apply(ReturnNode& node);

private:
    std::shared_ptr<LLVMState> compiler_state;
    const std::unordered_map<std::string, llvm::Function*>& defined_functions;
    std::unordered_map<int, llvm::Value*>& values;

    llvm::Value* tid;

    llvm::Value* calc_ptr_from_offset(
        const llvm::Type* ltype, 
        llvm::Value* ptr,
        llvm::Value* idx);
};
