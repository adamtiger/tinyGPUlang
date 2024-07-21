#include "codegen.hpp"
#include "core.hpp"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include <fstream>

PTXGenerator::PTXGenerator()
{
    compiler_state = std::make_shared<LLVMState>();

    // state related initialization
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    compiler_state->context = std::make_unique<llvm::LLVMContext>();
    compiler_state->ir_builder = std::make_unique<llvm::IRBuilder<>>(*compiler_state->context);
    compiler_state->gmodule = std::make_unique<llvm::Module>("xgir jit", *compiler_state->context);
}

static llvm::Type* get_llvm_type_of_variable(std::unique_ptr<llvm::LLVMContext>& ctx, const VariableNodePtr var)
{
    llvm::Type *var_type = nullptr;
    if (var->vtype == VariableType::SCALAR)
    {
        if (var->dtype == DataType::FLOAT32)
        {
            var_type = llvm::Type::getFloatTy(*ctx);
        }
        else if (var->dtype == DataType::FLOAT16)
        {
            var_type = llvm::Type::getHalfTy(*ctx);
        }
    }
    else
    {
        if (var->dtype == DataType::FLOAT32)
        {
            var_type = llvm::Type::getFloatPtrTy(*ctx, 1U);
        }
        else if (var->dtype == DataType::FLOAT16)
        {
            var_type = llvm::Type::getHalfPtrTy(*ctx, 1U);
        }
    }
    return var_type;
}

void PTXGenerator::build_ir_from_kernel(const KernelNodePtr kernel)
{
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;
    auto& lmod = compiler_state->gmodule;

    /*
        The function to be defined
    */
    std::string func_name = kernel->name;

    std::vector<llvm::Type*> arg_types;
    for (auto arg : kernel->arguments)
    {
        auto* arg_type = get_llvm_type_of_variable(ctx, arg);
        arg_types.push_back(arg_type);
    }

    llvm::Type* ret_type = nullptr;
    if (kernel->return_value)
    {
        ret_type = get_llvm_type_of_variable(ctx, kernel->return_value);
    }
    else  // void
    {
        ret_type = llvm::Type::getVoidTy(*ctx);
    }

    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ret_type, arg_types, false);
    
    auto* kernel_llvm_fn = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        func_name,
        compiler_state->gmodule.get()
    );

    defined_functions.insert({func_name, kernel_llvm_fn});

    // set up function block
    llvm::BasicBlock* BB = llvm::BasicBlock::Create(*ctx, "entry", kernel_llvm_fn);
    irb->SetInsertPoint(BB);
    
    // insert values
    std::unordered_map<int, llvm::Value*> values;
    for (int ix = 0; ix < kernel_llvm_fn->arg_size(); ++ix)
    {
        values.insert({kernel->arguments[ix]->ast_id, kernel_llvm_fn->getArg(ix)});
    }

    // apply IR blocks of graph operations
    NVIRBuilder builder(compiler_state, defined_functions, values);
    kernel->accept(builder);
    
    if (kernel->scope == KernelScope::GLOBAL)
    {
        // https://stackoverflow.com/questions/19743861/what-is-llvm-metadata
        std::vector<llvm::Metadata*> metadata_fields =
        {
            llvm::ValueAsMetadata::get(kernel_llvm_fn),
            llvm::MDString::get(*ctx, "kernel"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), 1))
        };

        llvm::NamedMDNode* nnvm_meta_node = lmod->getOrInsertNamedMetadata("nvvm.annotations");
        nnvm_meta_node->addOperand(llvm::MDNode::get(*ctx, metadata_fields));
    }

    std::cout << "Kernel was built in IR " << func_name << "\n";
}

void PTXGenerator::generate_ptx(const std::string& ptx_file, const bool save_temps)
{
    // for further ideas: https://github.com/apache/tvm/blob/main/src/target/llvm/codegen_nvptx.cc

    // Initialize the target registry etc.
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    
    if (save_temps)
    {
        auto ll_file_path = replace_extension(ptx_file, "ll");
        std::ofstream ll_file(ll_file_path);
        llvm::raw_os_ostream llvm_ostream(ll_file);
        for (auto& name_func : defined_functions)
        {
            name_func.second->print(llvm_ostream);
        }
    }

    auto TargetTriple = "nvptx64-nvidia-cuda";

    std::string Error;
    auto Target = llvm::TargetRegistry::lookupTarget(TargetTriple, Error);

    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialise the
    // TargetRegistry or we have a bogus target triple.
    if (!Target) {
        llvm::errs() << Error;
        return;
    }

    // this will make possible to use the right intrinsics when possible
    auto CPU = "";
    auto Features = "";

    llvm::TargetOptions opt;
    auto RM = std::optional<llvm::Reloc::Model>();
    
    auto TheTargetMachine = Target->createTargetMachine(
        TargetTriple, CPU, Features, opt, RM);

    //compiler_state->gmodule->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
    compiler_state->gmodule->setDataLayout(TheTargetMachine->createDataLayout());

    std::error_code EC;
    llvm::raw_fd_ostream dest(ptx_file, EC, llvm::sys::fs::OF_None);

    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message();
        return;
    }

    llvm::legacy::PassManager pass;
    auto FileType = llvm::CGFT_AssemblyFile;

    if (TheTargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
        llvm::errs() << "TheTargetMachine can't emit a file of this type";
        return;
    }

    pass.run(*compiler_state->gmodule);
    dest.flush();

    std::cout << "Ptx was generated into " << ptx_file << "\n";
}

// IR builder for NVIDIA gpus

NVIRBuilder::NVIRBuilder(
    std::shared_ptr<LLVMState> compiler_state,
    const std::unordered_map<std::string, llvm::Function*>& defined_functions,
    std::unordered_map<int, llvm::Value*>& values
    ) : compiler_state(compiler_state), 
        defined_functions(defined_functions),
        values(values)
{

}

llvm::Value* NVIRBuilder::calc_ptr_from_offset(
    const llvm::Type* ltype, 
    llvm::Value* ptr,
    llvm::Value* idx)
{
    auto& irb = compiler_state->ir_builder;
    auto& ctx = compiler_state->context;

    llvm::Type* llvm_type = llvm::Type::getFloatTy(*ctx); //(??? ? llvm::Type::getFloatTy(*ctx) : llvm::Type::getHalfTy(*ctx));
    llvm::Value* ptr_element = irb->CreateGEP(llvm_type, ptr, idx, "ptr");
    return ptr_element;
}

void NVIRBuilder::apply(KernelNode& node)
{
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    tid = irb->CreateIntrinsic(
        llvm::Type::getVoidTy(*ctx), 
        llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, 
        {}
    );

    for (auto ast_node : node.body)
    {
        ast_node->accept(*this);
    }
}

void NVIRBuilder::apply(KernelCallNode &node)
{
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    // each call is a separate object, so it can not be defined twice
    llvm::Function* kernel = defined_functions.at(node.kernel->name);

    std::vector<llvm::Value*> llvm_args;
    for (auto& arg : node.arguments)
    {
        auto* llvm_arg = values.at(arg->ast_id);
        llvm_args.push_back(llvm_arg);
    }

    llvm::Value* ret = irb->CreateCall(kernel, llvm_args);

    values.insert({node.ast_id, ret});
}

void NVIRBuilder::apply(ConstantNode &node)
{
    if (values.contains(node.ast_id))
        return;

    auto& ctx = compiler_state->context;
    llvm::Value* const_float = llvm::ConstantFP::get(*ctx, llvm::APFloat(node.val_f32));

    values.insert({node.ast_id, const_float});
}

void NVIRBuilder::apply(ScalarNode &node)
{
    // scalars arrives from outside
}

void NVIRBuilder::apply(TensorNode &node)
{
    // tensors arrives from outside
}

void NVIRBuilder::apply(AddNode &node)
{
    if (values.contains(node.ast_id))
        return;

    node.lhs->accept(*this);
    node.rhs->accept(*this);
    
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto* lhs = values.at(node.lhs->ast_id);
    auto* rhs = values.at(node.rhs->ast_id);
    
    llvm::Value* lhs_val = lhs;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.lhs))
    {
        auto* lhs_ptr = calc_ptr_from_offset(lhs->getType(), lhs, tid);
        lhs_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), lhs_ptr);
    }

    llvm::Value* rhs_val = rhs;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.rhs))
    { 
        auto* rhs_ptr = calc_ptr_from_offset(rhs->getType(), rhs, tid);
        rhs_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), rhs_ptr);
    }

    auto* ret = irb->CreateFAdd(lhs_val, rhs_val);

    values.insert({node.ast_id, ret});
}

void NVIRBuilder::apply(SubNode &node)
{
    if (values.contains(node.ast_id))
        return;

    node.lhs->accept(*this);
    node.rhs->accept(*this);
    
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto* lhs = values.at(node.lhs->ast_id);
    auto* rhs = values.at(node.rhs->ast_id);
    
    llvm::Value* lhs_val = lhs;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.lhs))
    {
        auto* lhs_ptr = calc_ptr_from_offset(lhs->getType(), lhs, tid);
        lhs_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), lhs_ptr);
    }

    llvm::Value* rhs_val = rhs;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.rhs))
    { 
        auto* rhs_ptr = calc_ptr_from_offset(rhs->getType(), rhs, tid);
        rhs_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), rhs_ptr);
    }

    auto* ret = irb->CreateFSub(lhs_val, rhs_val);

    values.insert({node.ast_id, ret});
}

void NVIRBuilder::apply(MulNode &node)
{
    if (values.contains(node.ast_id))
        return;

    node.lhs->accept(*this);
    node.rhs->accept(*this);
    
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto* lhs = values.at(node.lhs->ast_id);
    auto* rhs = values.at(node.rhs->ast_id);
    
    llvm::Value* lhs_val = lhs;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.lhs))
    {
        auto* lhs_ptr = calc_ptr_from_offset(lhs->getType(), lhs, tid);
        lhs_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), lhs_ptr);
    }

    llvm::Value* rhs_val = rhs;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.rhs))
    { 
        auto* rhs_ptr = calc_ptr_from_offset(rhs->getType(), rhs, tid);
        rhs_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), rhs_ptr);
    }

    auto* ret = irb->CreateFMul(lhs_val, rhs_val);

    values.insert({node.ast_id, ret});
}

void NVIRBuilder::apply(DivNode &node)
{
    if (values.contains(node.ast_id))
        return;

    node.lhs->accept(*this);
    node.rhs->accept(*this);
    
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto* lhs = values.at(node.lhs->ast_id);
    auto* rhs = values.at(node.rhs->ast_id);
    
    llvm::Value* lhs_val = lhs;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.lhs))
    {
        auto* lhs_ptr = calc_ptr_from_offset(lhs->getType(), lhs, tid);
        lhs_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), lhs_ptr);
    }

    llvm::Value* rhs_val = rhs;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.rhs))
    { 
        auto* rhs_ptr = calc_ptr_from_offset(rhs->getType(), rhs, tid);
        rhs_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), rhs_ptr);
    }

    auto* ret = irb->CreateFDiv(lhs_val, rhs_val);

    values.insert({node.ast_id, ret});
}

void NVIRBuilder::apply(SqrtNode &node)
{
    if (values.contains(node.ast_id))
        return;

    node.x->accept(*this);
    
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto* x = values.at(node.x->ast_id);
    
    llvm::Value* x_val = x;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.x))
    {
        auto* x_ptr = calc_ptr_from_offset(x->getType(), x, tid);
        x_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), x_ptr);
    }

    auto* ret = irb->CreateIntrinsic(x_val->getType(), llvm::Intrinsic::nvvm_sqrt_f, x_val);

    values.insert({node.ast_id, ret});
}

void NVIRBuilder::apply(Log2Node &node)
{
    if (values.contains(node.ast_id))
        return;

    node.x->accept(*this);
    
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto* x = values.at(node.x->ast_id);
    
    llvm::Value* x_val = x;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.x))
    {
        auto* x_ptr = calc_ptr_from_offset(x->getType(), x, tid);
        x_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), x_ptr);
    }

    auto* ret = irb->CreateIntrinsic(x_val->getType(), llvm::Intrinsic::nvvm_lg2_approx_f, x_val);

    values.insert({node.ast_id, ret});
}

void NVIRBuilder::apply(Exp2Node &node)
{
    if (values.contains(node.ast_id))
        return;

    node.x->accept(*this);
    
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto* x = values.at(node.x->ast_id);
    
    llvm::Value* x_val = x;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.x))
    {
        auto* x_ptr = calc_ptr_from_offset(x->getType(), x, tid);
        x_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), x_ptr);
    }

    auto* ret = irb->CreateIntrinsic(x_val->getType(), llvm::Intrinsic::nvvm_ex2_approx_f, x_val);

    values.insert({node.ast_id, ret});
}

void NVIRBuilder::apply(AssignmentNode &node)
{
    node.src->accept(*this);

    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto* src = values.at(node.src->ast_id);
    auto* trg = values.at(node.trg->ast_id);
    
    llvm::Value* src_val = src;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.src))
    {
        auto* src_ptr = calc_ptr_from_offset(src->getType(), src, tid);
        src_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), src_ptr);
    }

    auto* trg_ptr = calc_ptr_from_offset(trg->getType(), trg, tid);
    irb->CreateStore(src, trg_ptr);
}

void NVIRBuilder::apply(AliasNode &node)
{
    if (values.contains(node.ast_id))
        return;
    
    node.src->accept(*this);
    
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    auto src = values.at(node.src->ast_id);
    llvm::Value* src_val = src;  // if not a tensor
    if (std::dynamic_pointer_cast<TensorNode>(node.src))
    {
        auto* src_ptr = calc_ptr_from_offset(src->getType(), src, tid);
        src_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), src_ptr);
    }

    values.insert({node.ast_id, src_val});
}

void NVIRBuilder::apply(ReturnNode &node)
{
    auto& ctx = compiler_state->context;
    auto& irb = compiler_state->ir_builder;

    if (node.return_value)
    {
        node.return_value->accept(*this);

        auto* return_value = values.at(node.return_value->ast_id);
    
        llvm::Value* return_value_val = return_value;  // if not a tensor
        if (std::dynamic_pointer_cast<TensorNode>(node.return_value))
        {
            auto* return_value_ptr = calc_ptr_from_offset(return_value->getType(), return_value, tid);
            return_value_val = irb->CreateLoad(llvm::Type::getFloatTy(*ctx), return_value_ptr);
        }

        irb->CreateRet(return_value_val);
    }
    else
    {
        irb->CreateRetVoid();
    }
}
