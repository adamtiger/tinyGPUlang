#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_set>

// forward declaration
class ASTVisitor;

// AST base
struct ASTNode
{
    int ast_id;  // unique id

    explicit ASTNode();
    virtual void accept(ASTVisitor& visitor) = 0;
};

using ASTNodePtr = std::shared_ptr<ASTNode>;

// variables
enum class VariableType
{
    SCALAR,
    TENSOR
};

std::ostream& operator<<(std::ostream& os, const VariableType var_type);

enum class DataType
{
    FLOAT32
};

std::ostream& operator<<(std::ostream& os, const DataType var_type);


struct ConstantNode : public ASTNode
{
    union {
        float val_f32;
    };

    DataType dtype;

    explicit ConstantNode(
        const float value,
        const DataType dtype);

    virtual void accept(ASTVisitor& visitor) override;
};

using ConstantNodePtr = std::shared_ptr<ConstantNode>;

ConstantNodePtr create_constant_node(
    const float value,
    const DataType dtype);


struct VariableNode : public ASTNode
{
    VariableType vtype;
    DataType dtype;
    std::string name;

    explicit VariableNode(
        const VariableType vtype, 
        const DataType dtype, 
        const std::string& name);
};

using VariableNodePtr = std::shared_ptr<VariableNode>;


struct ScalarNode : public VariableNode
{
    explicit ScalarNode(
        const DataType dtype, 
        const std::string& name);

    virtual void accept(ASTVisitor& visitor) override;
};

using ScalarNodePtr = std::shared_ptr<ScalarNode>;

ScalarNodePtr create_scalar_node(
    const DataType dtype,
    const std::string& name);


struct TensorNode : public VariableNode
{
    explicit TensorNode(
        const DataType dtype, 
        const std::string& name);

    virtual void accept(ASTVisitor& visitor) override;
};

using TensorNodePtr = std::shared_ptr<TensorNode>;

TensorNodePtr create_tensor_node(
    const DataType dtype,
    const std::string& name);

// kernel node, represents a kernel function

enum class KernelScope
{
    GLOBAL,
    DEVICE
};

std::ostream& operator<<(std::ostream& os, const KernelScope scope);

struct KernelNode : public ASTNode
{
    KernelScope scope;
    std::string name;  // handled as unique (no overloading)
    std::vector<VariableNodePtr> arguments;
    VariableNodePtr return_value;  // nullptr if void

    std::vector<ASTNodePtr> body;  // set of expressions, topological ordering

    explicit KernelNode(
        const std::string& name,
        const KernelScope scope, 
        const std::vector<VariableNodePtr>& arguments,
        const VariableNodePtr return_value);

    virtual void accept(ASTVisitor& visitor) override;
};

using KernelNodePtr = std::shared_ptr<KernelNode>;

KernelNodePtr create_kernel_node(
    const std::string& name,
    const KernelScope scope, 
    const std::vector<VariableNodePtr>& arguments,
    const VariableNodePtr return_value);


struct KernelCallNode : public ASTNode
{
    KernelNodePtr kernel;
    std::vector<VariableNodePtr> arguments;

    explicit KernelCallNode(
        const KernelNodePtr kernel,
        const std::vector<VariableNodePtr>& arguments);

    virtual void accept(ASTVisitor& visitor) override;
};

using KernelCallNodePtr = std::shared_ptr<KernelCallNode>;

KernelCallNodePtr create_kernelcall_node(
    const KernelNodePtr kernel,
    const std::vector<VariableNodePtr>& arguments);


// arithmetic nodes (binary ops.)
struct BinaryNode : ASTNode
{
    ASTNodePtr lhs;
    ASTNodePtr rhs;

    explicit BinaryNode(const ASTNodePtr lhs, const ASTNodePtr rhs);
};


struct AddNode : BinaryNode
{
    explicit AddNode(const ASTNodePtr lhs, const ASTNodePtr rhs);
    virtual void accept(ASTVisitor& visitor) override;
};

using AddNodePtr = std::shared_ptr<AddNode>;

AddNodePtr create_add_node(
    const ASTNodePtr lhs, 
    const ASTNodePtr rhs);


struct SubNode : BinaryNode
{
    explicit SubNode(const ASTNodePtr lhs, const ASTNodePtr rhs);
    virtual void accept(ASTVisitor& visitor) override;
};

using SubNodePtr = std::shared_ptr<SubNode>;

SubNodePtr create_sub_node(
    const ASTNodePtr lhs, 
    const ASTNodePtr rhs);


struct MulNode : BinaryNode
{
    explicit MulNode(const ASTNodePtr lhs, const ASTNodePtr rhs);
    virtual void accept(ASTVisitor& visitor) override;
};

using MulNodePtr = std::shared_ptr<MulNode>;

MulNodePtr create_mul_node(
    const ASTNodePtr lhs, 
    const ASTNodePtr rhs);


struct DivNode : BinaryNode
{
    explicit DivNode(const ASTNodePtr lhs, const ASTNodePtr rhs);
    virtual void accept(ASTVisitor& visitor) override;
};

using DivNodePtr = std::shared_ptr<DivNode>;

DivNodePtr create_div_node(
    const ASTNodePtr lhs, 
    const ASTNodePtr rhs);


// unary ops
struct UnaryNode : ASTNode
{
    ASTNodePtr x;

    explicit UnaryNode(const ASTNodePtr x);
};


struct AbsNode : UnaryNode
{
    explicit AbsNode(const ASTNodePtr x);
    virtual void accept(ASTVisitor& visitor) override;
};

using AbsNodePtr = std::shared_ptr<AbsNode>;

AbsNodePtr create_abs_node(
    const ASTNodePtr x);


struct SqrtNode : UnaryNode
{
    explicit SqrtNode(const ASTNodePtr x);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_sqrt_node(const ASTNodePtr x);
};

using SqrtNodePtr = std::shared_ptr<SqrtNode>;

SqrtNodePtr create_sqrt_node(
    const ASTNodePtr x);


struct Log2Node : UnaryNode
{
    explicit Log2Node(const ASTNodePtr x);
    virtual void accept(ASTVisitor& visitor) override;
};

using Log2NodePtr = std::shared_ptr<Log2Node>;

Log2NodePtr create_log2_node(
    const ASTNodePtr x);


struct Exp2Node : UnaryNode
{
    explicit Exp2Node(const ASTNodePtr x);
    virtual void accept(ASTVisitor& visitor) override;
};

using Exp2NodePtr = std::shared_ptr<Exp2Node>;

Exp2NodePtr create_exp2_node(
    const ASTNodePtr x);

// data movement ops
struct AssignmentNode : ASTNode  // d = a + b;
{
    ASTNodePtr trg;
    ASTNodePtr src;

    explicit AssignmentNode(const ASTNodePtr trg, const ASTNodePtr src);
    virtual void accept(ASTVisitor& visitor) override;
};

using AssignmentNodePtr = std::shared_ptr<AssignmentNode>;

AssignmentNodePtr create_assignment_node(
    const ASTNodePtr trg, 
    const ASTNodePtr src);


struct AliasNode : ASTNode  // var d = a + b;
{
    std::string name;
    ASTNodePtr src;

    explicit AliasNode(const std::string& alias_name, const ASTNodePtr src);
    virtual void accept(ASTVisitor& visitor) override;
};

using AliasNodePtr = std::shared_ptr<AliasNode>;

AliasNodePtr create_alias_node(
    const std::string& alias_name, 
    const ASTNodePtr src);


struct ReturnNode : ASTNode  // return d;
{
    ASTNodePtr return_value;

    explicit ReturnNode(const ASTNodePtr return_value);
    virtual void accept(ASTVisitor& visitor) override;
};

using ReturnNodePtr = std::shared_ptr<ReturnNode>;

ReturnNodePtr create_return_node(const ASTNodePtr return_value);

// defintion of visitor base class
class ASTVisitor
{
public:

    virtual void apply(KernelNode& node) = 0;
    virtual void apply(KernelCallNode& node) = 0;
    
    virtual void apply(ConstantNode& node) = 0;
    virtual void apply(ScalarNode& node) = 0;
    virtual void apply(TensorNode& node) = 0;

    virtual void apply(AddNode& node) = 0;
    virtual void apply(SubNode& node) = 0;
    virtual void apply(MulNode& node) = 0;
    virtual void apply(DivNode& node) = 0;

    virtual void apply(AbsNode& node) = 0;
    virtual void apply(SqrtNode& node) = 0;
    virtual void apply(Log2Node& node) = 0;
    virtual void apply(Exp2Node& node) = 0;

    virtual void apply(AssignmentNode& node) = 0;
    virtual void apply(AliasNode& node) = 0;
    virtual void apply(ReturnNode& node) = 0;
};


// printer visitor (to visualize the AST)
class ASTPrinter : public ASTVisitor
{
public:
    explicit ASTPrinter();
    void save_into_file(const std::string& out_path) const;
    void reset();

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
    std::string ast_as_string;

    std::unordered_set<int> already_printed;
};
