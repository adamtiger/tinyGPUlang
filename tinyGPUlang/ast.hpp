#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_set>

// fprward declaration
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
    FLOAT32,
    FLOAT16
};

std::ostream& operator<<(std::ostream& os, const DataType var_type);

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

struct ScalarNode : public VariableNode
{
    explicit ScalarNode(
        const VariableType vtype, 
        const DataType dtype, 
        const std::string& name);

    virtual void accept(ASTVisitor& visitor) override;

    static ASTNodePtr create_scalar_node(
        const VariableType vtype, 
        const DataType dtype,
        const std::string& name);
};

struct TensorNode : public VariableNode
{
    std::vector<int> shape;

    explicit TensorNode(
        const VariableType vtype, 
        const DataType dtype, 
        const std::string& name,
        const std::vector<int>& shape);

    virtual void accept(ASTVisitor& visitor) override;

    static ASTNodePtr create_tensor_node(
        const VariableType vtype, 
        const DataType dtype,
        const std::string& name,
        const std::vector<int>& shape);
};

using VariableNodePtr = std::shared_ptr<VariableNode>;

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

    static ASTNodePtr create_kernel_node(
        const std::string& name,
        const KernelScope scope, 
        const std::vector<VariableNodePtr>& arguments,
        const VariableNodePtr return_value);
};

using KernelNodePtr = std::shared_ptr<KernelNode>;

struct KernelCallNode : public ASTNode
{
    KernelNodePtr kernel;
    std::vector<VariableNodePtr> arguments;

    explicit KernelCallNode(
        const KernelNodePtr kernel,
        const std::vector<VariableNodePtr>& arguments);

    virtual void accept(ASTVisitor& visitor) override;

    static ASTNodePtr create_kernelcall_node(
        const KernelNodePtr kernel,
        const std::vector<VariableNodePtr>& arguments);
};

using KernelCallNodePtr = std::shared_ptr<KernelCallNode>;

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
    static ASTNodePtr create_add_node(const ASTNodePtr lhs, const ASTNodePtr rhs);
};

struct SubNode : BinaryNode
{
    explicit SubNode(const ASTNodePtr lhs, const ASTNodePtr rhs);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_sub_node(const ASTNodePtr lhs, const ASTNodePtr rhs);
};

struct MulNode : BinaryNode
{
    explicit MulNode(const ASTNodePtr lhs, const ASTNodePtr rhs);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_mul_node(const ASTNodePtr lhs, const ASTNodePtr rhs);
};

struct DivNode : BinaryNode
{
    explicit DivNode(const ASTNodePtr lhs, const ASTNodePtr rhs);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_div_node(const ASTNodePtr lhs, const ASTNodePtr rhs);
};

// unary ops
struct UnaryNode : ASTNode
{
    ASTNodePtr x;

    explicit UnaryNode(const ASTNodePtr x);
};

struct SqrtNode : UnaryNode
{
    explicit SqrtNode(const ASTNodePtr x);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_sqrt_node(const ASTNodePtr x);
};

struct Log2Node : UnaryNode
{
    explicit Log2Node(const ASTNodePtr x);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_log2_node(const ASTNodePtr x);
};

struct Exp2Node : UnaryNode
{
    explicit Exp2Node(const ASTNodePtr x);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_exp2_node(const ASTNodePtr x);
};

// data movement ops
struct AssignmentNode : ASTNode  // d = a + b;
{
    ASTNodePtr trg;
    ASTNodePtr src;

    explicit AssignmentNode(const ASTNodePtr trg, const ASTNodePtr src);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_assignment_node(const ASTNodePtr trg, const ASTNodePtr src);
};

using AssignmentNodePtr = std::shared_ptr<AssignmentNode>;

struct AliasNode : ASTNode  // var d = a + b;
{
    std::string name;
    ASTNodePtr src;

    explicit AliasNode(const std::string& alias_name, const ASTNodePtr src);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_alias_node(const std::string& alias_name, const ASTNodePtr src);
};

using AliasNodePtr = std::shared_ptr<AliasNode>;

struct ReturnNode : ASTNode  // return d;
{
    ASTNodePtr return_value;

    explicit ReturnNode(const ASTNodePtr return_value);
    virtual void accept(ASTVisitor& visitor) override;
    static ASTNodePtr create_node(const ASTNodePtr return_value);
};

using ReturnNodePtr = std::shared_ptr<ReturnNode>;

// defintion of visitor base class
class ASTVisitor
{
public:

    virtual void apply(KernelNode& node) = 0;
    virtual void apply(KernelCallNode& node) = 0;
    
    virtual void apply(ScalarNode& node) = 0;
    virtual void apply(TensorNode& node) = 0;

    virtual void apply(AddNode& node) = 0;
    virtual void apply(SubNode& node) = 0;
    virtual void apply(MulNode& node) = 0;
    virtual void apply(DivNode& node) = 0;

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
    
    virtual void apply(ScalarNode& node);
    virtual void apply(TensorNode& node);

    virtual void apply(AddNode& node);
    virtual void apply(SubNode& node);
    virtual void apply(MulNode& node);
    virtual void apply(DivNode& node);

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
