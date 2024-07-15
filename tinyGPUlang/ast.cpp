#include "ast.hpp"
#include "core.hpp"

ASTNode::ASTNode()
{
    ast_id = GlobalUUIDGenerator::generate_uuid();
}

VariableNode::VariableNode(
    const VariableType vtype, const DataType dtype, const std::string& name
    ) : ASTNode(), vtype(vtype), dtype(dtype), name(name)
{
}


ScalarNode::ScalarNode(
    const VariableType vtype, const DataType dtype, const std::string& name
    ) : VariableNode(vtype, dtype, name)
{
}

void ScalarNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr ScalarNode::create_scalar_node(
    const VariableType vtype, 
    const DataType dtype,
    const std::string& name)
{
    return std::make_shared<ScalarNode>(vtype, dtype, name);   
}


TensorNode::TensorNode(
    const VariableType vtype, 
    const DataType dtype,
    const std::string& name,
    const std::vector<int>& shape
    ) : VariableNode(vtype, dtype, name), shape(shape)
{
}

void TensorNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr TensorNode::create_tensor_node(
    const VariableType vtype, 
    const DataType dtype,
    const std::string& name,
    const std::vector<int>& shape)
{
    return std::make_shared<TensorNode>(vtype, dtype, name, shape);   
}


KernelNode::KernelNode(
    const std::string& name,
    const KernelScope scope, 
    const std::vector<VariableNodePtr>& arguments,
    const VariableNodePtr return_value
    ) : ASTNode(), name(name), scope(scope), 
        arguments(arguments),
        return_value(return_value)
{
}

void KernelNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr KernelNode::create_kernel_node(
    const std::string& name,
    const KernelScope scope, 
    const std::vector<VariableNodePtr>& arguments,
    const VariableNodePtr return_value)
{
    return std::make_shared<KernelNode>(name, scope, arguments, return_value);
}


KernelCallNode::KernelCallNode(
    const KernelNodePtr kernel,
    const std::vector<VariableNodePtr>& arguments,
    const VariableNodePtr return_value
    ) : ASTNode(), kernel(kernel), 
        arguments(arguments),
        return_value(return_value)
{
}

void KernelCallNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr KernelCallNode::create_kernelcall_node(
    const KernelNodePtr kernel, 
    const std::vector<VariableNodePtr>& arguments,
    const VariableNodePtr return_value)
{
    return std::make_shared<KernelCallNode>(kernel, arguments, return_value);
}


BinaryNode::BinaryNode(const ASTNodePtr lhs, const ASTNodePtr rhs) : ASTNode(), lhs(lhs), rhs(rhs)
{
}


AddNode::AddNode(const ASTNodePtr lhs, const ASTNodePtr rhs) : BinaryNode(lhs, rhs)
{
}

void AddNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr AddNode::create_add_node(const ASTNodePtr lhs, const ASTNodePtr rhs)
{
    return std::make_shared<AddNode>(lhs, rhs);
}


SubNode::SubNode(const ASTNodePtr lhs, const ASTNodePtr rhs) : BinaryNode(lhs, rhs)
{
}

void SubNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr SubNode::create_sub_node(const ASTNodePtr lhs, const ASTNodePtr rhs)
{
    return std::make_shared<SubNode>(lhs, rhs);
}


MulNode::MulNode(const ASTNodePtr lhs, const ASTNodePtr rhs) : BinaryNode(lhs, rhs)
{
}

void MulNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr MulNode::create_mul_node(const ASTNodePtr lhs, const ASTNodePtr rhs)
{
    return std::make_shared<MulNode>(lhs, rhs);
}


DivNode::DivNode(const ASTNodePtr lhs, const ASTNodePtr rhs) : BinaryNode(lhs, rhs)
{
}

void DivNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr DivNode::create_div_node(const ASTNodePtr lhs, const ASTNodePtr rhs)
{
    return std::make_shared<DivNode>(lhs, rhs);
}


UnaryNode::UnaryNode(const ASTNodePtr x) : ASTNode(), x(x)
{
}


SqrtNode::SqrtNode(const ASTNodePtr x) : UnaryNode(x)
{
}

void SqrtNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr SqrtNode::create_sqrt_node(const ASTNodePtr x)
{
    return std::make_shared<SqrtNode>(x);
}


Log2Node::Log2Node(const ASTNodePtr x) : UnaryNode(x)
{
}

void Log2Node::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr Log2Node::create_log2_node(const ASTNodePtr x)
{
    return std::make_shared<Log2Node>(x);
}


Exp2Node::Exp2Node(const ASTNodePtr x) : UnaryNode(x)
{
}

void Exp2Node::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr Exp2Node::create_exp2_node(const ASTNodePtr x)
{
    return std::make_shared<Exp2Node>(x);
}


AssignmentNode::AssignmentNode(const ASTNodePtr trg, const ASTNodePtr src) : ASTNode(), trg(trg), src(src)
{
}

void AssignmentNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr AssignmentNode::create_assignment_node(const ASTNodePtr trg, const ASTNodePtr src)
{
    return std::make_shared<AssignmentNode>(trg, src);
}


AliasNode::AliasNode(const ASTNodePtr src) : ASTNode(), src(src)
{
}

void AliasNode::accept(ASTVisitorPtr visitor)
{
    visitor->apply(*this);
}

ASTNodePtr AliasNode::create_alias_node(const ASTNodePtr src)
{
    return std::make_shared<AliasNode>(src);
}
