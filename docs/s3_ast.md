# Abstract Syntax Tree (AST)

*An abstract syntax tree (AST) is a data structure used in computer science
 to represent the structure of a program or code snippet.*

See [this wikipedia article](https://en.wikipedia.org/wiki/Abstract_syntax_tree) for more details, but the concise sentence above gives a good hunch.

## AST for the TGL lang

The TGL language is simple. The following language constructs need to have a represenetation:

### Kernel

The kernel needs to be represented. Representation has to contain:
- kernel name
- argument types and names
- return type
- scope (global or device)
- kernel body

### Call of (device) kernel from another kernel

The representation should store information about:
- the called kernel
- arguments to be passed

### Call of builtin functions

The functions like sqrt, abs are represented by separate objects. They stores the single argument.

### Binary operations

This are like +, - etc. They are represented by separate objects. Stores the left and right operands.

### Alias

```
var s = (a + b + c);
```
The s variable will be an alias, no storage space associated with it.
In every subsequent expression, s will be an alias to the arithmetic expression on the right side.

### Assignment

```
s = (a + b + c);
```
s has to be a tensor, can not be a scalar. The new value will be stored in memory. Represents the assignment. Stores the left and right side of the assignment.

### Return

Represents the return statement.

## Processing the AST

The AST is processed according to the visitor pattern.
There are two implemented visitors:
- ast printer
- nv llvm ir builder

The following is an example for the output of the printer (.ast file):

```
-- KernelNode 
  id:    3
  name:  add_vec
  scope: GLOBAL
  args:  0, 1, 2, 
  ret:   void
  body:  5, 6, 

-- TensorNode 
  id:        0
  name:      a
  var_type:  TENSOR
  data_type: FLOAT32

-- TensorNode 
  id:        1
  name:      b
  var_type:  TENSOR
  data_type: FLOAT32

-- TensorNode 
  id:        2
  name:      d
  var_type:  TENSOR
  data_type: FLOAT32

-- AssignmentNode 
  id:    5
  src:   4
  trg:   2

-- AddNode 
  id:    4
  lhs:   0
  rhs:   1

-- ReturnNode 
  id:    6
  ret:   void
```

The connection can be understood by the help of the id values.

The tgl code is in the examples/add_vec.tgl file. Which contains:

```
func global void add_vec(f32[] a, f32[] b, f32[] d)
{
    d = a + b;
    return;
}
```

## Next

[The code generator for NVPTX backend](s4_codegen.md)
