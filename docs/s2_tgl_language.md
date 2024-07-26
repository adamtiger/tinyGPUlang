## The language structure

### Phylosophy

The language only capable of doing pointwise operations.
This simplification means, in the thread level, the operations
only happens on the same element for each tensor. 
Pointwise operations are easy to implement on gpus.

Therefore no need for slicing, branching and looping.

### Data

Two data structures: scalar and tensor.
Scalars are single numbers, tensors can have any dimension
but they need to have the same size in a kernel.

Data type: float32 (f32).

* To define a scalar: f32 scalar_name;
* To define a tensor: f32[] tensor_name;

### Operations

The operands can be (for binary):
* scalar, scalar
* tensor, scalar
* scalar, tensor
* tensor, tensor (requires the same shape and type)

The operands (for unary):
* scalar
* tensor

Operators:
* elementwise add
* elementwise sub
* elementwise mul
* elementwise div
* sqrt
* exp2
* log2
* abs

### Example code

Files should end with **tgl**. No import of other file is supported.

```
func device f32 calc_square_diff(f32 a, f32 b)
{
    var e = a - b;     # result is stored in a temporary variable (defined with var)
    var e2 = e * e;
    return e2;   
}

func global void calc_mse(f32[] a, f32[] b, f32[] c, f32[] d)
{
    var e2 = calc_square_diff(a, b);  # calling device function
    var me2 = e2 * c;                 # some normalization factor
    var me2h = me2 * 0.5;             # 0.5 constant scalar, immediate value
    d = sqrt(me2h);                   # calling built-inf cuntion, then copies the result into d
    return;                           # return is compulsory

    # other examples
    # d = c;  // copy c to d
    # d = d + a;
}
```

## Next

[Abstract Syntax Tree](s3_ast.md)
