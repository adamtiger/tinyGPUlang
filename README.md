# tinyGPUlang

Tutorial on building a gpu compiler backend in LLVM

## The language structure

### Data

Two data structures: scalar and tensor.
Scalars are single numbers, tensors can be have any dimension
but they need to have the same size.

Data types: float32 (f32), float16 (f16).

### Operations

The operands can be:
* scalar, scalar
* tensor, scalar
* scalar, tensor
* tensor, tensor (requires the same shape and type)

Operators:
* elementwise add
* elementwise sub
* elementwise mul
* elementwise div
* sin
* sqrt
* exp2
* log2
* pow
* abs

### Example code

Files should end with **tgl**. No import of other file is supported.

```
func device f32[4, 5, 6] calc_square_diff(in: f32[4, 5, 6] a, f32[4, 5, 6] b)
{
    var e = a - b;     // result is stored in a temporary variable (defined with var)
    var e2 = e * e;
    return e2;   
}

func global void calc_mse(in: f32[4, 5, 6] a, f32[4, 5, 6] b, f32 c, out: f32[4, 5, 6] d)
{
    var e2 = calc_square_diff(a, b);  // calling device function
    var me2 = e2 * c;                 // some normalization factor
    d = sqrt(me2);                    // copies the result into d

    // other examples
    // d = c;  // broadcast c to d
    // d = d + a;
}
```

The argument of the function contains **in** and **out** keywords.
The keyword is valid until the other type appears or there are no more arguments.
