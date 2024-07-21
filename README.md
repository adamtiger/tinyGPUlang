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
* abs

### Example code

Files should end with **tgl**. No import of other file is supported.

```
func device f32 calc_square_diff(f32 a, f32 b)
{
    var e = a - b;     // result is stored in a temporary variable (defined with var)
    var e2 = e * e;
    return e2;   
}

func global void calc_mse(f32[4, 5, 6] a, f32[4, 5, 6] b, f32 c, f32[4, 5, 6] d)
{
    var e2 = calc_square_diff(a, b);  // calling device function
    var me2 = e2 * c;                 // some normalization factor
    d = sqrt(me2);                    // copies the result into d

    // other examples
    // d = c;  // broadcast c to d
    // d = d + a;
}
```

### TODO

- [x] compiler should have argument parameters (for command line useage)
- [x] output llvm assembly to *ll files
- [ ] consider inline constants (e.g. var e = c + 4.6f;)
- [ ] refactor (parser should give better results and more checks for illegal situation, cleaning, ensure consistency, no size for tensor)
- [ ] test for edge cases
- [ ] test on linux
- [ ] how to set sm?
- [ ] document, write the tutorial
