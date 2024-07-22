# tinyGPUlang

Tutorial on building a gpu compiler backend in LLVM

## The language structure

### Data

Two data structures: scalar and tensor.
Scalars are single numbers, tensors can be have any dimension
but they need to have the same size.

Data type: float32 (f32).

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

func global void calc_mse(f32[] a, f32[] b, f32[] c, f32[] d)
{
    var e2 = calc_square_diff(a, b);  // calling device function
    var me2 = e2 * c;                 // some normalization factor
    var me2h = me2 * 0.5;             // 0.5 constant scalar, immediate value
    d = sqrt(me2h);                   // copies the result into d

    // other examples
    // d = c;  // copy c to d
    // d = d + a;
}
```

### TODO

- [x] compiler should have argument parameters (for command line useage)
- [x] output llvm assembly to *ll files
- [x] consider inline constants (e.g. var e = c + 4.6f;)
- [x] remove float16
- [x] in parser error, report the line and column positions
- [x] add abs node
- [x] add output option to cmd
- [ ] refactor (cleaning, some more checks)
- [x] test for edge cases
- [ ] test on linux
- [ ] document, write the tutorial
