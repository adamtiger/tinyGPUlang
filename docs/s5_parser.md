# TGL parser

The parser reads the tgl file and builds the corresponding AST.

## How does it work?

The parser uses internally a tokenizer to read the relevant text pieces.
These can be keywords (e.g. func, global etc.), variable or function names,
characters like +, ) etc.

The parser first always looks for the next kernel function header. The kernel header is assumed to be a single line (like any other expression).
After reading the header, the statements in the body is read one-by-one to an AST. 

The parser also checks for some common syntax errors, included but not limited to:
* missing artihmetic operator among operands
* unknown variable name
* wrong paranthesis (e.g. missing)
* function call argument checks (type, number etc.)
* ...

## Back to readme

[README](../README.md)
