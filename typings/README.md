# Local typing declarations

`cupy/__init__.pyi` describes only the array identity and metadata needed by
ToMoBAR's initial Pyright scope. It lets solver-state array annotations resolve
without installing CuPy, ASTRA or CUDA in the type-check environment.

This is not a declaration of CuPy's numerical API. Expand or replace it with
upstream typing support before adding GPU execution modules to the checked scope.
The dtype parameter is intentionally unrestricted: metadata may describe arrays
with different NumPy-compatible scalar dtypes.

These files are static-analysis inputs only. They are outside the `tomobar`
package and are not distributed as replacement CuPy runtime modules.
