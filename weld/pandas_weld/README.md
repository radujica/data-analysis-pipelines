## Notes about the lazy evaluation within pandas_weld

All pandas_weld objects shall conform to the following:
- `__repr__` shall show the class info without any data or weld_expr
- `.data` shall show the underlying data, be it np.ndarray or weld_expr
- `evaluate()` shall return a new object of the same type but with raw evaluated data within
- `__str__` shall pretty print the data with no guarantee if raw or weld_expr; e.g. for `Series` it is just the `str()`
but for `DataFrame` it's a tabulate pretty print which (NOTE!) evaluates the index if of type `MultiIndex` in order to
create the actual values from labels and levels format