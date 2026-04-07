 
Implement the parameters module, that converts the parameters x in real use into the complex number c_k. The module includes 3 steps:

1. prepare real values

The x includes two parts: fixed values and free values. It is managed by a table of fixed values.
```
fixed_table = {
"{name1}": "{value2}",
"{name2}": "{value1}",
}
```

Free parameters are automatically inferred as all unique names
in `product_structure` that are not in `fixed_table`. No separate `free_parameters` list is needed.

2. convert real value to complex value, here we use polar form:

```
r = x["{name}_r"]
phi = x["{name}_i"]
y = r * exp(i * phi)
```

3. convert y to c_k, it is managed by a list of lists like:
```
[
 ["a", "b", "c"],
 ["a", "b", "d"],
 ["e", "c"],
 ["e", "d"],
 ["e", "a"],
]
```
meaning:
```
c = [ y["a"] * y["b"] * y["c"],
      y["a"] * y["b"] * y["d"],
      y["e"] * y["c"],
      y["e"] * y["d"],
      y["e"] * y["a"],
    ]
```
Take the product of each inner list and concatenate. In the implementation, we can use int numbers to replace them.

For an efficient implementation, the strings can be replaced into int numbers, and just provide strings in the interface.



