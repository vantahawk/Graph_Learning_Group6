# structure of the repo

## 1. main.py

This is the main file that is the entry point of the program. It contains the main function that is called when the program is run. It should not contain any real implementation

## 2. decorators.py

This file contains the decorators that are used in the program.
These decorators can be used for various purposes, feel free to add more, e.g. to measure the time a function takes to execute.

## 3. preprocessing.py

This file contains the preprocessing functions that are used for the gcn input data.

## 4. layers.py

This file contains the layers implementations that are used in all the gcn models.

## 5. models.py

This file should contain the implementation of the models that are used in the program.

# how to import

To import a file from another file, you can use the following syntax:

```python
from file_name import function_name, module_name

module_name.foo()
bar = function_name()
```