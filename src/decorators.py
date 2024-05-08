import inspect
from typing import Any, Callable, TypeVar, Type, Tuple, Dict, Union, List
import argparse
from numbers import Number

Argument = Dict[str, Dict[str, Union[Any, str, List[str], Type]]]
"""One argument may look like this:

```python
argname={
    "default":"default_value", 
    "type":str, 
    "help":"A descriptive help message, aaaah, HEEEELP!",
    "flags":["argn", "an"]
}, 
```
"""

RT = TypeVar("RT")
def parseargs(__description:str="", __help:bool=False, **parseArgs:Argument)->Callable[[Callable[[Any], RT]], Callable[[Any], RT]]:
    """Decorator to set the arguments of a function from system arguments
    Arguments parsed that are not part of the function signature are ignored.

    ### Args:
        __description (str, optional): The description of the parser. Defaults to "".
        __help (bool, optional): If the help flag should be added. Defaults to False.

        
    ### Things to consider:
    All other arguments are the arguments that should be parsed. 
    The key is the name of the argument, the value is the default value. 
    The type is inferred from the function signature, so please type your functions. (if not, the type is inferred from the default value)
    If a default value is given, the argument is optional, if not it is positional and required.
    If this positional argument has no type, it is assumed to be a string.
    If you need to parse arguments that have a whitespace in them, replace the whitespace with a #.

    ### Example:
    ```python
    @parseargs(your_arg={"default":"default", "type":str, "help":"A help message for your_arg", "flags":["ya"]},#use flags to provide other names for the argument, is a list 
        __description="The entry point of your program.\nMay be be called this way:\n\tpython src/main.py [--arg value]*", 
        __help=True
    )
    def main(your_arg:str):
        ...

    ```
    --- 
    If you don't want the decorated function to parse the arguments from sys.argv (cmdline args the python program got called with), but only a subset of these, remove the unwanted subset of sys.argv and parse the rest like so:

    ```bash
    python your_program.py --unwanted True --wanted 4
    ```

    ```python
    @parseargs(wanted={"default":1})
    def your_function(wanted:int):
        ...
    
    import argparse
    if __name__ == "__main__":
        #get the unwanted args out.
        
        parser = argparse.ArgumentParser()
        parser.addArgument("--unwanted", type=bool)
        unwanted, wanted = parser.parse_known_args() #parses only the known args, leaves the rest as is

        #do something with the unwanted stuff
        ...

        #now that only parses \" --wanted 4 \" to your_function
        your_function(*wanted)
    ```
    """
    def wrapper(func:Callable[[Any],RT])->Callable[[Any],RT]:
        def wrapped(*args, **kwargs)->RT:

            class WhiteSpaceAction(argparse.Action):
                def __call__(self, parser, namespace, values, option_string=None):
                    setattr(namespace, self.dest, [v.replace("#", " ") if isinstance(v, str) else v for v in values] if len(values)>1 else values[0])

            parser = argparse.ArgumentParser()
            if __description:
                parser.description = __description
            if __help:
                parser.add_help
            
            is_annotated = {arg:False for arg in inspect.signature(func).parameters}
            is_annotated.update({arg:True for arg in func.__annotations__ if arg!="return"})

            def inferTypeFromDefault(arg_name:str, default_value:Any, func:Callable)->Tuple[Type, int]:
                """Infer the type, and number of the argument from the default value or the function signature."""
                #find out what type based on the function signature or the default value
                if isinstance(default_value, list) and not is_annotated[arg_name]:
                    raise TypeError(f"Got no type for the argument {arg_name}, but it's value suggests List, this is not allowed.")

                #make types (remove List[str] and make str and give n_args="*")
                argtype:Type = func.__annotations__[arg_name]
                is_list:bool = False
                inferred_type:Type = str
                
                if str(argtype).startswith("typing.List") or argtype==list:#if type is typing lst
                    argtype = argtype.__args__[0]
                    is_list=True

                if default_value is None:
                    inferred_type =str if not is_annotated[arg_name] else argtype
                    nargs = 1
                else:
                    inferred_type = type(default_value) if not is_annotated[arg_name] else argtype
                    nargs = "*" if is_list else 1

                return inferred_type, nargs

            for arg in parseArgs:
                nargs:int = 0
                inferred_type:Type = str
                if isinstance(parseArgs[arg], str):
                    #only the name and the default provided, no help, no type, no nargs, no required
                    default_value:Any = parseArgs[arg]

                    inferred_type, nargs = inferTypeFromDefault(arg, default_value, func)

                    parseArgs[arg] = {
                        "default":default_value, 
                        "help":(f"Default: {default_value}" if str not in inferred_type.__mro__  \
                            else f"Default: \"{default_value}\"") \
                            + f", Type: {inferred_type}",
                        "nargs":nargs, 
                        "required":False, 
                        "type":inferred_type,
                        "flags":[arg]
                    }

                elif isinstance(parseArgs[arg], dict):
                    #make sure it has the desired fields, if not add empty ones
                    if "default" not in parseArgs[arg]:
                        parseArgs[arg]["default"] = None

                    inferred_type, nargs = inferTypeFromDefault(arg, parseArgs[arg]["default"], func)
                    if "nargs" not in parseArgs[arg]:
                        parseArgs[arg]["nargs"] = nargs
                    if "required" not in parseArgs[arg]:
                        parseArgs[arg]["required"] = False
                    if "type" not in parseArgs[arg]:
                        parseArgs[arg]["type"] = inferred_type

                    helpMsg:str = (f"Default: {parseArgs[arg]['default']}" if str not in parseArgs[arg]['type'].__mro__  \
                            else f"Default: \"{parseArgs[arg]['default']}\"") \
                            + f", Type: {parseArgs[arg]['type']}"

                    if "help" not in parseArgs[arg]:
                        parseArgs[arg]["help"] = helpMsg
                    else:
                        parseArgs[arg]["help"] += "; "+helpMsg
                    
                    if "flags" in parseArgs[arg]:
                        if not isinstance(parseArgs[arg]["flags"], list):
                            raise TypeError(f"Got a value of type {type(parseArgs[arg]['flags'])} for the flags of the argument {arg}, but only lists are allowed.")
                        if arg not in parseArgs[arg]["flags"]:
                            
                            parseArgs[arg]["flags"].insert(0,arg)
                    else:
                        parseArgs[arg]["flags"] = [arg]
                else:
                    raise TypeError(f"Got a value of type {type(parseArgs[arg])} for the argument {arg}, but only str and dict are allowed.")

            for arg, properties in parseArgs.items():
                #add the args to the argparser parser with all the info from the dict for each argument
                parser.add_argument(*[f"--{flag}" for flag in properties["flags"]], type=properties["type"], default=properties["default"], nargs=properties["nargs"], help=properties["help"], required=properties["required"], action=WhiteSpaceAction)
                
            parsedArgs = parser.parse_args(args) if len(args)>0 else parser.parse_args()

            if kwargs:
                kwargs.update(vars(parsedArgs))
            else:
                kwargs = vars(parsedArgs)

            return func(**kwargs)
        return wrapped
    return wrapper
