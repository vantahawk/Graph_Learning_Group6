import os
import itertools
from json import JSONEncoder
import numpy

def unique_file(basename:str)->str:
    """Return a unique file name by appending a number to the file name if the file already exists.
        Linear runtime in the number of files with the same basename.
    """
    basename, ext = os.path.splitext(basename)
    ext = ext.removeprefix(".") #save to call even if no . is present
    actualname = "%s.%s" % (basename, ext) if ext!="" else "%s" % basename
    c = itertools.count()
    next(c)#start at 1
    while os.path.exists(actualname):
        actualname = "%s_(%d).%s" % (basename, next(c), ext) if ext!="" else "%s_(%d)" % (basename, next(c))
    return actualname

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "item") and callable(obj.item):
            return obj.item()
        return JSONEncoder.default(self, obj)