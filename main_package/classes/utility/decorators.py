from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print (f"{f.__name__} args:[{args},{kw}] took: {te-ts} sec") 
          
        return result
    return wrap