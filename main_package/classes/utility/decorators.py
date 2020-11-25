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


def timing_write(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print (f"{f.__name__} args:[{args},{kw}] took: {te-ts} sec") 

        with open("../results/results.csv", 'a+') as fi:
            fi.write(f"\n{round(te-ts,3)},")
          
        return result
    return wrap