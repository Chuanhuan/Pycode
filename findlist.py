import numpy as np
import pandas as pd

a = np.array([1,3,4,5,6,9,11,15])
b = [1,3,4,5,6,9,11,15]
#  np.where(a==4)
#  b.index(4)

def myfuc(a,x):
    l = len(a)
    for i in a:
        if i == x  :
            print('found')
            break
        elif a.index(i)==l-1:
            print('not in list')

myfuc(b,8)
print('in file')
if __name__ == '__main__':
    #  myfuc(b,8) 
    print('name in Main')
