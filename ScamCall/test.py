import pandas as pd

t1 = {'a': [1, 2], 'b': [2, 35]}
d1 = pd.DataFrame(t1)
cl = [1111, 1111]
d1['test'] = cl
print(d1)