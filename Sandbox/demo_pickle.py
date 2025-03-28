import pickle
import shelve


# %%

data = {'a':[1,2,3,4], 'b':['A','B','B']}


filename = 'data.pickle'

with open(filename, 'wb') as f:
    pickle.dump(data, f)
    
    
# %%

del data

# %%

filename = 'data.pickle'

with open(filename, 'rb') as f:
    data = pickle.load(f)
    
print(data)