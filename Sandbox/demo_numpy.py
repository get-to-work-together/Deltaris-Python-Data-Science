
import numpy as np
import matplotlib.pyplot as plt

# %%


a = np.arange(1, 10)

print(a)
print(type(a))


print( np.zeros(10) )
print( np.ones(10) )

print( np.linspace(1, 10, 11) )

a = np.linspace(1, 10, 10)

a[3] = 5.1

print(a)
print(type(a))
print(a.dtype)
print(a.ndim)
print(a.shape)


m = np.arange(1, 13).reshape(3, 4)

print(m)
print(m.ndim)
print(m.shape)
print(m.dtype)

print(m.size)
print(len(m))

print(a)
print(a[4])
print(a[4:7])


b = a[4:7] # VIEW !!!!
print(b)

a[5] = 99
print(a)

print(b)


print(m)
print(m[1])
print(m[1][2])

print(m[1, 2])

print(m[:2, 2:])


print(a)
print(a[ [3,1,2,5] ])

print(a + a)
print(a * 100)

print(a > 5)

print(a[ a > 5 ])

# %%

x = np.linspace(1, 10, 100)
y = np.sin(x)/(x + 1)

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.grid()
plt.show()


noise = np.random.randn(100)*1


# %%

m = np.array([1,2,3,4,5, 6, 7, 8]).reshape(4, 2)

print( m )
print( (m.T + [10, 20, 30, 40]).T )


# %%


a = np.arange(1, 11)

b = a[3:6].copy()

print(a)
print(b)

a[4] = 9999

print(a)
print(b)

# %%


data = np.random.randn(1000)

plt.hist(data, bins=np.linspace(-4, 4, 17))
plt.show()

plt.scatter(data, np.random.randn(1000))

