print('Hello world!!!!')
print("Hello world!!!!")

print("""Hello world!!!!
How do you do?
Beautiful""")

print("Hello world!!!!\nHow do you do?\nBeautiful")

# %% type str

number_of_cars = 'ten'

print(number_of_cars)
print(type(number_of_cars))


# %% type int

number_of_cars = 121


number_of_cars = number_of_cars + 1
number_of_cars += 1

print(number_of_cars)
print(type(number_of_cars))

# %% type bool

end_of_loop = True

end_of_loop = not end_of_loop

print(end_of_loop)

# %% List

shopping_list = []

shopping_list.append('butter')
shopping_list.append('milk')
shopping_list.append('cheese')

print(shopping_list)

# %% indexing and slicing

item = shopping_list[0]

print(item)

print( shopping_list[-1] )

print( shopping_list[1:] )

s = 'ABCDEFGHIJKLM'

print(s[5:10])
print(s[:10])
print(s[5:])
print(s[-3:])


# %%

copy = shopping_list

shopping_list.append('eggs')
shopping_list.insert(0, 'beer')
shopping_list.sort()

print(shopping_list)

print(copy)

# %% Set

st = {1, 2, 3, 5, 5, 5, 5, 5}

st.add(4)
st.add(1)

print(st)

print(4 in st)

# %% dict

d = {'Delft':'015', 'Amsterdam':'020', 'Rotterdam':'010'}

d['Den Haag'] = '070'

print(d['Delft'])

# %% string formatting

name = 'Guido'
age = 62

print(name, 'is', age, 'years old.')
print(name + ' is ' + str(age) + ' years old.')

print(f'{name} is {age} years old.')

print('%s is %d years old.' % (name, age))
print('{} is {} years old.'.format(name, age))


# %% string methods

s = 'the cat jumped on the mouse'

print(s)
print(s.upper())
print(s.title())
print(s.count('o'))
print(len(s))

words = s.split()

print(words)

print( ' - '.join(words) )

words.join('-')

           
# %% flow control

gender = 'x'

if gender == 'm':
    print('Dear sir')
    
elif gender == 'f':
    print('Dear madam')
    
else:
    print('Dear')
            
print('How do you do?')



# %% for

words = 'the big brown dog'.split()
print(words)
for word in words:
    print('>', word)


# %%

magic_number = 14

for number in range(1, 21):
    
    if number == magic_number:
        continue
    
    print(number)


# %% comprehension

numbers = [3,5,7,8]

cubes = []
for number in numbers:
    if number > 5:  
        cubes.append(number ** 3)

print(cubes)


cubes = [number ** 3 for number in numbers if number > 5]

print(cubes)


# %% function

a = 5

import my_functions


print( my_functions.add_and_multiply(4, 6, 3) )

print( my_functions.add_and_multiply(4, 6) )

print( my_functions.add_and_multiply(4, 6, f=5) )



# this is a comment


# %% Vector


class Vector:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'[{self.x}, {self.y}]'
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)


v1 = Vector(2, 3)
v2 = Vector(5, -1)

print(v1)
print(v2)

v3 = v1 + v2

print(v3)


# %% Function

def do_something():
    return 30

print( do_something() )
print( do_something )







