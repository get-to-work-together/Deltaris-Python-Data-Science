import random

secret_number = random.randint(1, 100)

print('Raad een getal tussen 1 en 100.')

aantal_keer = 0
while True:
    gok = int(input('Wat denk je dat het getal is? '))
    aantal_keer += 1
    
    if gok > secret_number:
        print('lager ...')

    elif gok < secret_number:
        print('hoger ...')

    else:
        print('Jaaaaa! Goed geraden!')
        print(f'Je hebt daar {aantal_keer} keer voor nodig gehad.')
        break

