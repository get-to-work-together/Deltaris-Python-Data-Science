namen = []
while True:
    naam = input('Geef de volgende naam: ')
    if naam:
        namen.append(naam)
    else:
        break

print('De ingevoerde namen zijn:')
for naam in sorted(namen):
    print(f'>> {naam}')
