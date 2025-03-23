def get_number_input(lower, upper):
    """Prompt the user to enter a number between the given lower and upper bounds.
    Continue asking until a correct number is entered"""
    
    while True:
        try:
            number = int(input(f'Give a number between {lower} and {upper}: '))
            if lower <= number <= upper:
                return number
            else:
                print(f'That number is not between {lower} and {upper}!')
        except ValueError:
            print(f'That is not a number!')


if __name__ is '__main__':
    print( get_number_input(1, 10) )
