from .umeyama import umeyama

def get_power_of_two(x):
    i = 0
    while (1 << i) < x:
        i += 1
    return i