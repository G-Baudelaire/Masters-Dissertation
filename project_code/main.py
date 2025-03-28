import math





if __name__ == '__main__':
    for n in [15, 21, 57, 123, 253, 511, 1011, 2047]:
        p, q = bits_needed(n)
        print(p + q - 2)
