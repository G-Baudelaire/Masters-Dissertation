import math


class Utils:
    @staticmethod
    def bits_needed(n):
        bits_p = math.ceil(math.log2(math.sqrt(n)))
        bits_q = math.ceil(math.log2(n)) - 1
        return bits_p, bits_q