import random
import numpy as np

class MinHash:
    def __init__(self, num_hashes):
        self.num_hashes = num_hashes
        self.bucket_params = []
        self.reduction_params = []
        self.p = 2**31 - 1  # A large prime number
        random.seed()
        for i in range(num_hashes):
            a = random.randint(1, self.p - 1)
            b = random.randint(0, self.p - 1)
            self.bucket_params.append((a, b))
        random.seed()
        for i in range(num_hashes):
            a = random.randint(1, self.p - 1)
            b = random.randint(0, self.p - 1)
            self.reduction_params.append((a, b))
    
    def index_entry(self, document):
        bit_list = []
        for i in range(self.num_hashes):
            term_hashes = []
            for term in document:
                a, b = self.bucket_params[i]
                h = (a * term + b) % self.p
                term_hashes.append(h)
            y = np.min(term_hashes)
            a_r, b_r = self.reduction_params[i]
            reduced = ((a_r * y + b_r) % self.p) % 2
            bit_list.append(reduced)
        return int("".join(str(int(b)) for b in bit_list), 2)