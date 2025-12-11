from minhash import MinHash
from collections import defaultdict
import random
class LSHTable:
    def __init__(self, num_hashes):
        self.minhash = MinHash(num_hashes)
        self.table = defaultdict(list)
    
    def add(self, document, document_name):
        index = self.minhash.index_entry(document)
        self.table[index].append(document_name)
    
    def query(self, query):
        index = self.minhash.index_entry(query)
        results = self.table[index]
        return results
    