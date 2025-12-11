from lsh import LSHTable
import random

class LSHIndex:
    def __init__(self, num_tables, num_hashes):
        self.num_tables = num_tables
        self.tables = []
        for i in range(num_tables):
            self.tables.append(LSHTable(num_hashes))
    
    def insert(self, document, document_name):
        for table in self.tables:
            table.add(document, document_name)

    def candidate_search(self, query, c):
        candidates = []
        for table in self.tables:
            results = table.query(query)
            if len(results) > c:
                random.shuffle(results)
            candidates.extend(results[:c])
        return candidates