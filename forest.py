from tree import LSHTree, bits_from_int
import random
import bisect

class LSHForest:
    def __init__(self, num_trees, num_hashes):
        self.num_trees = num_trees
        self.num_hashes = num_hashes
        self.trees = []
        for i in range(num_trees):
            self.trees.append(LSHTree(self.num_hashes))
    
    def insert(self, document, document_name):
        for tree in self.trees:
            tree.insert(document, document_name)

    def candidate_search(self, query, c, m):
        candidates = []
        for tree in self.trees:
            results = tree.query(query, c, m)
            for result in results:
                candidates.append(result)
        return candidates
    
    def candidate_search_sync(self, query, c, m):
        leaves = []
        depths = []
        for tree in self.trees:
            leaf = tree.descend(query)
            leaves.append(leaf)
            depths.append(leaf.get_depth())
        candidates = []
        current_depth = max(depths)
        while current_depth > 0 and (len(candidates) < c * len(self.trees) or len(list(set(candidates))) < m):
            candidates = []
            for i, tree in enumerate(self.trees):
                if(depths[i] != current_depth):
                    continue
                candidates.extend(tree.query_sync(leaves[i], current_depth))
                depths[i] -= 1
            current_depth -= 1
        return list(set(candidates))


    def aggregate_level_counts(self):
        return self.trees[0].calculate_depth_stats()