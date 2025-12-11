from minhash import MinHash

class Node:
    def __init__(self, depth, parent = None):
        self.parent = parent
        self.children = [None, None]
        self.points = [] #leaf functionality
        self.depth = depth

    def get_children(self):
        return self.children
    
    def set_child(self, index, node):
        self.children[index] = node
    
    def add_point(self, label, document_name):
        self.points.append((label, document_name))
    
    def get_depth(self):
        return self.depth
    
    def set_depth(self, depth):
        self.depth = depth
    
    def get_parent(self):
        return self.parent
    
    def set_parent(self, parent):
        self.parent = parent



def bits_from_int(n, k):
    if n < 0:
        raise ValueError("label integer must be non-negative")
    mask = (1 << k) - 1
    n = n & mask
    return [ (n >> (k - 1 - i)) & 1 for i in range(k) ]

def common_prefix_len(a, b):
    # a, b are lists; return length of common prefix
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n

class LSHTree:
    def __init__(self, k):
        self.k = k
        self.minhash = MinHash(self.k)
        self.root = Node(0)
    
    def insert(self, document, document_name):
        label = self.minhash.index_entry(document)
        label = bits_from_int(label, self.k)
        current_node = self.root
        index = 0
        while index < self.k and current_node.get_children()[label[index]] != None: #descend until no child or maximum depth reached
            current_node = current_node.get_children()[label[index]]
            index = current_node.get_depth()

        if index == self.k:#reached max depth (leaf)
            current_node.add_point(label, document_name)
            return
        if not current_node.points: #intermediate node (or root) with no child in label's direction
            new_node = Node(current_node.get_depth() + 1, parent = current_node)
            new_node.add_point(label, document_name)
            current_node.set_child(label[index], new_node)
            return
        #last case remaining: leaf but label must be extended
        old_leaf = current_node
        other_label = current_node.points[0][0]
        index -= 1
        current_node = current_node.get_parent() # start from above node to descend the common label from the last common internal node
        while index < self.k:
            #parent = current_node.get_parent()
            if(label[index] == other_label[index]): # interpose internal node 
                intermediate_node = Node(current_node.get_depth() + 1, parent = current_node)
                current_node.set_child(label[index], intermediate_node)
                current_node = intermediate_node
            else: #found divergent label, fill both the current internal node's children
                new_leaf = Node(current_node.get_depth() + 1, parent = current_node)
                new_leaf.add_point(label, document_name)
                current_node.set_child(label[index], new_leaf)
                old_leaf.set_depth(current_node.get_depth() + 1)
                old_leaf.set_parent(current_node)
                current_node.set_child(other_label[index], old_leaf)

                return
            index += 1
        #we get here only if there's no difference between the reached leaf and the inserted document
        #so we just put the old leaf as placeholder for this path
        current_node = current_node.get_parent()
        current_node.set_child(label[-1], old_leaf)
        old_leaf.set_depth(index)   #should be self.k
        old_leaf.add_point(label, document_name = document_name)
        old_leaf.set_parent(current_node)

    def find_descendants(self, node):
        nodes = []
        nodes.append(node)
        points = []
        while len(nodes) > 0:
            current_explored_node = nodes.pop()
            if current_explored_node.points:
                points.extend([point[1] for point in current_explored_node.points])
            children = current_explored_node.get_children()
            if children[0] != None:
                nodes.append(children[0])
            if children[1] != None:
                nodes.append(children[1])
        return points
    
    def descend(self, query):
        label = self.minhash.index_entry(query)
        label = bits_from_int(label, self.k)
        current_node = self.root
        index = 0
        while index < self.k and current_node.get_children()[label[index]] != None: #descend until no child
            current_node = current_node.get_children()[label[index]]
            index = current_node.get_depth()
        return current_node

    def query(self, query, c, m): #async query, each tree thinks for itself
        current_node = self.descend(query)
        candidates = []
        limit = max(c, m)
        while len(set(candidates)) < limit and current_node.get_depth() > 0:
            candidates = self.find_descendants(current_node)
            current_node = current_node.get_parent()
        return list(set(candidates))[:limit]
    
    def query_sync(self, leaf_node, level): #sync query, level by level scan, so no limit to candidates retrieved
        current_node = leaf_node
        candidates =[]
        while current_node.get_depth() >= level: # iteratively going up preserves similarity ordering
            candidates = self.find_descendants(current_node)
            current_node = current_node.get_parent()
        return candidates
    
    def calculate_depth_stats(self):
        nodes_per_depths = {}
        points_per_depths = {}
        nodes = []
        nodes_stack = [self.root]
        for i in range(0, self.k + 1):
            nodes_per_depths[i] = 0
            points_per_depths[i] = 0
        
        while(nodes_stack): #BFS search (kind of)
            current_node = nodes_stack.pop(0)
            nodes.append(current_node)
            left = current_node.get_children()[0]
            right = current_node.get_children()[1]
            if(left != None): nodes_stack.append(left)
            if(right != None): nodes_stack.append(right)
        for node in nodes:
            points = self.find_descendants(node)
            nodes_per_depths[node.get_depth()] += 1
            points_per_depths[node.get_depth()] += len(points)
        
        return points_per_depths, nodes_per_depths