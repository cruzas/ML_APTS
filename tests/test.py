import re
import ast
import networkx as nx
import matplotlib.pyplot as plt

# The provided data string
data_str = "{'sd0': {'ranks': [0, 1, 2, 3, 4, 5, 6, 7], 'r0': {'ranks': [0, 1, 2, 3], 's0': {'ranks': [0, 1], 'sh0': {'global_ranks': ([0, 4, 8, 12],), 'local_ranks': ([0, 4],), 'shard_ranks': [0, 1], 'global_group': None, 'local_group': None, 'shard_group': None}, 'sh1': {'global_ranks': ([1, 5, 9, 13],), 'local_ranks': ([1, 5],), 'shard_ranks': [0, 1], 'global_group': None, 'local_group': None, 'shard_group': None}}, 's1': {'ranks': [2, 3], 'sh0': {'global_ranks': ([2, 6, 10, 14],), 'local_ranks': ([2, 6],), 'shard_ranks': [2, 3], 'global_group': None, 'local_group': None, 'shard_group': None}, 'sh1': {'global_ranks': ([3, 7, 11, 15],), 'local_ranks': ([3, 7],), 'shard_ranks': [2, 3], 'global_group': None, 'local_group': None, 'shard_group': None}}}, 'r1': {'ranks': [4, 5, 6, 7], 's0': {'ranks': [4, 5], 'sh0': {'global_ranks': ([4, 8, 12, 16],), 'local_ranks': ([4, 8],), 'shard_ranks': [4, 5], 'global_group': None, 'local_group': None, 'shard_group': None}, 'sh1': {'global_ranks': ([5, 9, 13, 17],), 'local_ranks': ([5, 9],), 'shard_ranks': [4, 5], 'global_group': None, 'local_group': None, 'shard_group': None}}, 's1': {'ranks': [6, 7], 'sh0': {'global_ranks': ([6, 10, 14, 18],), 'local_ranks': ([6, 10],), 'shard_ranks': [6, 7], 'global_group': None, 'local_group': None, 'shard_group': None}, 'sh1': {'global_ranks': ([7, 11, 15, 19],), 'local_ranks': ([7, 11],), 'shard_ranks': [6, 7], 'global_group': None, 'local_group': None, 'shard_group': None}}}}, 'sd1': {'ranks': [8, 9, 10, 11, 12, 13, 14, 15], 'r0': {'ranks': [8, 9, 10, 11], 's0': {'ranks': [8, 9], 'sh0': {'global_ranks': ([8, 12, 16, 20],), 'local_ranks': ([8, 12],), 'shard_ranks': [8, 9], 'global_group': None, 'local_group': None, 'shard_group': None}, 'sh1': {'global_ranks': ([9, 13, 17, 21],), 'local_ranks': ([9, 13],), 'shard_ranks': [8, 9], 'global_group': None, 'local_group': None, 'shard_group': None}}, 's1': {'ranks': [10, 11], 'sh0': {'global_ranks': ([10, 14, 18, 22],), 'local_ranks': ([10, 14],), 'shard_ranks': [10, 11], 'global_group': None, 'local_group': None, 'shard_group': None}, 'sh1': {'global_ranks': ([11, 15, 19, 23],), 'local_ranks': ([11, 15],), 'shard_ranks': [10, 11], 'global_group': None, 'local_group': None, 'shard_group': None}}}, 'r1': {'ranks': [12, 13, 14, 15], 's0': {'ranks': [12, 13], 'sh0': {'global_ranks': ([12, 16, 20, 24],), 'local_ranks': ([12, 16],), 'shard_ranks': [12, 13], 'global_group': None, 'local_group': None, 'shard_group': None}, 'sh1': {'global_ranks': ([13, 17, 21, 25],), 'local_ranks': ([13, 17],), 'shard_ranks': [12, 13], 'global_group': None, 'local_group': None, 'shard_group': None}}, 's1': {'ranks': [14, 15], 'sh0': {'global_ranks': ([14, 18, 22, 26],), 'local_ranks': ([14, 18],), 'shard_ranks': [14, 15], 'global_group': None, 'local_group': None, 'shard_group': None}, 'sh1': {'global_ranks': ([15, 19, 23, 27],), 'local_ranks': ([15, 19],), 'shard_ranks': [14, 15], 'global_group': None, 'local_group': None, 'shard_group': None}}}}}"

# Step 1: Clean the data string by replacing invalid entries with 'None'
data_str = re.sub(r'<[^>]+>', 'None', data_str)

# Step 2: Correct tuple/list representations
data_str = data_str.replace('([', '[').replace('],)', ']')

# Parse the string into a Python dictionary
data = ast.literal_eval(data_str)
G = nx.DiGraph()

def traverse_dict(d, parent_label=''):
    if not isinstance(d, dict):
        return

    # Get ranks if present
    ranks = d.get('ranks', None)

    # Current node label
    node_label = parent_label

    # Add the node if it doesn't exist
    if node_label not in G:
        if ranks is not None:
            G.add_node(node_label, ranks=ranks)
        else:
            G.add_node(node_label)

    # Iterate over the items in the dictionary
    for key, value in d.items():
        if key != 'ranks':
            # Construct the child label
            child_label = f"{node_label}/{key}" if node_label else key
            # Add an edge from parent to child
            G.add_edge(node_label, child_label)
            # Recursively traverse the child dictionary
            traverse_dict(value, parent_label=child_label)
        else:
            # 'ranks' already processed
            pass

# Start traversing from the root nodes
for key in data:
    traverse_dict(data[key], parent_label=key)

# Function to handle multiple root nodes
def hierarchy_pos_multi_root(G, roots, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    ''' Positions nodes in a hierarchy layout for multiple root nodes.

    G: the graph (must be a tree or forest)
    roots: list of root nodes
    width: horizontal space allocated for the graph
    vert_gap: gap between levels of hierarchy
    '''
    if not isinstance(roots, list):
        roots = [roots]
    pos = {}
    if len(roots) == 1:
        pos = hierarchy_pos(G, roots[0], width=width, vert_gap=vert_gap, vert_loc=vert_loc, xcenter=xcenter, pos=pos)
    else:
        # Compute positions for each root separately
        dx = width / len(roots)
        nextx = xcenter - width / 2 - dx / 2
        for root in roots:
            nextx += dx
            pos = hierarchy_pos(G, root, width=dx, vert_gap=vert_gap, vert_loc=vert_loc, xcenter=nextx, pos=pos)
    return pos

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    ''' Positions nodes in a hierarchy layout.

    G: the graph (must be a tree)
    root: the root node
    width: horizontal space allocated for this branch - avoids overlap
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    pos: a dict saying where all nodes go if they have been assigned
    parent: parent of this branch
    '''
    if pos is None:
        pos = {}
    pos[root] = (xcenter, vert_loc)
    neighbors = list(G.neighbors(root))
    if parent:
        neighbors.remove(parent)
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos = hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos

# Get the list of root nodes (e.g., 'sd0', 'sd1', etc.)
roots = [key for key in data.keys()]

# Get positions using the custom hierarchy_pos_multi_root function
pos = hierarchy_pos_multi_root(G, roots)

# Prepare labels for nodes
labels = {}
for node in G.nodes():
    node_name = node.split('/')[-1]  # Get the last part of the path
    ranks = G.nodes[node].get('ranks', None)
    if ranks is not None:
        label = f"{node_name}\nRanks: {ranks}"
    else:
        label = f"{node_name}"
    labels[node] = label

# Plot the graph
plt.figure(figsize=(15, 10))
nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color='skyblue', font_size=8, arrows=False)
plt.title("Neural Network Structure Visualization")
plt.show()