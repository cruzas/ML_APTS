nn_structure_example = {
    "sd0": {  # Subdomain 0
        "r0": {  # Replica 0
            "ranks": [0, 1, 2],
            "s0": {  # Stage 0
                "ranks": [0],
                "sh0": {
                    "global_ranks": [0, 3, 6, 9],
                    "local_ranks": [0, 3],
                    "shard_ranks": [0],
                    "global_group": "dist_group_for_stage_0_global",
                    "local_group": "dist_group_for_stage_0_local"
                }
            },
            "s1": {  # Stage 1
                "ranks": [1],
                "sh0": {
                    "global_ranks": [1, 4, 7, 10],
                    "local_ranks": [1, 4],
                    "shard_ranks": [1],
                    "global_group": "dist_group_for_stage_1_global",
                    "local_group": "dist_group_for_stage_1_local"
                }
            },
            "s2": {  # Stage 2
                "ranks": [2],
                "sh0": {
                    "global_ranks": [2, 5, 8, 11],
                    "local_ranks": [2, 5],
                    "shard_ranks": [2],
                    "global_group": "dist_group_for_stage_2_global",
                    "local_group": "dist_group_for_stage_2_local"
                }
            }
        },
        "r1": {  # Replica 1
            "ranks": [3, 4, 5],
            "s0": {
                "ranks": [3],
                "sh0": {
                    "global_ranks": [0, 3, 6, 9],
                    "local_ranks": [0, 3],
                    "shard_ranks": [3],
                    "global_group": "dist_group_for_stage_0_global",
                    "local_group": "dist_group_for_stage_0_local"
                }
            },
            "s1": {
                "ranks": [4],
                "sh0": {
                    "global_ranks": [1, 4, 7, 10],
                    "local_ranks": [1, 4],
                    "shard_ranks": [4],
                    "global_group": "dist_group_for_stage_1_global",
                    "local_group": "dist_group_for_stage_1_local"
                }
            },
            "s2": {
                "ranks": [5],
                "sh0": {
                    "global_ranks": [2, 5, 8, 11],
                    "local_ranks": [2, 5],
                    "shard_ranks": [5],
                    "global_group": "dist_group_for_stage_2_global",
                    "local_group": "dist_group_for_stage_2_local"
                }
            }
        }
    },
    "sd1": {  # Subdomain 1
        "r0": {  # Replica 0
            "ranks": [6, 7, 8],
            "s0": {
                "ranks": [6],
                "sh0": {
                    "global_ranks": [0, 3, 6, 9],
                    "local_ranks": [6, 9],
                    "shard_ranks": [6],
                    "global_group": "dist_group_for_stage_0_global",
                    "local_group": "dist_group_for_stage_0_local"
                }
            },
            "s1": {
                "ranks": [7],
                "sh0": {
                    "global_ranks": [1, 4, 7, 10],
                    "local_ranks": [7, 10],
                    "shard_ranks": [7],
                    "global_group": "dist_group_for_stage_1_global",
                    "local_group": "dist_group_for_stage_1_local"
                }
            },
            "s2": {
                "ranks": [8],
                "sh0": {
                    "global_ranks": [2, 5, 8, 11],
                    "local_ranks": [8, 11],
                    "shard_ranks": [8],
                    "global_group": "dist_group_for_stage_2_global",
                    "local_group": "dist_group_for_stage_2_local"
                }
            }
        },
        "r1": {  # Replica 1
            "ranks": [9, 10, 11],
            "s0": {
                "ranks": [9],
                "sh0": {
                    "global_ranks": [0, 3, 6, 9],
                    "local_ranks": [6, 9],
                    "shard_ranks": [9],
                    "global_group": "dist_group_for_stage_0_global",
                    "local_group": "dist_group_for_stage_0_local"
                }
            },
            "s1": {
                "ranks": [10],
                "sh0": {
                    "global_ranks": [1, 4, 7, 10],
                    "local_ranks": [7, 10],
                    "shard_ranks": [10],
                    "global_group": "dist_group_for_stage_1_global",
                    "local_group": "dist_group_for_stage_1_local"
                }
            },
            "s2": {
                "ranks": [11],
                "sh0": {
                    "global_ranks": [2, 5, 8, 11],
                    "local_ranks": [8, 11],
                    "shard_ranks": [11],
                    "global_group": "dist_group_for_stage_2_global",
                    "local_group": "dist_group_for_stage_2_local"
                }
            }
        }
    }
}

import matplotlib.pyplot as plt
import networkx as nx

def plot_nn_structure_hierarchical(nn_structure):
    G = nx.DiGraph()  # Create a directed graph

    # Add nodes and edges to represent the hierarchy
    for sd_key, sd_value in nn_structure.items():
        G.add_node(sd_key)
        for replica_key, replica_value in sd_value.items():
            if replica_key != "ranks":  # Ignore the "ranks" key and plot only the hierarchy
                G.add_node(f"{sd_key}_{replica_key}")
                G.add_edge(sd_key, f"{sd_key}_{replica_key}")
                for stage_key, stage_value in replica_value.items():
                    if stage_key != "ranks":
                        G.add_node(f"{sd_key}_{replica_key}_{stage_key}")
                        G.add_edge(f"{sd_key}_{replica_key}", f"{sd_key}_{replica_key}_{stage_key}")
                        for shard_key, shard_value in stage_value.items():
                            if shard_key != "ranks":
                                G.add_node(f"{sd_key}_{replica_key}_{stage_key}_{shard_key}")
                                G.add_edge(f"{sd_key}_{replica_key}_{stage_key}", f"{sd_key}_{replica_key}_{stage_key}_{shard_key}")

    # Use Graphviz's hierarchical layout for top-down representation
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=False)

    plt.title("Hierarchical Structure of Neural Network (Top-Down)", fontsize=15)
    plt.show()

# Example usage with nn_structure_example
plot_nn_structure_hierarchical(nn_structure_example)
