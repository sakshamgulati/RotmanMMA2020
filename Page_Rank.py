# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:14:48 2020

@author: saksh
"""

import networkx as nx
import numpy as np

def create_toy_network():
    G = nx.DiGraph()
    edges = [('9','7'), ('8','7'), ('8','6'), ('8','5'), ('7','6'), ('7','5'), ('6','5'), ('6','4'), ('5','4'), ('4','3'), ('4','1'), ('3','2'), ('3','1'), ('2','1')]
    G.add_edges_from(edges)
    G.graph
    return G

G = create_toy_network()

neighbor = {}
def neighbor_node(G):
    for item in sorted(list(G.nodes())):
        #print(node)
        neighbor[item] = len(list(G.neighbors(item)))
    return neighbor

init_mat=[]
nodes= len(G)
init_prob=1/nodes

for item in range(nodes):
    init_mat.append(init_prob)
    
def transition_matrix(G, nodes=nodes, neighbor_node=neighbor_node(G)):
    fin = []
    temp = []
    for i in range(1, nodes+1):
        for j in range(1, nodes+1):
            if(G.has_edge(str(i), str(j))):
                temp.append(round(1 / neighbor_node[str(i)], 2))
            else:
                temp.append(0)
        fin.append(temp)
        temp = []
    return np.array(fin).transpose()

def matrix_mul(transition,weight_prob):
    mat=np.matmul(transition,weight_prob)
    return mat

def pagerank(G,maxruns=1000,tau=0.01):
    page_rank_dict={}
    page_rank_dict[0]=init_mat
    for item in range(1,maxruns):
        page_rank_dict[item]=matrix_mul(transition_matrix(G),page_rank_dict[item-1])
        if all((page_rank_dict[item]-page_rank_dict[item-1])<tau):
            return page_rank_dict
            break
        else:
            continue

answer=pagerank(G)            
print(answer)



