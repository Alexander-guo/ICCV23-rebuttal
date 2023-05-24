from sklearn.neighbors import radius_neighbors_graph
import numpy as np
import time
from torch_geometric.data import Data as pyg_Data
from evimo1_loader import EVIMO_Sequence
import torch

if __name__ == '__main__':
    num_points = 23073
    slice_width = 0.02  # [s]
    t_upscale = 200
    edge_radius = 10    # [px]

    # y = np.random.randint(260, size=num_points)
    # x = np.random.randint(346, size=num_points)
    # t = np.sort(np.random.rand(num_points)) * slice_width
    # t *= t_upscale
    
    # events_slice = np.stack([x, y, t], axis=-1)
    # print(events_slice)
    
    # time1 = time.time()
    # rng = radius_neighbors_graph(events_slice, radius=50., mode='distance', include_self=False, n_jobs=-1)

    # time2 = time.time()
    # print(f"Time to build up graph with slice width {slice_width}s: {time2 - time1}s")
    # rng_array = rng.toarray()
    # print(rng_array)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    evimo1_seq = EVIMO_Sequence(, dt=slice_width)   //TODO: seq_path

    for i, events_slice in enumerate(evimo1_seq):
        ## build up graph from events slice
        time1 = time.time()
        node_features = events_slice['events_list'].to(device)     # [x, y, t, p]
        node_features[:, 2] *= t_upscale
        node_coords = node_features[:, :2]
        edge_index = list()
        for j, node in enumerate(node_coords):
            distance = (node_coords - node[None, ...]).pow(2).sum(-1).sqrt()
            connected_nodes = torch.where(distance <= edge_radius)
            connected_nodes = connected_nodes[torch.where(node_features[connected_nodes, 2] >= node_features[j, 2])]    # time constraints (greater than)
            edge_idx = torch.stack([[j, n] for n in connected_nodes if n != j], dim=-1)
            edge_index.append(edge_idx)
        edge_index = torch.concat(edge_index, dim=-1, dtype=torch.long)

        events_graph = pyg_Data(x=node_features, edge_index=edge_index, device=device)     # events slice graph
        time2 = time.time()
        