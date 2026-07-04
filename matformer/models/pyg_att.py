from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pydantic.typing import Literal
from torch import nn
from .utils import RBFExpansion
from features import angle_emb_mp
from torch_scatter import scatter
from .transformer import MatformerConv
import pdb

from torch import Tensor

from torch_geometric.nn.inits import glorot

from pydantic import BaseSettings as PydanticBaseSettings

from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
import pdb

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
from torch_geometric.typing import PairTensor
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_torch_csr_tensor,
    to_dense_adj,
    to_dense_batch,
    select,
    softmax,
    scatter,
)


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"
        
        
class MatformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["matformer"]
    conv_layers: int = 5
    edge_layers: int = 0
    atom_input_features: int = 92
    edge_features: int = 128
    triplet_input_features: int = 40
    node_features: int = 128
    fc_layers: int = 1
    fc_features: int = 128
    output_features: int = 1
    node_layer_head: int = 4
    edge_layer_head: int = 4
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False
    pre_train: bool = False
    position_noise: float = None
    lattice_noise: float = None
    mask_ratio: float = None
    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"

# Node-level Prompt
class NodePrompt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(NodePrompt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
        
class Matformer(nn.Module):
    """att pyg implementation."""

    def __init__(self, config=None):
        """Set up att modules."""
        super().__init__()
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        
        self.advanced_prompt_node = NodePrompt(p_num=10,in_channels=config.node_features)  # Node Prompt
        self.advanced_prompt_graph = nn.Embedding(num_embeddings=7,embedding_dim=config.node_features) # Graph Prompt
        
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
            nn.Linear(config.node_features, config.node_features),
        )
        self.att_layers = nn.ModuleList(
            [
                MatformerConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )
        
        self.fc = nn.Sequential(
                nn.Linear(128, config.fc_features), nn.SiLU()
            )
        
        self.fc_out = nn.Linear(
                config.fc_features, 1
            )
                
    def forward(self, data) -> torch.Tensor:
        data, ldata,_ = data
        # initial node features: atom feature network...
        collect_dict = {}
        
        node_features = self.atom_embedding(data.x)
        
        node_features = self.advanced_prompt_node.add(node_features)
        
        edge_feat = torch.norm(data.edge_attr, dim=1)
        
        edge_features = self.rbf(edge_feat)
        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[4](node_features, data.edge_index, edge_features)
        
        features = scatter(node_features, data.batch, dim=0, reduce="mean")

        space_grp = self.advanced_prompt_graph(data.label_attr)
                
        features = features + space_grp
        
        features = self.fc(features)
        out = self.fc_out(features)
        
        return torch.squeeze(out)