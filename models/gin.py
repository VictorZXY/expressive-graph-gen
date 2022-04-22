from torchdrug import models


class GIN_GCPN(models.GIN):
    """
    Graph Ismorphism Network proposed in `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        num_mlp_layer (int, optional): number of MLP layers
        eps (int, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim=None, hidden_dims=None, num_relation=None, edge_input_dim=None, num_mlp_layer=2, eps=0,
                 learn_eps=False, short_cut=False, batch_norm=False, activation='relu', concat_hidden=False,
                 readout='sum'):
        super(GIN_GCPN, self).__init__(input_dim, hidden_dims, edge_input_dim, num_mlp_layer, eps, learn_eps,
                                       short_cut, batch_norm, activation, concat_hidden, readout)

        self.num_relation = num_relation
