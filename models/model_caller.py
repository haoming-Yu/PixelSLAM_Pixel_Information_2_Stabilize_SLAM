import decoder as proto_decoder

"""
Here cfg is a dictionary to represent all the parameters.
As a consequence of having too much parameters to manage, 
a config file to manage all parameters will be a wise choice.
"""
def get_decoder(cfg):
    """
    A wapper function to get the decoder with proper argument
    """
    c_dim = cfg['model']['c_dim']
    pos_embedding_method = cfg['model']['pos_embedding_method']
    use_view_direction = cfg['model']['use_view_direction']
    decoder = proto_decoder.POINT(
        cfg=cfg, c_dim=c_dim,
        pos_embedding_method=pos_embedding_method, 
        use_view_direction=use_view_direction
    )
    return decoder