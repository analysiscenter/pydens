""" Auxiliary functions for models """


def unpack_args(args, layer_no, layers_max):
    """ Return layer parameters """
    new_args = {}
    for arg in args:
        if isinstance(args[arg], list) and layers_max > 1:
            arg_value = args[arg][layer_no]
        else:
            arg_value = args[arg]
        new_args.update({arg: arg_value})
    return new_args


def unpack_fn_from_config(param, config=None):
    """ Return params from config """
    par = config.get(param)

    if par is None:
        return None, {}

    if isinstance(par, (tuple, list)):
        if len(par) == 0:
            par_name = None
        elif len(par) == 1:
            par_name, par_args = par[0], {}
        elif len(par) == 2:
            par_name, par_args = par
        else:
            par_name, par_args = par[0], par[1:]
    elif isinstance(par, dict):
        par = par.copy()
        par_name, par_args = par.pop('name', None), par
    else:
        par_name, par_args = par, {}

    return par_name, par_args
