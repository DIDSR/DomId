"""
make an experiment
"""
from domid.arg_parser import mk_parser_main
from domid.compos.exp.exp_main import Exp


def mk_exp(
    task,
    model,
    trainer: str,
    train_domain: str,
    test_domain: str,
    batchsize: int,
    nocu=True,
    prior="Gaus",
    random_batching=False,
    model_method="cnn",
    feat_extract="vae",
    pre_tr=5,
    epos=10,
    inject_var=None,
    dim_inject=0,
    **kwargs
):
    """
    Creates a custom experiment. The user can specify the input parameters.

    Input Parameters:
        - task: create a task to a custom dataset by importing "mk_task_dset" function from
        "domainlab.tasks.task_dset". For more explanation on the input params refer to the
        documentation found in "domainlab.tasks.task_dset.py".
        - model: create a model [NameOfModel] by importing "mk_[NameOfModel]" function from
        "domainlab.models.model_[NameOfModel]". For a concrete example and explanation of the input
        params refer to the documentation found in "domainlab.models.model_[NameOfModel].py"
        - trainer: string,
        - test_domain: string,
        - batch size: int
        - **kwargs: any additional parameters that can be processed by the arg_parser of DomId or DomainLab

    Returns: experiment
    """
    str_arg = (
        f"--model={model} --trainer={trainer} --bs={batchsize} --task={task} "
        f"--prior={prior} --model_method={model_method} --feat_extract={feat_extract} "
        f"--pre_tr={pre_tr} --epos={epos} --d_dim={len(train_domain.split(' '))} "
    )
    if nocu:
        str_arg += " --nocu "
    str_arg += " --tr_d " + train_domain
    str_arg += " --te_d " + test_domain
    if random_batching:
        str_arg += " --random_batching "
    if inject_var:
        str_arg += f"--inject_var={inject_var} --dim_inject_y={dim_inject}"

    # Iterating over the Python kwargs dictionary
    for key, value in kwargs.items():
        if type(value) == bool:
            str_arg += f" --{key}"
        else:
            str_arg += f" --{key}={value} "

    parser = mk_parser_main()
    conf = parser.parse_args(str_arg.split())
    print(conf)
    exp = Exp(conf, task, model=model)
    return exp
