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
    pre_tr=5,
    epos=10,
    nocu=True,
    **kwargs,
):
    """
    Creates a custom experiment. The user can specify the input parameters.

    :param task: use a predefined task in DomId, or create a task to a custom dataset. For more explanation on the
    input params refer to the documentation found in "domainlab.tasks.task_dset.py".
    :param model: create a model [NameOfModel] by importing the appropriate "mk_[NameOfModel]". For a concrete example
    and explanation of the input params refer to the documentation found in "domainlab.models.model_[NameOfModel].py"
    :param string trainer: for instance, 'ae', 'cluster', 'sdcn'; see the available trainers under 'domid/trainers/'
    and 'domainlab/trainers/' or define your own trainer.
    :param string test_domain: enumerate which domains from the chosen 'task' to use for training (separated by spaces)
    :param string test_domain: enumerate which domains from the chosen 'task' to use for testing (separated by spaces)
    :param int batchsize: batch size to use for training
    :param \**kwargs: any additional parameters that can be processed by the arg_parser of DomId or DomainLab
    :return exp: the experiment object
    """
    str_arg = (
        f"--model={model} --trainer={trainer} --bs={batchsize} --task={task} "
        f"--pre_tr={pre_tr} --epos={epos} --d_dim={len(train_domain.split(' '))} "
    )
    if nocu:
        str_arg += " --nocu "
    str_arg += " --tr_d " + train_domain
    str_arg += " --te_d " + test_domain

    # Iterating over the Python kwargs dictionary
    for key, value in kwargs.items():
        if type(value) == bool:
            str_arg += f" --{key}"
        else:
            str_arg += f" --{key}={value}"

    parser = mk_parser_main()
    conf = parser.parse_args(str_arg.split())
    print(conf)
    exp = Exp(conf, task, model=model)
    return exp
