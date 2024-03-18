import os
import shutil

from domid.compos.exp.exp_main import Exp


def experiment_train(args, save_path=None):

    exp = Exp(args)
    # exp.execute()
    exp.trainer.before_tr()
    exp.trainer.tr_epoch(0)
    # exp.trainer.post_tr()
    if not save_path is None:
        # This is used to move the saved model weights and all byproducts of the test run into a temporary
        # directory, so that the actual results directory will not be polluted with the byproduct files of the tests.
        # This is also used as a mechanism to reuse the same saved weights between different test functions.
        source_dir = exp.trainer.storage.ex_path
        dest_dir = save_path
        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(dest_dir, file_name)
            shutil.move(source_file, destination_file)
        if not os.listdir(source_dir):
            os.rmdir(source_dir)
        else:
            raise OSError(f"Trying to delete {source_dir}, but it is not empty.")
