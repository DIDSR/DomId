import tensorboardX
import torch


def tensorboard_write(
    writer,
    model,
    epoch,
    lr,
    warmup_beta,
    acc_tr,
    loss,
    pretraining_finished,
    tensor_x,
    inject_tensor,
):
    writer.add_scalar("learning rate", lr, epoch)
    writer.add_scalar("warmup", warmup_beta, epoch)
    if not pretraining_finished:
        writer.add_scalar("Pretraining", acc_tr, epoch)
        writer.add_scalar("Pretraining Loss", loss, epoch)
    else:
        writer.add_scalar("Training acc", acc_tr, epoch)
        writer.add_scalar("Loss", loss, epoch)

    preds, z_mu, z, _, _, x_pro = model.infer_d_v_2(tensor_x, inject_tensor)
    name = "Output of the decoder" + str(epoch)
    imgs = torch.cat(
        (
            tensor_x[0:8, :, :, :],
            x_pro[0:8, :, :, :],
        ),
        0,
    )
    # mse = torch.nn.MSELoss()#(dim=1, eps=1e-08)
    # sample1 = tensor_x[0, :, :, :].flatten().unsqueeze(0)
    # sample2 = x_pro[0, :, :, :].flatten().unsqueeze(0)
    # # acc_ = torch.mean(torch.abs(sample1-sample2)/sample1)

    # print('SIMILARITY', acc_)
    writer.add_images(name, imgs, epoch)
