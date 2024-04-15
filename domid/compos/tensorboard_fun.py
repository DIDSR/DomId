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
    inject_tensor=None,
    other_info=None,
):
    if lr > 0:
        writer.add_scalar("learning rate", lr, epoch)
        writer.add_scalar("warmup", warmup_beta, epoch)
        if not pretraining_finished:
            writer.add_scalar("Pretraining", acc_tr, epoch)
            writer.add_scalar("Pretraining Loss", loss, epoch)
        else:
            writer.add_scalar("Training acc", acc_tr, epoch)
            writer.add_scalar("Loss", loss, epoch)

        if not pretraining_finished:
            name = "Output of the decoder pretraining"
        else:
            name = "Output of the decoder training"

        if other_info is not None and epoch > 3:
            kl_total, ce_total, re_total = other_info
            writer.add_scalar("KL", kl_total, epoch)
            writer.add_scalar("CE", ce_total, epoch)
            writer.add_scalar("RE", re_total, epoch)

        if inject_tensor is not None:
            preds, *_, x_pro = model.infer_d_v_2(tensor_x, inject_tensor)
        else:
            preds, *_, x_pro = model.infer_d_v_2(tensor_x)
        
        if len(x_pro.shape) < 3:
            x_pro = torch.reshape(x_pro, (x_pro.shape[0], tensor_x.shape[1], tensor_x.shape[2], tensor_x.shape[3]))
        
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
