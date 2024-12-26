import torch
from tqdm import tqdm
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_one_epoch(model, diceloss, celoss, optimizer, dataloader, device, epoch, arg):
    model.train()
    diceloss = diceloss.to(device)
    celoss = celoss.to(device)

    celoss_sum = 0
    diceloss_sum = 0
    loss_sum = 0
    iteration = 0

    with tqdm(enumerate(dataloader), total=len(dataloader)) as loop:
        for i, batch in loop:
            image = batch[0]
            label = batch[1]

            dice_label = label.unsqueeze(1)

            image = image.to(device)
            label = label.to(device)
            dice_label = dice_label.to(device)

            out_put = model(image)

            Diceloss = diceloss(out_put, dice_label)
            CEloss = celoss(out_put, label)
            loss = Diceloss + CEloss

            loss_sum += loss.item()
            diceloss_sum += Diceloss.item()
            celoss_sum += CEloss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr']
                             , total_loss=loss_sum / iteration, CEloss=celoss_sum / iteration,
                             Diceloss=diceloss_sum / iteration)

    torch.save(model, os.path.join(arg.result_dir, 'model_epoch{}.pth'.format(epoch)))
    adjust_learning_rate_poly(optimizer=optimizer, epoch=epoch, num_epochs=arg.epochs, base_lr=arg.lr, power=2)

    return loss_sum / iteration


def validation(model_path, save_dir, device, img_dir, input_size):
    unloader = transforms.ToPILImage()
    device = device
    model = torch.load(model_path)
    model.to(device)
    model.eval()


    img_dir = img_dir
    result_dir = save_dir

    img_list = os.listdir(img_dir)
    for name in img_list:
        img = Image.open(os.path.join(img_dir, name)).convert('RGB')

        orininal_h = np.array(img).shape[0]
        orininal_w = np.array(img).shape[1]
        img = img.resize(size=input_size, resample=Image.Resampling.BICUBIC)
        img = F.to_tensor(img)
        # image = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = img.unsqueeze(0)
        image = image.to(device)

        out = model(image)
        out_soft = torch.nn.functional.softmax(out, dim=1)
        outmax = torch.argmax(out_soft, dim=1)
        # print(torch.unique(outmax))
        # outmax[outmax == 1] = 255
        outmax = outmax / 255.0
        out_last = outmax.to(torch.float32)

        result = unloader(out_last)
        result = result.convert('L')
        result = result.resize(size=(orininal_w, orininal_h), resample=Image.Resampling.NEAREST)
        result_name = name.split('.')[0]
        result.save(os.path.join(result_dir, result_name + '.png'))


def eval(model_path, dataloader, device, diceloss, celoss):
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    iteration = 0

    diceloss = diceloss.to(device)
    celoss = celoss.to(device)
    val_celoss_sum = 0
    val_diceloss_sum = 0
    val_loss_sum = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image = batch[0]
            label = batch[1]
            dicelabel = label.unsqueeze(1)

            image = image.to(device)
            label = label.to(device)
            dicelabel = dicelabel.to(device)

            out = model(image)
            val_diceloss = diceloss(out, dicelabel)
            val_celoss = celoss(out, label)

            iteration += 1
            val_celoss_sum += val_celoss.item()
            val_diceloss_sum += val_diceloss.item()
            val_loss_sum = val_celoss_sum + val_diceloss_sum
    return val_loss_sum / iteration, val_celoss_sum / iteration, val_diceloss_sum / iteration



def save_checkpoint(model, optim, epoch, save_fre, checkpoint_dir):
    if epoch % save_fre == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
            },
            os.path.join(checkpoint_dir, 'checkpoint_epoch{}.pth'.format(epoch))
        )
