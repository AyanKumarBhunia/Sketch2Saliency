from Baseline.network import SalCls
from evaluate_sal import *
import numpy as np
from PIL import Image
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import sys
from network.network import *
from torch import optim
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classification(nn.Module):
    def __init__(self, hp, num_class=104):
        super(Classification, self).__init__()

        #self.backbone_sketch = Resnet_Backbone()
        self.netc = SalCls(n_class=201)

        self.train_params = list(self.netc.parameters())
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.loss = nn.CrossEntropyLoss()
        self.hp = hp
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img, label):
        self.netc.train()
        big_msk, msk, output = self.netc(img)
        # loss = self.loss(output, label) + self.hp.wr*F.binary_cross_entropy(F.sigmoid(msk),
        # torch.zeros_like(msk, device=msk.device))
        loss = self.loss(output, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_sal(self, sal_val_loader, num):
        self.hp.root_dir_saliency = os.path.join(
            self.hp.base_dir, 'Dataset/ECSSD')
        pathvrst = self.append_dir("val_output")
        self.eval()
        for i, data in enumerate(sal_val_loader):
            img, img_gt, WWs, HHs, names = data
            img = img.to(device)
            with torch.no_grad():
                msk_big, _, _ = self.netc(img)
                msk_big = F.sigmoid(msk_big)

            msk_big = msk_big.squeeze(1)
            msk_big = msk_big.cpu().numpy() * 255
            for b, _msk in enumerate(msk_big):
                name = names[b]
                WW = WWs[b]
                HH = HHs[b]
                _msk = Image.fromarray(_msk.astype(np.uint8))
                _msk = _msk.resize((WW, HH))
                _msk.save(f"{pathvrst}/{name}.png")
        maxfm, mae, _, _ = fm_and_mae(pathvrst, os.path.join(
            self.hp.root_dir_saliency, 'ground_truth_mask'), output_dir=None)
        print(f"val iteration {num} | FM {maxfm} | MAE {mae}")
        return maxfm, mae

    def freeze_weights(self, module):
        for name, x in module.named_parameters():
            x.requires_grad = False

    def Unfreeze_weights(self, module):
        for name, x in module.named_parameters():
            x.requires_grad = False

    def append_dir(self, name):
        if not os.path.exists(self.hp.saved_models):
            os.mkdir(self.hp.saved_models)
        pathappend = self.hp.saved_models + "/" + name
        if not os.path.exists(pathappend):
            os.makedirs(pathappend)
        return pathappend

    def cls_evaluation(self, dataloader_Test, epoch):

        self.eval()
        correct = 0
        test_loss = 0
        start_time = time.time()

        for i_batch, (img, one_hot_label, label) in enumerate(dataloader_Test):
            img = img.to(device)

            big_msk, msk, output = self.netc(img)
            test_loss += self.loss(output, label.to(device)).item()
            prediction = output.argmax(dim=1, keepdim=True).to('cpu')
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(dataloader_Test.dataset)
        accuracy = 100. * correct / len(dataloader_Test.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time_Takes: {}\n'.format(
            test_loss, correct, len(dataloader_Test.dataset), accuracy, (time.time() - start_time)))

        return accuracy
