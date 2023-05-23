import sys
from network.salcap import *
from torch import optim
import torch
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import torch.nn as nn
import tqdm
from PIL import Image
import numpy as np
from evaluate_sal import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Caption(nn.Module):
    def __init__(self, hp, num_class = 104):
        super(Classification, self).__init__()

        #self.backbone_sketch = Resnet_Backbone()
        self.netcap = SalCap(hp.vocab_size)

        self.train_params = list(self.netcap.parameters())
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.loss = nn.CrossEntropyLoss()
        self.hp = hp
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()[None, ..., None, None]
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()[None, ..., None, None]

    def forward(self, img, captions, lengths):
        self.netcap.train()
        img = (img.cuda()-self.mean)/self.std
        targets = pack_padded_sequence(captions, lengths, batch_first=True)
        big_msk, msk, output = self.netcap(img, captions, lengths)
        big_msk = F.sigmoid(big_msk)
        loss = F.cross_entropy(outputs, targets.data) + \
                args.wr*F.binary_cross_entropy(F.sigmoid(msk), 
                        torch.zeros_like(msk, device=msk.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_sal(self, sal_val_loader, num):
        pathvrst = self.append_dir("val_output")
        self.eval()
        for i, data in enumerate(sal_val_loader):
            img, img_gt, WWs, HHs, names = data
            img = img.to(device)
            with torch.no_grad():

                msk_big, _, _ = self.netcap(img)
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
        maxfm, mae, _, _ = fm_and_mae(pathvrst, os.path.join(self.hp.root_dir_saliency, 'ground_truth_mask'), output_dir=None)
        print(f"val iteration {num} | FM {maxfm} | MAE {mae}")
        return maxfm, mae

    def freeze_weights(self, module):
        for name, x in module.named_parameters():
            x.requires_grad = False

    def Unfreeze_weights(self, module):
        for name, x in module.named_parameters():
            x.requires_grad = False

    def append_dir(self, name):
        pathappend = self.hp.saved_models + "/" + name
        if not os.path.exists(pathappend):
            os.mkdir(pathappend)
        return pathappend


