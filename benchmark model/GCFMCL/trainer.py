from time import time
import numpy as np
from torch.nn.utils.clip_grad import clip_grad_norm_
import os
from recbole.trainer import Trainer
import torch

class GCFMCLTrainer(torch.nn.Module):

    def __init__(self, args, model):
        super(GCFMCLTrainer, self).__init__()

        self.model = model
        self.num_m_step = args.m_step
        self.epochs = args.epochs
        self.data_path = args.data_path

        self.clip_grad_norm = {
            "max_norm": args.clip_max_norm,
            "norm_type": args.clip_norm_type,
        }


        assert self.num_m_step is not None

    def fit(self, train_data, dataset=None):
        self.model.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=5e-4
        )

        for epoch_idx in range(0 , self.epochs):

            if epoch_idx % self.num_m_step == 0:
                self.model.e_step()
            train_loss = self._train_epoch(train_data)
        
        embedding1, embedding2 = self.model.predict(dataset)
        np.savetxt(fname=f'mlp/data/{self.data_path}/embedding1.txt', X=embedding1, newline='\n', encoding='UTF-8')
        np.savetxt(fname=f'mlp/data/{self.data_path}/embedding2.txt', X=embedding2, newline='\n', encoding='UTF-8')
        #os.system('python mlp/test.py')

    def _train_epoch(self, train_data):
        self.model.train()
        self.model.optimizer.zero_grad()

        losses = self.model.calculate_loss(train_data)
        loss = sum(losses) if isinstance(losses, tuple) else losses

        assert not torch.isnan(loss), "Loss is NaN"

        loss.backward()
        if self.clip_grad_norm:
            clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
        self.model.optimizer.step()

        if isinstance(losses, tuple):
            return sum(l.item() for l in losses)
        return loss.item()
