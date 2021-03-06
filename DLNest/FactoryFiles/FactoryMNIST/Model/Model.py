from DLNest.Common.ModelBaseTorch import ModelBaseTorch

import torch
import torch.nn as nn
from MNISTCNN import MNISTModel

class Model(ModelBaseTorch):
    def init(self,args : dict,datasetInfo : dict = None):
        model = MNISTModel(args)
        self.model = self.register(model,syncBN=True) # But no BN
        cost = nn.CrossEntropyLoss()
        self.cost = self.register(cost)

    def initLog(self):
        return {
            "loss" : [],
            "acc" : [],
        }

    def initOptimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def runOneStep(self,data, log : dict, iter : int, epoch : int):
        self.model.zero_grad()
        x,y = data
        pred = self.model(x)
        loss = self.cost(pred,y)
        loss.backward()
        self.optimizer.step()

        loss = self._reduceMean(loss)
        log["loss"].append(loss.detach().item())

    def visualize(self,epoch : int, iter : int, log : dict):
        if iter % 10 == 0 and iter != 0:
            print("iter",iter,"loss:",sum(log["loss"][-10:]))

    def validate(self,valLoader,log : dict):
        totalCorrect = 0
        total = 0
        for _iter,data in enumerate(valLoader):
            x,y = data
            if self._envType != "CPU":
                x = x.cuda()
                y = y.cuda()
            with torch.no_grad():
                output = self.model(x)
                _,pred = torch.max(output, 1)
                correct = (pred == y).sum() / y.shape[0]
                correct = self._reduceMean(correct)
                totalCorrect += correct
                total += 1
        acc = totalCorrect / total
        log["acc"].append(acc.item())
        print(acc.item())
        print("validated")