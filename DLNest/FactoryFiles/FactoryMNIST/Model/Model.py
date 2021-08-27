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
        pass
    
    def validationInit(self):
        self.totalCorrect = 0
        self.total = 0
    
    def validateABatch(self,data, iter : int):
        x,y = data
        with torch.no_grad():
            output = self.model(x)
            _,pred = torch.max(output, 1)
            correct = (pred == y).sum() / y.shape[0]
            correct = self._reduceMean(correct)
            self.totalCorrect += correct
            self.total += 1
    
    def validationAnalyze(self, log : dict):
        acc = self.totalCorrect / self.total
        log["acc"].append(acc.item())