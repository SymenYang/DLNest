import torch

def experience(self):
    valLoader = self.dataset.testLoader
    acc = 0
    total = 0
    for _iter,data in enumerate(valLoader):
        x,y = data
        if self.runner._status.env != "CPU":
            x,y = x.cuda(),y.cuda()
        with torch.no_grad():
            output = self.runner.model(x)
            _,pred = torch.max(output,1)
            acc += sum(pred == y).item()
            total += x.shape[0]
    print("correct count:",acc,"total count:",total,"accuracy:",acc / total)
