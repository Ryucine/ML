import random
import math

TotalInput = 2
TotalHidden = 100

import pandas as pd

data = pd.read_excel('테트리스.xlsx') 
df = pd.DataFrame(data, columns=['APM', 'PPS', 'VS', 'TR'])
data = [[float(i[0])/10, float(i[1]), float(i[2])/50] for i in df.values]
labels = [float(i[3])/1000 for i in df.values]

testdata = pd.read_excel('테트리스 test case.xlsx') 
testdf = pd.DataFrame(testdata, columns=['APM', 'PPS', 'VS', 'TR'])
testdata = [[float(i[0])/10, float(i[1]), float(i[2])/50] for i in testdf.values]
testlabels = [float(i[3].replace('\xa0', ''))/1000 for i in testdf.values]

class InputNode():
    def __init__(self, value):
        self.value = value
    def getValue(self):
        return self.value

class HiddenNode():
    def __init__(self, weights, bias):
        self.weights, self.bias = weights, bias
    def CalcValue(self,PrevNodes):
        value = 0
        for i in range(len(PrevNodes)):
            value += PrevNodes[i].value*self.weights[i]
        value += self.bias
        return value
    def Update(self, dw, db, eta):
        for i in range(len(self.weights)):
            self.weights[i] += -eta*dw[i]
        self.bias += -eta*db

InputNodes = []
HiddenNodes = []
eta = 0.0001

def sigmoid(x):
    if x > 100:
        return 1
    elif x < -100:
        return 0
    return 1.0/(1.0+math.exp(-x))

for _ in range(TotalHidden):
    w = [random.uniform(-1,1) for _ in range(TotalInput)]
    b = random.uniform(-1,1)
    HiddenNodes.append(HiddenNode(w,b))

FinalWeights = [random.uniform(-1,1) for _ in range(TotalHidden)]
FinalBias = random.uniform(-1,1)

for _ in range(200):
    dFinalWeights = [0.0 for _ in FinalWeights]
    dFinalBias = 0.0
    dweights = [[0.0 for _ in range(TotalInput)] for _ in range(TotalHidden)]
    dbiases = [0.0 for _ in range(TotalHidden)]
    for idx in range(len(data)):
        InputNodes = [InputNode(data[idx][i]) for i in range(TotalInput)]

        zs = [Node.CalcValue(InputNodes) for Node in HiddenNodes]

        z = 0
        for i in range(TotalHidden):
            z += sigmoid(zs[i])*FinalWeights[i]
        z += FinalBias

        dz = 2*(z-labels[idx])
        dzs = [dz*FinalWeights[i]*sigmoid(zs[i])*(1-sigmoid(zs[i])) for i in range(TotalHidden)]
        for i in range(TotalHidden):
            for j in range(TotalInput):
                dweights[i][j] += dzs[i]*InputNodes[j].getValue()
            dbiases[i] += dzs[i]
        for i in range(len(dFinalWeights)):
            dFinalWeights[i] += dz*sigmoid(zs[i])
        dFinalBias += dz

    for i in range(TotalHidden):
        HiddenNodes[i].Update(dweights[i], dbiases[i], eta)
    for i in range(len(FinalWeights)):
        FinalWeights[i] += -eta*dFinalWeights[i]
    FinalBias += -eta*dFinalBias

    zValue = [0 for _ in range(len(testdata))]
    FinalzValue = [0 for _ in range(len(testdata))]
    for idx in range(len(testdata)):
        InputNodes = [InputNode(testdata[idx][i]) for i in range(TotalInput)]
        zs = [Node.CalcValue(InputNodes) for Node in HiddenNodes]

        z = 0
        for i in range(TotalHidden):
            z += sigmoid(zs[i])*FinalWeights[i]
        z += FinalBias
            
        zValue[idx] = z
    for idx in range(len(testdata)):
        FinalzValue[idx] = zValue[idx]

    loss = 0
    for i in range(len(FinalzValue)):
        loss += abs(FinalzValue[i]-testlabels[i])*1000

    print(loss/len(FinalzValue))

while True:
    APM, PPS, VS = map(float, input().split())
    test = [APM/10, PPS, VS/100]
    print(test)
    InputNodes = [InputNode(test[i]) for i in range(TotalInput)]

    zs = [Node.CalcValue(InputNodes) for Node in HiddenNodes]

    z = 0
    for i in range(TotalHidden):
        z += sigmoid(zs[i])*FinalWeights[i]
    z += FinalBias
            
    print('TR: ', z*1000)
