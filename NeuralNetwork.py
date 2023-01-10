import random
import math

TotalInput = 2
TotalHidden = 200

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
    
data = [(5,10),(7,18),(10,8),(15,15),(20,3),(23,12)]
labels = [-1,-1,+1,+1,-1,-1]
InputNodes = []
HiddenNodes = []
eta = 0.0001

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

for _ in range(TotalHidden):
    w = [random.uniform(-1,1) for _ in range(TotalInput)]
    b = random.uniform(-1,1)
    HiddenNodes.append(HiddenNode(w,b))

FinalWeights = [random.uniform(-1,1) for _ in range(TotalHidden)]
FinalBias = random.uniform(-1,1)

for _ in range(1000):
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

print([(i.weights, i.bias) for i in HiddenNodes], FinalWeights, FinalBias)

zValue = [0 for _ in range(len(data))]
FinalzValue = [0 for _ in range(len(data))]
for count in range(10000):
    for idx in range(len(data)):
        InputNodes = [InputNode(data[idx][i]) for i in range(TotalInput)]

        zs = [Node.CalcValue(InputNodes) for Node in HiddenNodes]

        z = 0
        for i in range(TotalHidden):
            z += sigmoid(zs[i])*FinalWeights[i]
        z += FinalBias
        
        zValue[idx] += z
    print(zValue)
for idx in range(len(data)):
    FinalzValue[idx] = zValue[idx]/10000

print(FinalzValue)
