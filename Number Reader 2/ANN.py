import random as rd
import math
import numpy as np
import pickle



from keras.datasets import mnist

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_Y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_Y.shape))


train_X = train_X[:60000]
test_X = test_X[:10000]
train_Y = train_Y[:60000]
test_Y = test_Y[:10000]
train_X = train_X.reshape([-1,784])
test_X = test_X.reshape([-1,784])
L = []
for i in train_Y:
    L.append([int(i == j) for j in range(10)])
train_Y = np.array(L)
L = []
for i in test_Y:
    L.append([int(i == j) for j in range(10)])
test_Y = np.array(L)
train_X = train_X / 255
test_X = test_X / 255
print('Yay!')

pickup = True

def Activation(x):
    return (x+np.abs(x))/2
    return 1/(1+np.exp(-x))

def DiffAct(y):
    L = []
    for i in y:
        if i > 0:
            L.append(1.0)
        else:
            L.append(0.0)
    return np.array(L)
    return y*(1-y)

def Loss(Answer, Output):
    Output[Output >= 1] = 1-1e-4
    Output[Output <= 0] = 1e-4
    loss1 = -Answer*np.log(Output)
    loss2 = -(1-Answer)*np.log(1-Output)
    loss = loss1+loss2
    return (Answer-Output)**2

def DiffLoss(Answer, Output):
    Output[Output >= 1] = 1-1e-4
    Output[Output <= 0] = 1e-4
    diffloss1 = -Answer/Output 
    diffloss2 = (1-Answer)/(1-Output)
    diffloss = diffloss1+diffloss2
    return (Output-Answer)*2

LayerNodeCount = [784,70,30,10]

class Layer:
    def __init__(self, PrevNodes, NextNodes):
        self.PrevNodes, self.NextNodes = PrevNodes, NextNodes
        self.Weights, self.Biases = np.random.random((PrevNodes, NextNodes))/100, np.random.random(NextNodes)/100
        self.GradWeights, self.GradBiases = np.zeros((PrevNodes, NextNodes)), np.zeros((NextNodes))
    def CalculateOutput(self, CurrentLayerInput):
        self.Input = CurrentLayerInput
        self.Outputs = Activation(np.matmul(self.Input, self.Weights) + self.Biases)
        return self.Outputs
    def CalculateGrad(self, GradOutputs):
        GradActivation = DiffAct(self.Outputs)
        self.GradWeights += np.matmul(self.Input.reshape(-1,1), GradOutputs.reshape(1,-1))
        self.GradBiases += GradOutputs
        DiffNextLayerOfPrevLayer = self.Weights*GradActivation
        GradInputs = np.matmul(DiffNextLayerOfPrevLayer, GradOutputs.reshape(-1,1))
        return GradInputs.flatten()
    def Differentiate(self, eta):
        self.Weights += -self.GradWeights*eta
        self.Biases += -self.GradBiases*eta
        self.GradWeights, self.GradBiases = np.zeros((self.PrevNodes, self.NextNodes)), np.zeros((self.NextNodes))

import os.path

Layers = [Layer(LayerNodeCount[i], LayerNodeCount[i+1]) for i in range(len(LayerNodeCount)-1)]
if os.path.exists('vars6'):
    filename = 'vars6'
    infile = open(filename,'rb')
    Layers = pickle.load(infile)
    infile.close()
eta = 1e-7

def Compute(InputVars):
    temp = InputVars
    for layer in Layers:
        temp = layer.CalculateOutput(temp)
    return temp

from tkinter import *


class Cell():
    FILLED_COLOR_BG = "blue"
    EMPTY_COLOR_BG = "white"
    FILLED_COLOR_BORDER = "blue"
    EMPTY_COLOR_BORDER = "black"

    def __init__(self, master, x, y, size):
        """ Constructor of the object called by Cell(...) """
        self.master = master
        self.abs = x
        self.ord = y
        self.size= size
        self.fill = False

    def On(self):
        """ Switch if the cell is filled or not. """
        self.fill= True

    def Off(self):
        self.fill = False

    def draw(self):
        """ order to the cell to draw its representation on the canvas """
        if self.master != None :
            fill = Cell.FILLED_COLOR_BG
            outline = Cell.FILLED_COLOR_BORDER

            if not self.fill:
                fill = Cell.EMPTY_COLOR_BG
                outline = Cell.EMPTY_COLOR_BORDER

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size

            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill = fill, outline = outline)

class CellGrid(Canvas):
    def __init__(self,master, rowNumber, columnNumber, cellSize, *args, **kwargs):
        Canvas.__init__(self, master, width = cellSize * columnNumber , height = cellSize * rowNumber, *args, **kwargs)

        self.cellSize = cellSize

        self.state = np.zeros((28,28))
        self.pencil = 1

        self.grid = []
        for row in range(rowNumber):

            line = []
            for column in range(columnNumber):
                line.append(Cell(self, column, row, cellSize))

            self.grid.append(line)

        #memorize the cells that have been modified to avoid many switching of state during mouse motion.
        self.switched = []

        #bind click action
        self.bind("<Button-1>", self.handleMouseClick)  
        #bind moving while clicking
        self.bind("<B1-Motion>", self.handleMouseMotion)
        #bind release button action - clear the memory of midified cells.
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())

        self.draw()



    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()

    def _eventCoords(self, event):
        row = int(event.y / self.cellSize)
        column = int(event.x / self.cellSize)
        return row, column

    def handleMouseClick(self, event):
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]
        if cell.fill:
            cell.Off()
            cell.draw()
            self.pencil = 0
            self.state[row][column] = 0.0
        else:
            cell.On()
            cell.draw()
            self.pencil = 1
            self.state[row][column] = 1.0
        print(self.state)
        #add the cell to the list of cell switched during the click
        self.switched.append(cell)

    def handleMouseMotion(self, event):
        row, column = self._eventCoords(event)
        neighbors = [[row,column],[row+1,column],[row-1,column],[row,column+1],[row,column-1]]

        for coord in neighbors:
            cell = self.grid[coord[0]][coord[1]]
            if cell not in self.switched:
                if self.pencil:
                    cell.On()
                    self.state[coord[0]][coord[1]] = 1.0
                else:
                    cell.Off()
                    self.state[coord[0]][coord[1]] = 0.0
                cell.draw()
                self.switched.append(cell)
        print(np.argmax(Compute(self.state.flatten())))


if __name__ == "__main__" :
    app = Tk()

    grid = CellGrid(app, 28, 28, 20)
    grid.pack()

    app.mainloop()


'''while True:
    loss = 0
    accuracy = 0
    for i in range(len(train_X)):
        InputVars = train_X[i]
        #print(InputVars)
        OutputVars = train_Y[i]
        #print(OutputVars)
        Output = Compute(InputVars)
        loss += sum(Loss(OutputVars, Output))
        accuracy += int(np.argmax(Output) == np.argmax(OutputVars))
        #print(temp)
        #print(Loss)
        diffloss = DiffLoss(OutputVars, Output)
        #print(DiffLoss)
        temp = diffloss
        for layer in reversed(Layers):
            temp = layer.CalculateGrad(temp)
            #print(layer.GradWeights)
        if i % 60 == 60-1:
            for layer in Layers:
                # layer.Differentiate(eta)
                pass
    loss = 0
    accuracy = 0
    for i in range(len(test_X)):
        InputVars = test_X[i]
        OutputVars = test_Y[i]
        temp = Compute(InputVars)
        loss += sum(Loss(OutputVars, Output))
        accuracy += int(np.argmax(temp) == np.argmax(OutputVars))
        print(temp)
        print(np.argmax(temp), np.argmax(OutputVars))
    print(loss)
    print(accuracy/10000)'''

'''randseq = np.arange(60000)
    np.random.shuffle(randseq)
    newX = np.zeros([60000,784])
    newY = np.zeros([60000,10])
    for i in range(len(randseq)):
        newX[i] = train_X[randseq[i]]
        newY[i] = train_Y[randseq[i]]
    train_X = newX
    train_Y = newY'''

'''filename = 'newloss2'
    outfile = open(filename,'wb')
    pickle.dump(Layers,outfile)
    outfile.close()'''
    
    


