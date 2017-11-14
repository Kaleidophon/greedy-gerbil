import torch.nn as nn
import torch
import torch.optim as optim


class BoWModel(nn.Module):
    def __init__(self, vocabSize, wordFeaturesDim, imageFeaturesDim, outputDim):
        super(BoWModel,self).__init__()
        self.wordFeaturesLayer = nn.Embedding(vocabSize, wordFeaturesDim)
        self.bias = nn.Parameter(torch.zeros(wordFeaturesDim))

        self.linearLayer = nn.Linear(imageFeaturesDim + wordFeaturesDim,outputDim)
        self.softMax = nn.Softmax(outputDim)


    def forward(self, wordInputs, imageFeatures):
        wordFeatures = self.wordFeaturesLayer(wordInputs) + self.bias
        features = torch.cat((wordFeatures,imageFeatures),1)

        linearOut = self.linearLayer(features)
        softMaxOut = self.softMax(linearOut)

        return softMaxOut



def train(model: BoWModel, iterations, learningRate = 0.01, cuda = False):
    if cuda:
        trainingModel = model.cuda()
    else:
        trainingModel = model


    optimizer = optim.SGD(trainingModel.parameters(),lr = learningRate)
