import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from models.rnn import RNNModel
from data_loading import *

def train(model, dataset, iterations, learning_rate=.01, batch_size=100):
    # prepare data
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # set expectations
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), learning_rate)
    # start training
    for epoch in range(iterations):
        running_loss = 0
        # iterate over batches
        for i_batch, batch in enumerate(dataset_loader):
            # separate batch into relevant parts
            questions, answers, images, _, _, _ = batch
            # choose top answer
            answers = answers[1].view(batch_size)
            # zero the parameter gradients
            optimizer.zero_grad()
            # iterate over questions
            for question, image, answer in zip(questions, images, answers):
                # prepare variables
                question = Variable(torch.LongTensor([i for i in range(len(question)) if question[i] == 1]))
                image = Variable(image)
                answer = Variable(torch.LongTensor([answer]))
                # perform forward pass
                model_answer = model(question, image)
                # calculate and backpropagate loss
                loss = criterion(model_answer, answer)
                loss.backward()
                optimizer.step()
    print('Training complete.')

if __name__ == "__main__":
    vec_collection = VQADataset(
        load_path="./data/vqa_vecs_test.pickle",
        image_features_path="./data/VQA_image_features.h5",
        image_features2id_path="./data/VQA_img_features2id.json",
        inflate_vecs=False
    )
    model = RNNModel(QUESTION_VOCAB_SIZE, IMAGE_FEATURE_SIZE, 256, ANSWER_VOCAB_SIZE)
    train(model, vec_collection, 1)
    torch.save(model, "models/debug")