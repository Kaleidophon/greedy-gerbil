import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from models.rnn import RNNModel
from data_loading import *

def test(model, dataset, cuda=False):
    dataset_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4, drop_last=True)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    correct = 0
    loss = 0.
    for i_batch, batch in enumerate(dataset_loader):
        questions, answers, images, _, _, _ = batch
        answers = answers[0]
        for question, image, answer in zip(questions, images, answers):
            # prepare variables
            question = Variable(torch.LongTensor(list(filter((QUESTION_VOCAB_SIZE).__ne__, question))))
            answer = Variable(torch.LongTensor([answer]))
            image = Variable(image)
            if cuda:
                question = question.cuda()
                image = image.cuda()
                answer = answer.cuda()

            # perform forward pass
            model_answer = model(question, image)
            m = torch.max(model_answer.cpu(), 1)[1]
            m = m.data.numpy()
            loss += criterion(model_answer, answer)
            if m[0] == answer.cpu().data.numpy()[0]:
                correct += 1
    print(loss.data[0]/len(dataset),correct/len(dataset))


def train(model, dataset, valid_set, iterations, batch_size=100, cuda=False):
    # prepare data
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # set expectations
    if cuda:
        model = model.cuda()

    lr_embed = 0.1
    lr_other = 1e-3 * 5
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    optimizer = optim.Adagrad([
        {'params': model.layer_transform.parameters()},
        {'params': model.gru.parameters()},
        {'params': model.embedding.parameters(), 'lr': lr_embed}
    ], lr=lr_other)

    # start training
    for epoch in range(iterations):
        # iterate over batches
        for i_batch, batch in enumerate(dataset_loader):
            #print("training in epoch", epoch+1, "on batch", i_batch+1)
            loss = 0.
            # separate batch into relevant parts
            questions, answers, images, _, _, _ = batch
            # choose top answer
            answers = answers[0]
            # zero the parameter gradients
            optimizer.zero_grad()
            # iterate over questions
            for question, image, answer in zip(questions, images, answers):
                # prepare variables
                question = Variable(torch.LongTensor(list(filter((QUESTION_VOCAB_SIZE).__ne__, question))))
                answer = Variable(torch.LongTensor([answer]))
                image = Variable(image)
                if cuda:
                    question = question.cuda()
                    image = image.cuda()
                    answer = answer.cuda()

                # perform forward pass
                model_answer = model(question, image)
                # calculate and backpropagate loss
                loss += criterion(model_answer, answer)
            #print("loss on this batch:", loss.data[0]/batch_size)
            loss /= batch_size
            loss.backward()
            optimizer.step()
        test(model, valid_set, True)


    print('Training complete.')

if __name__ == "__main__":
    vec_collection = VQADataset(
        load_path="../data/vqa_vecs_test.pickle",
        image_features_path="../data/VQA_image_features.h5",
        image_features2id_path="../data/VQA_img_features2id.json",
        inflate_vecs=False
    )

    vec_valid = VQADataset(
        load_path="../data/vqa_vecs_valid.pickle",
        image_features_path="../data/VQA_image_features.h5",
        image_features2id_path="../data/VQA_img_features2id.json",
        inflate_vecs=False
    )
    #This line is mysterious but prevents mysterious errors from cudnn (and took 2 hours of my sleep)
    torch.backends.cudnn.enabled = False

    model = RNNModel(vec_collection.question_dim, IMAGE_FEATURE_SIZE, 256, vec_collection.answer_dim, cuda_enabled=False)
    train(model, vec_collection, vec_valid, 100, cuda=False)
    torch.save(model, "../models/debugGRU256")

    #model = torch.load("../models/debugGRU256")
    #test(model.cuda(), vec_collection, True)