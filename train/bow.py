import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.bow import BoWModel
from torch.utils.data.dataloader import DataLoader
from data_loading import VQADataset
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torch
import numpy as np


def test_eval(model: nn.Module, dataset: VQADataset, cuda=False):
    if cuda:
        model = model.cuda()
    dataload = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)
    for i_batch, sample_batched in enumerate(dataload):
        valid_questions, valid_answers, valid_image, _, _, _ = sample_batched
        valid_questions = Variable(valid_questions.long())
        valid_answers = valid_answers[0].numpy()
        valid_image = Variable(valid_image)

        if cuda:
            valid_questions = valid_questions.cuda()
            valid_image = valid_image.cuda()

    outputs = model(valid_questions, valid_image)
    m = torch.max(outputs.cpu(), 1)[1]
    m = m.data.numpy()
    correct = 0
    for i in range(len(m)):
        if m[i] == valid_answers[i]:
            correct += 1
    print(correct / len(m) * 100)


def train(model: nn.Module, dataset_train: VQADataset, dataset_valid:VQADataset, iterations, batch_size=100, learn_rate=0.8, cuda=False):
    if cuda:
        model = model.cuda()

    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dataload_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False, num_workers=1)

    criterion = nn.NLLLoss()

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad([
        {'params': model.linearLayer.parameters()},
        {'params': model.embedding.parameters(), 'lr': 0.01}
    ], lr=1e-4 * 1)

    for i_batch, sample_batched in enumerate(dataload_valid):
        valid_questions, valid_answers, valid_image, _, _, _ = sample_batched
        valid_questions = Variable(valid_questions.long(), volatile=True)
        valid_answers = Variable(valid_answers[0], volatile=True)
        valid_image = Variable(valid_image, volatile=True)
        if cuda:
            valid_questions = valid_questions.cuda()
            valid_answers = valid_answers.cuda()
            valid_image = valid_image.cuda()
    for epoch in range(iterations):
        for i_batch, sample_batched in enumerate(dataload_train):
            question, answer, image, _, _, _ = sample_batched

            # wrap them in Variable
            question = Variable(question.long())
            answer = Variable(answer[0])
            image = Variable(image)

            if cuda:
                question = question.cuda()
                answer = answer.cuda()
                image = image.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(question, image)
            #a = torch.max(outputs, dim=1)
            #print(a)
            loss = criterion(outputs, answer)
            loss.backward()
            optimizer.step()

            # print statistics
        outputs = model(valid_questions, valid_image)
        loss_valid = criterion(outputs, valid_answers)

        print('[%d] loss_valid: %.3f' % (epoch + 1, loss_valid[0]))
        test_eval(model, dataset_valid, cuda)




if __name__ == "__main__":
    vec_train = VQADataset(
        load_path="../data/vqa_vecs_train.pickle",
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


    #model = torch.load("../models/debug")
    #test_eval(model, vec_train, True)
    model = BoWModel(vec_train.question_dim, 20, 2048, vec_train.answer_dim)
    train(model, vec_train, vec_valid, 100, cuda=False)
    torch.save(model, "../models/debug1")

