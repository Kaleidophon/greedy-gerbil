import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.bow import BoWModel
from torch.utils.data.dataloader import DataLoader
from data_loading import VQADataset
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torch
import pickle
import numpy as np


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * ((epoch-1) ** 0.5) / (epoch ** 0.5)

def get_loss(model: nn.Module, dataset: VQADataset, batch=1000, cuda=False):
    if cuda:
        model.cuda()

    dropout_before = model.dropout_enabled
    model.dropout_enabled = False
    batch_counter = 0

    loss = 0
    criterion = nn.NLLLoss()
    dataload = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
    for i_batch, sample_batched in enumerate(dataload):
        batch_counter+= 1
        valid_questions, valid_answers, valid_image, _, _, _ = sample_batched
        valid_questions = Variable(valid_questions.long(), volatile=True)
        valid_answers = Variable(valid_answers[0], volatile=True)
        valid_image = Variable(valid_image, volatile=True)

        if cuda:
            valid_questions = valid_questions.cuda()
            valid_image = valid_image.cuda()
            valid_answers = valid_answers.cuda()
        #print("Batch number: ", i_batch)
        outputs = model(valid_questions, valid_image)
        loss += criterion(outputs, valid_answers)
    model.dropout_enabled = dropout_before
    return loss/batch_counter

def test_eval(model: nn.Module, dataset: VQADataset, batch=1000, cuda=False):
    if cuda:
        model.cuda()

    dropout_before = model.dropout_enabled
    model.dropout_enabled = False

    correct = 0

    dataload = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
    for i_batch, sample_batched in enumerate(dataload):
        valid_questions, valid_answers, valid_image, _, _, _ = sample_batched
        valid_questions = Variable(valid_questions.long(), volatile=True)
        valid_answers = valid_answers[0].numpy()
        valid_image = Variable(valid_image, volatile=True)

        if cuda:
            valid_questions = valid_questions.cuda()
            valid_image = valid_image.cuda()

        outputs = model(valid_questions, valid_image)
        m = torch.max(outputs.cpu(), 1)
        m = m[1].data.numpy()
        for i in range(len(m)):
            if m[i] == valid_answers[i]:
                correct += 1
        print("Batch number: ", i_batch)
    model.dropout_enabled = dropout_before
    print(correct / len(dataset) * 100)


def train(model: nn.Module, model_name, dataset_train: VQADataset, dataset_valid:VQADataset, batch_size=100, learning_curve=True, cuda=False):
    if cuda:
        model.cuda()

    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    last_loss = 2000000
    epoch_undecreased = 0
    epoch = 0

    learning_curve_train = []
    learning_curve_valid = []

    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adagrad([
        {'params': model.linearLayer.parameters()},
        {'params': model.embedding.parameters(), 'lr': 0.1}
    ], lr=1e-3)

    # optimizer = optim.Adam([
    #     {'params': model.linearLayer.parameters()},
    #     {'params': model.embedding.parameters(), 'lr': 0.1}
    # ], lr=1e-3 * 1)

    while True:
        epoch += 1
        print("Epoch:", epoch)

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
            loss = criterion(outputs, answer)

            loss.backward()
            optimizer.step()

        loss_valid = get_loss(model, dataset_valid, 1000, cuda=cuda).cpu().data.numpy()
        if learning_curve:
            loss_train = get_loss(model, dataset_train, 1000, cuda=cuda).cpu().data.numpy()
            print('[%d] loss_train: %.3f' % (epoch, loss_train))
            learning_curve_train.append(loss_train[0])
            learning_curve_valid.append(loss_valid[0])

        print('[%d] loss_valid: %.3f' % (epoch, loss_valid))
        if loss_valid > last_loss:
            epoch_undecreased += 1
        else:
            epoch_undecreased = 0
            tup = (learning_curve_train, learning_curve_valid) if learning_curve else None
            save_model(model, model_name, tup, cuda=cuda)
            last_loss = loss_valid

        if epoch_undecreased >= 5:
            break

    if learning_curve:
        t = (learning_curve_train, learning_curve_valid)
        return t


def save_model(model, model_name, learning_curves=None, cuda=False):
    if cuda:
        model.cpu()
    torch.save(model, model_name)
    if learning_curves is not None:
        with open(model_name + ".pkl", 'wb') as f:
            pickle.dump(learning_curves, f)

    if cuda:
        model.cuda()


def load_model(model_name, learning_curves=True):
    model = torch.load(model_name)
    l = None
    if learning_curves:
        with open(model_name + ".pkl", 'rb') as f:
            l = pickle.load(f)

    return model, l


if __name__ == "__main__":
    #small_data or big_data
    data_type = "small_data"
    #where to save/load model
    model_name = "../models/" + data_type + "/BoW_256_drop0.6"
    cuda = True

    vec_train = VQADataset(
        load_path="../data/" + data_type + "/vqa_vecs_train.pickle",
        image_features_path="../data/" + data_type + "/VQA_image_features.h5",
        image_features2id_path="../data/" + data_type + "/VQA_img_features2id.json",
        inflate_vecs=False
    )
    vec_valid = VQADataset(
        load_path="../data/" + data_type + "/vqa_vecs_valid.pickle",
        image_features_path="../data/" + data_type + "/VQA_image_features.h5",
        image_features2id_path="../data/" + data_type + "/VQA_img_features2id.json",
        inflate_vecs=False
    )

    model = BoWModel(vec_train.question_dim, 256, 2048, vec_train.answer_dim, dropout_prob=0.8)
    train(model, model_name, vec_train, vec_valid, batch_size=500, cuda=cuda)
    test_eval(model, vec_valid, 1000, cuda=cuda)
    # model, ll = load_model(model_name)
    # print(ll)

