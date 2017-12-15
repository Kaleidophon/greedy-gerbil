import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from models.rnn import RNNModel
from data_loading import *

def prepare_batch(questions, images, answers, padding_idx, volatile=False ,cuda=False):
    res_question_lengths = []
    for question in questions:
        if padding_idx not in list(question):
            res_question_lengths.append(len(question)-1)
        else:
            res_question_lengths.append(list(question).index(padding_idx)-1)
    # prepare for computation
    # print(max(res_question_lengths))
    res_questions = Variable(torch.LongTensor(questions), volatile=volatile)
    res_images = Variable(images, volatile=volatile)
    res_answers = Variable(torch.LongTensor(answers[0]), volatile=volatile)
    res_question_lengths = Variable(torch.LongTensor(res_question_lengths), volatile=volatile)
    if cuda: # CUDA for David
        res_questions = res_questions.cuda()
        res_images = res_images.cuda()
        res_answers = res_answers.cuda()
        res_question_lengths = res_question_lengths.cuda()
    return res_questions, res_images, res_answers, res_question_lengths

def get_loss(model: nn.Module, dataset: VQADataset, batch=1000, cuda=False):
    if cuda:
        model = model.cuda()

    # dropout_before = model.dropout_enabled
    # model.dropout_enabled = False
    dropout = model.gru.dropout
    model.gru.dropout= 0

    loss = 0
    criterion = nn.NLLLoss()
    dataload = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
    for i_batch, batch in enumerate(dataload):
        questions, answers, images, _, _, _ = batch
        questions, images, answers, question_lengths = prepare_batch(questions, images, answers, dataset.question_dim, True, True)
        # perform forward pass
        model_answers = model(questions, images, question_lengths)
        loss += criterion(model_answers, answers)
    # model.dropout_enabled = dropout_before
    model.gru.dropout = dropout
    return loss

def test(model, dataset, cuda=False):
    dataset_loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4, drop_last=True)
    correct = 0

    dropout = model.gru.dropout
    model.gru.dropout = 0

    for i_batch, batch in enumerate(dataset_loader):
        questions, answers, images, _, _, _ = batch
        questions, images, answers, question_lengths = prepare_batch(questions, images, answers, dataset.question_dim, True, True)
        # perform forward pass
        answers = answers.cpu().data.numpy()
        model_answers = model(questions, images, question_lengths)
        m = torch.max(model_answers.cpu(), 1)
        m = m[1].data.numpy()
        for i in range(len(m)):
            if m[i] == answers[i]:
                correct += 1
        print("Batch number: ", i_batch)
    print(correct/len(dataset))
    model.gru.dropout = dropout


def train(model, dataset, valid_set, batch_size=100, cuda=False):
    # prepare data
    if cuda:
        model = model.cuda()

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # set expectations


    lr_embed = 0.1
    lr_other = 1e-3
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    last_loss = 2000000
    epoch_unincreased = 0
    epoch = 0
    optimizer = optim.Adagrad([
        {'params': model.layer_transform.parameters()},
        {'params': model.gru.parameters()},
        {'params': model.embedding.parameters(), 'lr': lr_embed}
    ], lr=lr_other)

    # start training
    while True:
        # iterate over batches
        epoch += 1
        print("Epoch", epoch)
        for i_batch, batch in enumerate(dataset_loader):
            # print("training in epoch", epoch, "on batch", i_batch+1)
            questions, answers, images, _, _, _ = batch
            # prepare the batch
            questions, images, answers, question_lengths = prepare_batch(questions, images, answers, dataset.question_dim, False, cuda)
            # zero the parameter gradients
            optimizer.zero_grad()
            # perform forward pass
            model_answers = model(questions, images, question_lengths)
            # calculate loss
            loss = criterion(model_answers, answers)
            # backpropagate
            loss.backward()
            optimizer.step()
            # if i_batch % 100 == 0:
            #     print(loss[0])
        loss_valid = get_loss(model, valid_set, 1000, True).cpu().data.numpy()
        print('[%d] loss_valid: %.3f' % (epoch, loss_valid))
        if loss_valid > last_loss:
            epoch_unincreased += 1
        else:
            epoch_unincreased = 0

        last_loss = loss_valid
        if epoch_unincreased >= 3:
            break


    print('Training complete.')

def save_model(model, model_name, cuda=False):
    if cuda:
        model.cpu()
    torch.save(model, model_name)

if __name__ == "__main__":
    #small_data or big_data
    data_type = "small_data"
    # where to save/load model
    model_name = "../models/" + data_type + "/RNN_Batch_Debug"
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
    #This line is mysterious but prevents mysterious errors from cudnn (and took 2 hours of my sleep)
    torch.backends.cudnn.enabled = False

    # model = torch.load(model_name)
    model = RNNModel(vec_train.question_dim, IMAGE_FEATURE_SIZE, 256, vec_train.answer_dim, num_layers=1, dropout_prob=0.6, cuda_enabled=cuda)
    train(model, vec_train, vec_valid, batch_size=1000, cuda=cuda)
    save_model(model, model_name, cuda=cuda)
    test(model.cuda(), vec_valid, cuda=cuda)