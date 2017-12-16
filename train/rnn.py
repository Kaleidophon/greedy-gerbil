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
    batch_counter = 0

    loss = 0
    criterion = nn.NLLLoss()
    dataload = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
    for i_batch, batch in enumerate(dataload):
        batch_counter += 1
        questions, answers, images, _, _, _ = batch
        questions, images, answers, question_lengths = prepare_batch(questions, images, answers, dataset.question_dim, True, True)
        # perform forward pass
        model_answers = model(questions, images, question_lengths)
        loss += criterion(model_answers, answers)
    # model.dropout_enabled = dropout_before
    model.gru.dropout = dropout
    return loss/batch_counter

def test_eval(model, dataset, batch_size=1000, cuda=False):
    if cuda:
        model.cuda()

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
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


def train(model, model_name, dataset_train, dataset_valid, parameters, batch_size=100, learning_curve=True, cuda=False):
    # prepare data
    if cuda:
        model = model.cuda()

    dataset_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # set expectations

    learning_curve_train = []
    learning_curve_valid = []

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    last_loss = 2000000
    epoch_undecreased = 0
    epoch = 0
    optimizer = optim.Adagrad([
        {'params': model.layer_transform.parameters()},
        {'params': model.gru.parameters()},
        {'params': model.embedding.parameters(), 'lr': parameters["embed_lr"]}
    ], lr=parameters["other_lr"])

    # start training
    while True:
        # iterate over batches
        epoch += 1
        print("Epoch", epoch)
        for i_batch, batch in enumerate(dataset_loader):
            # print("training in epoch", epoch, "on batch", i_batch+1)
            questions, answers, images, _, _, _ = batch
            # prepare the batch
            questions, images, answers, question_lengths = prepare_batch(questions, images, answers, dataset_train.question_dim, False, cuda)
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
        loss_valid = get_loss(model, dataset_valid, 1000, cuda=cuda).cpu().data.numpy()
        if learning_curve:
            loss_train = get_loss(model, dataset_train, 1000, cuda=cuda).cpu().data.numpy()
            print('[%d] loss_train: %.3f' % (epoch, loss_train))
            learning_curve_train.append(loss_train[0])
            learning_curve_valid.append(loss_valid[0])

        print('[%d] loss_valid: %.3f' % (epoch, loss_valid))
        if loss_valid[0] > last_loss:
            epoch_undecreased += 1
        else:
            epoch_undecreased = 0

            save_model(model, model_name, cuda=cuda)
            last_loss = loss_valid[0]

        if epoch_undecreased >= 5:
            if learning_curve:
                tup = (learning_curve_train, learning_curve_valid)
                save_learning_curves(model_name, tup)
            return last_loss


def save_model(model, model_name, cuda=False):
    if cuda:
        model.cpu()
    torch.save(model, model_name)
    if cuda:
        model.cuda()

def save_learning_curves(model_name, learning_curves):
    if learning_curves is not None:
        with open(model_name + ".pkl", 'wb') as f:
            pickle.dump(learning_curves, f)


def load_model(model_name, learning_curves=True):
    model = torch.load(model_name)
    l = None
    if learning_curves:
        with open(model_name + ".pkl", 'rb') as f:
            l = pickle.load(f)

    return model, l


def try_model(parameters, vec_train, vec_valid, cuda=False):
    model = RNNModel(vec_train.question_dim, IMAGE_FEATURE_SIZE, parameters["embed_size"], vec_train.answer_dim,
                     num_layers=parameters["hidden_layers"], dropout_prob=parameters["dropout_prob"], cuda_enabled=cuda)

    param_string = "RNN_" + str(parameters["embed_size"]) + "_"\
                   + str(parameters["dropout_prob"]) + "_" + str(parameters["embed_lr"]) + "_"\
                   + str(parameters["other_lr"]) + "_" + str(parameters["batch_size"]) + "_" \
                   + str(parameters["hidden_layers"])

    model_name = "../models/" + parameters["data_type"] + "/RNN/" + param_string
    loss = train(model, model_name, vec_train, vec_valid, parameters, batch_size=parameters["batch_size"], cuda=cuda)
    with open("../models/" + parameters["data_type"] + "/RNN/results.txt", "a") as myfile:
        myfile.write(param_string + " " + str(loss) + "\n")

if __name__ == "__main__":
    #small_data or big_data
    data_type = "small_data"
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

    configuration_list = [
        {"embed_size": 512, "dropout_prob": 0.6, "data_type": data_type, "embed_lr": 0.05, "other_lr": 1e-4 * 5,
         "batch_size": 1000, "hidden_layers": 1},
        {"embed_size": 256, "dropout_prob": 0.5, "data_type": data_type, "embed_lr": 0.05, "other_lr": 1e-4 * 5,
         "batch_size": 1000, "hidden_layers": 1},
        {"embed_size": 256, "dropout_prob": 0.6, "data_type": data_type, "embed_lr": 0.05, "other_lr": 1e-4 * 5,
         "batch_size": 1000, "hidden_layers": 1},
        {"embed_size": 256, "dropout_prob": 0.7, "data_type": data_type, "embed_lr": 0.05, "other_lr": 1e-4 * 5,
         "batch_size": 1000, "hidden_layers": 1},
        {"embed_size": 512, "dropout_prob": 0.8, "data_type": data_type, "embed_lr": 0.05, "other_lr": 1e-4 * 5,
         "batch_size": 1000, "hidden_layers": 1},
        {"embed_size": 512, "dropout_prob": 0.7, "data_type": data_type, "embed_lr": 0.05, "other_lr": 1e-4 * 5,
         "batch_size": 1000, "hidden_layers": 1},
    ]

    for par in configuration_list:
        try_model(par, vec_train, vec_valid, cuda=cuda)
    # model_name = "../models/" + data_type + "/RNN/no_smoothening/RNN_256_0.5_0.1_0.001_1000_2"
    # model, ll = load_model(model_name)
    # test_eval(model, vec_valid, 1000, cuda=cuda)
    # print(ll)