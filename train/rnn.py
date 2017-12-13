import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from models.rnn import RNNModel
from data_loading import *

def prepare_batch(questions, images, answers, padding_idx, cuda=False):
    res_question_lengths = []
    for question in questions:
        if padding_idx not in list(question):
            res_question_lengths.append(len(question)-1)
        else:
            res_question_lengths.append(list(question).index(padding_idx)-1)
    # prepare for computation
    print(max(res_question_lengths))
    res_questions = Variable(torch.LongTensor(questions))
    res_images = Variable(images)
    res_answers = Variable(torch.LongTensor(answers[0]))
    res_question_lengths = Variable(torch.LongTensor(res_question_lengths))
    if cuda: # CUDA for David
        res_questions = res_questions.cuda()
        res_images = res_images.cuda()
        res_answers = res_answers.cuda()
        res_question_lengths = res_question_lengths.cuda()
    return res_questions, res_images, res_answers, res_question_lengths

# def get_loss(model: nn.Module, dataset: VQADataset, batch=1000, cuda=False):
#     if cuda:
#         model = model.cuda()

#     loss = 0
#     criterion = nn.NLLLoss()
#     dataload = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
#     for i_batch, sample_batched in enumerate(dataload):
#         valid_questions, valid_answers, valid_image, _, _, _ = sample_batched
#         valid_questions = Variable(torch.LongTensor(list(filter((dataset.question_dim).__ne__,
#                                                                 valid_questions.view(valid_questions.numel())))), volatile=True)
#         valid_answers = Variable(valid_answers[0], volatile=True)
#         valid_image = Variable(valid_image.view(valid_image.numel()), volatile=True)

#         if cuda:
#             valid_questions = valid_questions.cuda()
#             valid_image = valid_image.cuda()
#             valid_answers = valid_answers.cuda()
#         if i_batch % 1000 == 0 : print("Batch number: ", i_batch)
#         outputs = model(valid_questions, valid_image)
#         loss += criterion(outputs, valid_answers)
#     return loss

def test(model, dataset, cuda=False):
    dataset_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4, drop_last=True)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    correct = 0
    loss = 0.
    for i_batch, batch in enumerate(dataset_loader):
        questions, answers, images, _, _, _ = batch
        questions, images, answers, question_lengths = prepare_batch(questions, images, answers, dataset.question_dim)
        # perform forward pass
        model_answers = model(questions, images)
        m = torch.max(model_answers.cpu(), 1)[1]
        m = m.data.numpy()
        loss += criterion(model_answers, answers)
        if m[0] == answers.cpu().data.numpy()[0]:
            correct += 1
    print(loss.data[0]/len(dataset),correct/len(dataset))


def train(model, dataset, valid_set, iterations, batch_size=100, cuda=False):
    # prepare data
    if cuda:
        model = model.cuda()

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    # set expectations


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
            print("training in epoch", epoch+1, "on batch", i_batch+1)
            loss = 0.
            # separate batch into relevant parts
            questions, answers, images, _, _, _ = batch
            # prepare the batch
            questions, images, answers, question_lengths = prepare_batch(questions, images, answers, dataset.question_dim, cuda)
            # zero the parameter gradients
            optimizer.zero_grad()
            # perform forward pass
            model_answers = model(questions, images, question_lengths)
            # calculate loss
            loss = criterion(model_answers, answers)
            # backpropagate
            loss.backward()
            optimizer.step()
            print(loss)
        #test(model, valid_set, True)


    print('Training complete.')

if __name__ == "__main__":
    #small_data or big_data
    data_type = "small_data"
    # where to save/load model
    model_name = "../models/" + data_type + "/BoW_512_drop0.8"

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

    model = RNNModel(vec_train.question_dim, IMAGE_FEATURE_SIZE, 128, vec_train.answer_dim, cuda_enabled=True)
    train(model, vec_train, vec_valid, 10, batch_size=1000, cuda=True)
    torch.save(model, "../models/debugGRU128")

    #model = torch.load("../models/debugGRU256")
    #test(model.cuda(), vec_collection, True)