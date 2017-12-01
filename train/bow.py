from models.bow import BoWModel
from torch.utils.data.dataloader import DataLoader
from data_loading import VQADataset
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torch


def train(model: nn.Module, dataset: VQADataset, iterations, learn_rate=0.01, cuda=False):
    if cuda:
        model = model.cuda()

    batch_size = 1000
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), learn_rate)


    for epoch in range(iterations):
        running_loss = 0
        for i_batch, sample_batched in enumerate(dataset_loader):
            question, answer, image, _, _, _ = sample_batched




            answer = answer[1].view(batch_size)
            # wrap them in Variable
            question, answer, image = Variable(question.long().cuda()), Variable(answer.cuda()), \
                                      Variable(image.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(question, image)
            #print(torch.max(outputs,dim=1))
            loss = criterion(outputs, answer)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            print_frequency = 1
            if i_batch % print_frequency == print_frequency-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch+ 1, running_loss / print_frequency))
                running_loss = 0.0

        print('Finished Training')


if __name__ == "__main__":
    vec_collection = VQADataset(
        load_path="./data/vqa_vecs_train.pickle",
        image_features_path="./data/VQA_image_features.h5",
        image_features2id_path="./data/VQA_img_features2id.json",
        inflate_vecs=False
    )
    #model = torch.load("models/debug1")
    model = BoWModel(7924, 256, 2048, 30806)
    train(model, vec_collection, 1, cuda=True)
    torch.save(model, "models/debug")
