# -*- coding: utf-8 -*-
"""
Helper classes for evaluating experiments.
"""

# STD
import codecs
from collections import namedtuple, defaultdict
from os import listdir
from os.path import isfile, join

# EXT
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn


# PROJECT
from data_loading import VQADataset
from one_hot import combine_data_sets
from models.rnn import RNNModel

# CONST
EvalResult = namedtuple("EvalResult", ["question_id", "image_id", "top1", "top5", "prediction_diff"])
EvalDatum = namedtuple("EvalResult", ["question_id", "image_id", "target", "predictions"])


def eval_models_in_dir(directory, data_set, batch_size, questions, eval_path, cuda=False, strongest_n=10, weakest_n=10,
                       whitelist=None, blacklist={".pkl", ".pickle", ".txt"}):
    """
    Evaluate all models in a directory.

    :param directory: Model directory
    :param data_set: Data set with evaluation data
    :param batch_size: Batch size for evaluator
    :param questions: Questions for more telling evaluation
    :param whitelist: Allowed file suffixes for model files
    :param blacklist: Disallowed file suffices for model files
    :param strongest_n: Number of strongest predictions per model
    :param weakest_n: Number of weakest predictions per model
    :return:
    """
    model_paths = [join(directory, file) for file in listdir(directory) if isfile(join(directory, file))]

    if whitelist is not None:
        model_paths = list(filter(lambda path: any([path.endswith(white) for white in whitelist]), model_paths))
    else:
        model_paths = list(filter(lambda path: all([not path.endswith(black) for black in blacklist]), model_paths))

    with codecs.open(eval_path, "wb", "utf-8") as eval_file:
        for model_path in model_paths:
            model = torch.load(model_path)
            model.dropout_enabled = False
            evaluator = VQAEvaluator(data_set, model, batch_size=batch_size, questions=questions)
            evaluator.eval(cuda=cuda)
            result = evaluator.results(strongest_n=strongest_n, weakest_n=weakest_n)
            eval_file.write(
                "{}:\n{}\n\n".format(
                    model_path, "\n".join(["{}: {}".format(key, value) for key, value in result.items()])
                )
            )


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


class VQAEvaluator:
    """
    Evaluation helper class for the VQA dataset.
    """
    def __init__(self, data_set, model, batch_size=1, questions=None, verbosity=1):
        self.data_set = data_set
        self.model = model
        self.batch_size = batch_size

        # Don't use dropout during evaluation
        # BoW
        if hasattr(self.model, "dropout"):
            self.model.dropout = nn.Dropout(0)
        # RNN w/ GRUs
        elif hasattr(self.model, "gru"):
            self.model.gru.dropout = 0

        self.data = EvaluationData(questions=questions)
        self.evaluated = False
        self.verbosity = verbosity
        self.question = questions

    def eval(self, cuda=False):
        if not self.evaluated:
            size = len(self.data_set)
            if cuda:
                self.model.cuda()
            if self.verbosity > 0: print("Evaluating model...", flush=True, end="")
            dataloader = DataLoader(self.data_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
            for n, batch in enumerate(dataloader):
                current_batch_size = len(batch[0])
                if self.verbosity > 0:
                    print(
                        "\rCollecting predictions {}/{} ({:.2f} % complete)".format(
                            (n+1)*self.batch_size, size, (n+1)*current_batch_size / size * 50
                        ), flush=True, end=""
                    )

                question_features, target, image_features, image_ids, question_id, answer_id = batch
                question_id = question_id.numpy()#[0]
                image_ids = image_ids.numpy()
                #question_features = Variable(question_features.long(), volatile=True)
                #image_features = Variable(image_features, volatile=True)
                target = target[0].numpy()#[0])
                if cuda:
                    question_features = question_features.cuda()
                    image_features = image_features.cuda()

                if isinstance(self.model, RNNModel):
                    predictions = self.model(question_features, image_features, self.data_set.question_dim)
                else:
                    predictions = self.model(question_features, image_features)

                if cuda:
                    predictions = predictions.cpu()
                predictions = predictions.data.numpy()#[0]

                for i in range(current_batch_size):
                    self.data.add(
                        EvalDatum(
                            question_id=question_id[i], image_id=image_ids[i], target=target[i],
                            predictions=predictions[i]
                        )
                    )

            self.data.aggregate()
            if cuda:
                self.model.cpu()
            self.evaluated = True
            if self.verbosity > 0: print("\rEvaluating model complete!")

    def results(self, strongest_n=50, weakest_n=50):
        self.eval()
        return {
            "top1": self.data.top1,
            "top5": self.data.top5,
            "strongest": self.data.strongest_predictions(strongest_n),
            "weakest": self.data.weakest_predictions(weakest_n)
        }

    def __iter__(self):
        return (result for result in self.data)

    def __len__(self):
        return len(self.data)


class EvaluationData:

    def __init__(self, verbosity=1, questions=None):
        self.results = []
        self.data = []
        self.top1_counter = 0
        self.top5_counter = 0
        self.top1_counter_by_category = defaultdict(int)  # Number of correct predictions per category
        self.top5_counter_by_category = defaultdict(int)  # Number of correct prediction in the top 5 per category
        self.instances_by_category = defaultdict(int)  # Number of questions per question category
        self.aggregated = False
        self.verbosity = verbosity
        self.questions = questions

    def __iter__(self):
        self.aggregate()
        return (result for result in self.results)

    def __len__(self):
        self.aggregate()
        return len(self.results)

    def __getitem__(self, item):
        self.aggregate()
        return self.results[item]

    def add(self, result):
        self.data.append(result)

    def aggregate(self):
        if not self.aggregated:
            size = len(self.data)

            for n, datum in enumerate(self.data):
                if self.verbosity > 0:
                    print(
                        "\rEvaluating predictions {}/{} ({:.2f} % complete)".format(
                            n+1, size, (size+n+1) / size*50
                        ), flush=True, end=""
                    )

                question_id, image_id, target, predictions = datum
                sorted_answers, sorted_predictions = zip(
                    *sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)
                )  # Answer indices sorted by prediction score
                answer_ranks = {answer_index: rank for rank, answer_index in enumerate(sorted_answers)}  # For diff
                top1, top5 = False, False

                if self.questions is not None:
                    question = self.questions[question_id]
                    self.instances_by_category[question.atype] += 1

                # Top1
                if target == sorted_answers[0]:
                    top1 = True
                    self.top1_counter += 1

                    if self.questions is not None:
                        self.top1_counter_by_category[question.atype] += 1

                # Top10
                if target in sorted_answers[:5]:
                    top5 = True
                    self.top5_counter += 1

                    if self.questions is not None:
                        self.top5_counter_by_category[question.atype] += 1

                # Difference prediction actual target
                # Get the rank of the actual answer in the predictions - should ideally be zero
                diff = answer_ranks[target]
                self.results.append(
                    EvalResult(question_id=question_id, image_id=image_id, top1=top1, top5=top5, prediction_diff=diff)
                )

            self.aggregated = True

    @property
    def top1(self):
        self.aggregate()

        if self.questions is None:
            return self.top1_counter / len(self)
        else:
            top1_results = {"all": self.top1_counter / len(self)}
            top1_results.update(
                {
                    category: count / self.instances_by_category[category]
                    for category, count in self.top1_counter_by_category.items()
                }
            )
            return top1_results

    @property
    def top5(self):
        self.aggregate()

        if self.questions is None:
            return self.top5_counter / len(self)
        else:
            top5_results = {"all": self.top5_counter / len(self)}
            top5_results.update(
                {
                    category: count / self.instances_by_category[category]
                    for category, count in self.top5_counter_by_category.items()
                }
            )
            return top5_results

    def weakest_predictions(self, n=50):
        results_sorted_by_diff = sorted(self.results, key=lambda res: res.prediction_diff, reverse=True)

        if self.questions is None:
            return results_sorted_by_diff[:n]
        else:
            return [
                (self.questions[result.question_id].question, result.prediction_diff)
                for result in results_sorted_by_diff[:n]
            ]

    def strongest_predictions(self, n=50):
        results_sorted_by_diff = sorted(self.results, key=lambda res: res.prediction_diff)

        if self.questions is None:
            return results_sorted_by_diff[:n]
        else:
            return [
                (self.questions[result.question_id].question, result.prediction_diff)
                for result in results_sorted_by_diff[:n]
            ]


if __name__ == "__main__":
    # Load resources
    data_type = "small_data"
    # where to save/load model
    vec_valid = VQADataset(
        load_path="../data/" + data_type + "/vqa_vecs_valid.pickle",
        image_features_path="../data/" + data_type + "/VQA_image_features.h5",
        image_features2id_path="../data/" + data_type + "/VQA_img_features2id.json",
        inflate_vecs=False
    )
    # map_location enables to load a CUDA trained model, wtf
    torch.backends.cudnn.enabled = False
    questions, _, _, _ = combine_data_sets("train", "valid", "test", data_type=data_type, unique_answers=True)

    eval_models_in_dir(
        "../models/small_data/RNN/no_smoothening", vec_valid, batch_size=1000, questions=questions, cuda=True,
        eval_path="./res.txt"
    )
