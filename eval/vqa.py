# -*- coding: utf-8 -*-
"""
Helper classes for evaluating experiments.
"""

# STD
from collections import namedtuple

# EXT
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

# PROJECT
from data_loading import VQADataset

# CONST
EvalResult = namedtuple("EvalResult", ["question_id", "top1", "top5", "prediction_diff"])
EvalDatum = namedtuple("EvalResult", ["question_id", "target", "predictions"])


class VQAEvaluator:
    """
    Evaluation helper class for the VQA dataset.
    """
    def __init__(self, data_set, model, verbosity=1):
        self.data_set = data_set
        self.model = model
        self.data = EvaluationData()
        self.evaluated = False
        self.verbosity = verbosity

    def eval(self):
        if not self.evaluated:
            size = len(self.data_set)

            if self.verbosity > 0: print("Evaluating model...", flush=True, end="")
            dataloader = DataLoader(self.data_set, batch_size=1, shuffle=False, num_workers=4)
            for n, batch in enumerate(dataloader):
                if self.verbosity > 0:
                    print(
                        "\rCollecting predictions {}/{} ({:.2f} % complete)".format(n+1, size, (n+1) / size * 50),
                        flush=True, end=""
                    )

                question_features, target, image_features, _, question_id, answer_id = batch
                question_id = question_id.numpy()[0]
                question_features = Variable(question_features.long())
                target = int(target[0].numpy()[0])
                image_features = Variable(image_features)

                predictions = self.model(question_features, image_features).data.numpy()[0]

                self.data.add(EvalDatum(question_id=question_id, target=target, predictions=predictions))

            self.data.aggregate()

            self.evaluated = True
            if self.verbosity > 0: print("\rEvaluating model complete!")

    def results(self, strongest_n=50, weakest_n=50):
        self.eval()
        return {
            "top1": self.data.top1,
            "top5": self.data.top10,
            "strongest": self.data.strongest_predictions(strongest_n),
            "weakest": self.data.weakest_predictions(weakest_n)
        }

    def __iter__(self):
        return (result for result in self.data)

    def __len__(self):
        return len(self.data)


class EvaluationData:

    def __init__(self, verbosity=1):
        self.results = []
        self.data = []
        self.top1_counter = 0
        self.top5_counter = 0
        self.aggregated = False
        self.verbosity = verbosity

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

                question_id, target, predictions = datum
                sorted_answers, sorted_predictions = zip(
                    *sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)
                )  # Answer indices sorted by prediction score
                answer_ranks = {answer_index: rank for rank, answer_index in enumerate(sorted_answers)}  # For diff
                top1, top5 = False, False

                # Top1
                if target == sorted_answers[0]:
                    top1 = True
                    self.top1_counter += 1

                # Top10
                if target in sorted_answers[:5]:
                    top5 = True
                    self.top5_counter += 1

                # Difference prediction actual target
                # Get the rank of the actual answer in the predictions - should ideally be zero
                diff = answer_ranks[target]
                self.results.append(EvalResult(question_id=question_id, top1=top1, top5=top5, prediction_diff=diff))

            self.aggregated = True

    @property
    def top1(self):
        self.aggregate()
        return self.top1_counter / len(self)

    @property
    def top10(self):
        self.aggregate()
        return self.top5_counter / len(self)

    def weakest_predictions(self, n=50):
        predictions_sorted_by_diff = sorted(self.results, key=lambda res: res.prediction_diff, reverse=True)
        return predictions_sorted_by_diff[:n]

    def strongest_predictions(self, n=50):
        predictions_sorted_by_diff = sorted(self.results, key=lambda res: res.prediction_diff)
        return predictions_sorted_by_diff[:n]


if __name__ == "__main__":
    vec_test = VQADataset(
        load_path="../data/vqa_vecs_test.pickle",
        image_features_path="../data/VQA_image_features.h5",
        image_features2id_path="../data/VQA_img_features2id.json",
        inflate_vecs=False
    )
    model = torch.load("../models/debug1")

    evaluator = VQAEvaluator(vec_test, model)
    evaluator.eval()

    result = evaluator.results(strongest_n=10, weakest_n=10)
    print(result)
