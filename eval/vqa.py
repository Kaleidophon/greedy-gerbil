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
EvalResult = namedtuple("EvalResult", ["question_id", "top1", "top10", "prediction_diff"])
EvalDatum = namedtuple("EvalResult", ["question_id", "target", "predictions"])


class VQAEvaluator:
    """
    Evaluation helper class for the VQA dataset.
    """
    def __init__(self, data_set, model):
        self.data_set = data_set
        self.model = model
        self.data = EvaluationData()
        self.evaluated = False

    def eval(self):
        if not self.evaluated:
            dataloader = DataLoader(self.data_set, batch_size=1, shuffle=False, num_workers=4)
            for _, batch in enumerate(dataloader):
                question_features, target, image_features, _, question_id, answer_id = batch
                question_id = question_id.numpy()[0]
                question_features = Variable(question_features.long())
                target = int(target[0].numpy()[0])
                image_features = Variable(image_features)

                predictions = self.model(question_features, image_features).data.numpy()[0]

                self.data.add(EvalDatum(question_id=question_id, target=target, predictions=predictions))

            self.data.aggregate()

            self.evaluated = True

    @property
    def results(self):
        self.eval()
        return {
            "top1": self.data.top1,
            "top10": self.data.top10
        }


class EvaluationData:

    def __init__(self):
        self.results = []
        self.data = []
        self.top1_counter = 0
        self.top10_counter = 0
        self.aggregated = False

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
            for datum in self.data:
                question_id, target, predictions = datum
                sorted_answers, sorted_predictions = zip(
                    *sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)
                )
                top1, top10 = False, False
                most_likely = np.argmax(predictions)

                # Top1
                if target == most_likely:
                    top1 = True
                    self.top1_counter += 1

                # Top10
                if target in sorted_answers[:10]:
                    top10 = True
                    self.top10_counter += 1

                # Difference prediction actual target
                diff = target - most_likely
                self.results.append(EvalResult(question_id=question_id, top1=top1, top10=top10, prediction_diff=diff))

            self.aggregated = True

    @property
    def top1(self):
        self.aggregate()
        return self.top1_counter / len(self)

    @property
    def top10(self):
        self.aggregate()
        return self.top10_counter / len(self)

    def weakest_predictions(self):
        pass

    def strongest_predictions(self):
        pass


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
    print(evaluator.results)
