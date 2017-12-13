# -*- coding: utf-8 -*-
"""
Helper classes for evaluating experiments.
"""

# STD
from collections import namedtuple

# PROJECT
from data_loading import VQADataset

# CONST
EvalResult = namedtuple("EvalResult", ["question_id", "target", "prediction"])


class VQAEvaluator:
    '''
    Evaluation helper class for the VQA dataset.
    '''
    def __init__(self, data_set, model):
        self.data_set = data_set
        self.model = model
        self.data = EvaluationData()

    def __call__(self):
        for vec_pair in self.data_set:
            word_features, image_features = vec_pair.question_vec, vec_pair.image_features
            question_id = vec_pair.question_id
            target = vec_pair.answer_vec

            prediction = self.model.forward(word_features, image_features)

            self.data.add(EvalResult(question_id=question_id, target=target, prediction=prediction))



class EvaluationData:

    def __init__(self):
        self.results = []

    def __iter__(self):
        return (result for result in self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        return self.results[item]

    def add(self, result):
        self.results.append(result)

    @property
    def top1(self):
        pass

    @property
    def top10(self):
        pass

    def weakest_predictions(self):
        pass

    def strongest_predictions(self):
        pass
