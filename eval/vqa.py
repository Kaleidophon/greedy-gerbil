# -*- coding: utf-8 -*-
"""
Helper classes for evaluating experiments.
"""

class VQAEvaluator:
    '''
    Evaluation helper class for the VQA dataset.
    '''
    def __init__(self, data_set, model):
        self.data_set = self._split_data_set(data_set)
        self.model = model

    def _split_data_set(self, data_set):
        '''
        Returns dictionary with questions assigned to their respective splits.
        '''
        res = {}
        for question in data_set:
            res[question.split] = res.get(question.split, []) += [question]
        return res

    def evaluate(self, split, multiple_choice=False):
        '''
        Evaluates model on given split.

        Returns a dictionary with accuracies by answer type.
        '''
        res = {'all': 0., 'other': 0., 'yes/no': 0., 'number': 0.} # accuracies
        count_atypes = dict(res)
        for question in self.data_set[split]:
            # TODO update according to final implementation
            # assumption that predictions is a dict of form {'word': float}
            predictions = self.model.answer(question.image_id, question.question)
            if multiple_choice:
                predictions = {answer.answer: predictions.get(answer.answer, 0.) for answer in question.answers}
            model_answer = max(predictions, key=predictions.get)
            if model_answer == question.choices:
                res['all'] += 1.
                res[question.atype] += 1.
            # count answer types
            count_atypes['all'] += 1.
            count_atypes[question.atype] += 1.
        # normalize
        for atype in res:
            res[atype] /= count_atypes[atype]
        return res
