# -*- coding: utf-8 -*-
"""
Preparing data for later experiments.
"""

# STD
from collections import defaultdict
import json
import gzip
import os

# EXT
import numpy as np

# CONST
DATA_SET_TYPES = ("test", "train", "valid")
DATA_SET_PATH = os.path.dirname(__file__) + "/data/vqa_{}_{}.gzip"


class Question:
    def __init__(self, image_id, question, uid, **additional_arguments):
        self.image_id = image_id
        self.question = question
        self.uid = uid
        self.split = additional_arguments.get("split", "unknown")
        self.answers = additional_arguments.get("answers", {})
        self.choices = additional_arguments.get("choices", 0)
        self.qtype = additional_arguments.get("qtype", None)  # Question type
        self.atype = additional_arguments.get("atype", None)  # Answer type

    def __repr__(self):
        return "<Question #{}: '{}'>".format(self.uid, self.question)

    def __contains__(self, item):
        """
        Check if a question already contains a specific answer.
        """
        return item in {answer.answer for answer in self.answers.values()}

    def __iter__(self):
        for _, answer in self.answers.items():
            yield answer


class Answer:
    def __init__(self, answer, uid, **additional_arguments):
        self.answer = answer
        self.uid = uid
        self.confidence = additional_arguments.get("answer_confidence", None)
        self.image_id = additional_arguments.get("image_id", None)
        self.atype = additional_arguments.get("atype", None)  # Answer type

    def __repr__(self):
        return "<Answer #{}: '{}'>".format(self.uid, self.answer)

    def __eq__(self, other):
        assert type(other) == Answer, "You can only compare an answer to an answer."
        return self.answer == other.answer

    def __ne__(self, other):
        return not self.__eq__(other)


class HotLookup:
    """
    Class to look up hot one vectors given a vocabulary.
    """
    def __init__(self, vocabulary, key_func=lambda key: key, entry2index=None):
        self.vocabulary = vocabulary
        self.key_func = key_func
        self.entry2index = entry2index

    def __getitem__(self, item):
        key = self.key_func(item)

        # Convert to indices if necessary
        if self.entry2index is not None:
            if type(key) == list:
                key = [self.entry2index[el] for el in key]
            else:
                key = [self.entry2index[key]]

        key = set(key) if type(key) == list else {key}

        return np.array([1 if entry in key else 0 for entry in self.vocabulary])


def get_data_set(set_name, unique_answers=False):
    """
    Retrieve all the question and their respective answers from a data set (possible options are
    "test", "valid", and "train").

    :param unique_answers: Flag to indicate whether duplicate answers to a question should be kept.
    :type unique_answers: bool
    """
    questions, answers = dict(), dict()

    assert set_name in DATA_SET_TYPES, "Expected set name to be in {}, '{}' found.".format(DATA_SET_TYPES, set_name)
    raw_questions = read_data_file(path=DATA_SET_PATH.format("questions", set_name))["questions"]
    annotations = read_data_file(path=DATA_SET_PATH.format("annotations", set_name))["annotations"]

    # Process questions
    for raw_question in raw_questions:
        question_id = raw_question["question_id"]
        questions[question_id] = Question(
            image_id=raw_question["image_id"], question=raw_question["question"], uid=raw_question["question_id"],
            split=raw_question["split"]
        )

    # Process annotations
    for annotation in annotations:
        # Augment respective question
        question_id = annotation["question_id"]
        question = questions[question_id]
        question_type, answer_type, choices = \
            annotation["question_type"], annotation["answer_type"], annotation["multiple_choice_answer"]
        question.qtype, question.atype, question.choices = question_type, answer_type, choices

        # Parse answers and add to question
        for raw_answer in annotation["answers"]:
            choice_number = raw_answer["answer_id"]
            answer_id = "{}/{}".format(question_id, choice_number)
            answer = raw_answer["answer"]

            answer_obj = Answer(
                uid=answer_id, image_id=question.image_id, answer=answer,
                confidence=raw_answer["answer_confidence"], atype=answer_type
            )
            if answer not in question or not unique_answers:
                question.answers[choice_number] = answer_obj

            answers[answer_id] = answer_obj

    return questions, answers


def read_data_file(path):
    """
    Read a json data set from a gzipped file.
    """
    with gzip.open(path, "rb") as file:
        return json.loads(file.read())


def get_vocabulary(source, entry_getter, threshold=0, index_vocab=True, add_unk=True):
    """
    Generate vocabulary from a collection.

    :param source: Source collection for vocabulary.
    :param entry_getter: Function to get a entry for the vocabulary from an item in the source collection.
    :param threshold: Number of times a term has to appear to be included in the vocabulary.
    :param index_vocab: Index the words in the vocabulary.
    :param add_unk: Add entry for unknowns.
    """
    frequencies = defaultdict(int)

    # Count frequencies
    for item in source:
        entries = entry_getter(item)
        if type(entries) == list:
            for entry in entries:
                frequencies[entry] += 1
        else:
            entry = entries
            frequencies[entry] += 1

    # Filter according to threshold
    vocabulary = {key for key, value in frequencies.items() if value >= threshold}
    if add_unk:
        vocabulary.add("<unk>")

    # Index entries
    entry2index = {entry: index for index, entry in enumerate(vocabulary)}
    index2entry = {index: entry for index, entry in enumerate(vocabulary)}

    # Convert vocab
    if index_vocab:
        vocabulary = [entry2index[entry] for entry in vocabulary]

    return vocabulary, entry2index, index2entry


if __name__ == "__main__":
    questions, answers = get_data_set("train", unique_answers=False)

    #for _, question in questions.items():
    #    print(question)
    #    for answer in question:
    #        print(answer)

    question_vocabulary, qe2i, qi2e = get_vocabulary(
        questions.values(), entry_getter=lambda question: question.question.replace("?", "").split(" "),
        index_vocab=True, threshold=1000
    )
    answer_vocabulary, ae2i, ai2e = get_vocabulary(
        list(answers.values()), entry_getter=lambda answer: answer.answer, index_vocab=False, add_unk=False,
        threshold=1000
    )

    # print(answer_vocabulary)
    # hots_answers = HotLookup(answer_vocabulary)
    # print(hots_answers["red"])

    print(question_vocabulary)
    hots_questions = HotLookup(
        question_vocabulary, key_func=lambda key: key.replace("?", "").split(" "), entry2index=qe2i
    )
    print(hots_questions["How many are you?"])
