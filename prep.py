# -*- coding: utf-8 -*-
"""
Preparing data for later experiments.
"""

# STD
from collections import defaultdict, namedtuple
import json
import gzip
import os
import pickle

# EXT
import numpy as np

# PROJECT
from macos import MacOSFile

# CONST
DATA_SET_TYPES = ("test", "train", "valid")
DATA_SET_PATH = os.path.dirname(__file__) + "/data/vqa_{}_{}.gzip"
QAVectors = namedtuple("QAVector", ["question_vec", "answer_vec", "image_id"])
globals()[QAVectors.__name__] = QAVectors  # Hack to make this namedtuple pickle-able


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


class HotVectorCollection:
    """
    Class to store pairs of one-hot question and answer vectors as well as easily saving and loading them.
    """
    def __init__(self, load_path=None, set_name=None, unique_answers=False, args_question_voc=None,
                 args_answer_voc=None):
        # Create vectors from scratch
        if set_name is not None:
            self.vec_pairs = get_data_hot_vectors(set_name, unique_answers, args_question_voc, args_answer_voc)

        # Load vectors from pickle file
        elif load_path is not None:
            self.vec_pairs = self.load(load_path)

    def save(self, path):
        with open(path, "wb") as file:
            converted_qavectors = [self._convert_qavectors_to_indices(vec_pair) for vec_pair in self.vec_pairs]
            pickle.dump(converted_qavectors, MacOSFile(file))

    def load(self, path):
        with open(path, "rb") as file:
            raw_qavectors = pickle.load(MacOSFile(file))
            return [self._convert_qavectors_to_vec(vec_pair) for vec_pair in raw_qavectors]

    def _convert_qavectors_to_indices(self, vec_pair):
        return (
            self._convert_vec_to_indices(vec_pair.question_vec),
            self._convert_vec_to_indices(vec_pair.answer_vec),
            vec_pair.image_id
        )

    def _convert_qavectors_to_vec(self, vec_pair):
        return QAVectors(
            question_vec=self._convert_indices_to_vec(vec_pair[0]),
            answer_vec=self._convert_indices_to_vec(vec_pair[1]),
            image_id=vec_pair[2]
        )

    @staticmethod
    def _convert_vec_to_indices(vec):
        hot_indices = [len(vec)]

        for i in range(vec.shape[0]):
            if vec[i] == 1:
                hot_indices.append(i)

        return hot_indices

    @staticmethod
    def _convert_indices_to_vec(indices):
        length = indices.pop(0)  # Length of vector is first entry
        vec = np.zeros(length)

        for hot_index in indices:
            vec[hot_index] = 1

        return vec

    def __iter__(self):
        for vec_pair in self.vec_pairs:
            yield vec_pair


def get_data_set(set_name, unique_answers=False):
    """
    Retrieve all the question and their respective answers from a data set (possible options are
    "test", "valid", and "train").

    :param set_name: Name of data set.
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


def read_data_file(path):
    """
    Read a json data set from a gzipped file.
    """
    with gzip.open(path, "rb") as file:
        return json.loads(file.read())


def get_data_hot_vectors(set_name, unique_answers=False, args_question_voc=None, args_answer_voc=None):
    """
    Read in a data set (possible options are "test", "valid", and "train") and return the questions and answers
    as pairs of one-hot (or "multiple-hot") vectors.

    :param set_name: Name of data set.
    :param unique_answers: Flag to indicate whether redundant answers should be discarded.
    :param args_question_voc: Additional arguments for creating the question vocabulary.
    :param args_answer_voc: Additional arguments for creating the answer vocabulary.
    :return List of namedtuples
    """
    # Initialize default arguments to create the vocabulary
    args_question_voc = args_question_voc if args_question_voc is not None else {"index_vocab": True}
    args_answer_voc = args_answer_voc if args_answer_voc is not None else {"add_unk": False, "index_vocab": True}

    # Read data set, get vocabulary
    questions, answers = get_data_set(set_name, unique_answers)
    question_vocabulary, qe2i, qi2e = get_vocabulary(
        questions.values(), entry_getter=lambda question: question.question.replace("?", "").split(" "),
        **args_question_voc
    )
    answer_vocabulary, ae2i, ai2e = get_vocabulary(
        list(answers.values()), entry_getter=lambda answer: answer.answer, **args_answer_voc
    )

    # Initialise one hot vectors
    hots_questions = HotLookup(
        question_vocabulary, key_func=lambda key: key.replace("?", "").split(" "), entry2index=qe2i
    )
    hots_answers = HotLookup(answer_vocabulary, entry2index=ae2i)

    # Retrieve question / answer vector pairs
    vec_pairs = []
    for _, question in questions.items():
        question_vec = hots_questions[question.question]

        for answer in question:
            answer_vec = hots_answers[answer.answer]
            qa_vec_pair = QAVectors(question_vec=question_vec, answer_vec=answer_vec, image_id=question.image_id)
            vec_pairs.append(qa_vec_pair)

    return vec_pairs

if __name__ == "__main__":
    vec_collection = HotVectorCollection(set_name="valid")
    vec_collection.save("./data/valid_vecs.pickle")

    vec_collection2 = HotVectorCollection(load_path="./data/valid_vecs.pickle")
    for pair in vec_collection2:
        print(pair)
