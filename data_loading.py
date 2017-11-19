# STD
import gzip
import json
import os
import pickle
from collections import namedtuple

# EXT
import h5py
import numpy as np
from torch.utils.data import Dataset

# CONST
DATA_SET_TYPES = ("test", "train", "valid")
DATA_SET_PATH = os.path.dirname(__file__) + "/data/vqa_{}_{}.gzip"
QAVectors = namedtuple("QAVector", ["question_vec", "answer_vec", "image_vec", "image_id", "question_id", "answer_id"])


class VQADataset(Dataset):
    """
    Class to store pairs of one-hot question and answer vectors and their corresponding image features as well as easily
     saving and loading them.
    """
    def __init__(self, load_path=None, image_features_path=None, image_features2id_path=None, verbosity=1):
        self.verbosity = verbosity

        # Load image features if necessary paths are given
        self.image_features = None
        if None not in (image_features_path, image_features2id_path):
            if verbosity > 0: print("Paths to image features are given, loading...", end="", flush=True)
            self.image_features = read_image_features_file(image_features_path, image_features2id_path)
            if verbosity > 0: print("\rLoading of image features complete!")
        else:
            print("WARNING! Paths to image features are not given. They will not be added to the data set.")

        # Load data from pickle file
        if load_path is not None:
            if verbosity > 0: print("Loading one hot vectors from pickle file...", end="", flush=True)
            self.data_vecs = self.load(load_path)
            if verbosity > 0: print("\rLoading one hot vectors from pickle file complete!")

    def save(self, path):
        save_qa_vectors(self.data_vecs, path, verbosity=self.verbosity)

    def load(self, path):
        return load_qa_vectors(path, image_features=self.image_features)

    def __iter__(self):
        for vec_pair in self.data_vecs:
            yield vec_pair

    def __len__(self):
        return len(self.data_vecs)

    def __getitem__(self, item):
        return self.data_vecs[item]


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


def read_image_features_file(features_path, features2id_path):
    img_features = np.asarray(h5py.File(features_path, 'r')['img_features'])

    with open(features2id_path, 'r') as f:
        visual_feat_mapping = json.load(f)['VQA_imgid2id']

    return {int(image_id): img_features[image_index] for image_id, image_index in visual_feat_mapping.items()}


def combine_data_sets(*data_sets, unique_answers=False):
    questions, answers, qid2set, aid2set = dict(), dict(), dict(), dict()

    for set_name in data_sets:
        current_questions, current_answers = get_data_set(set_name, unique_answers)

        questions.update(current_questions)
        answers.update(current_answers)

        # Remember which set they came from
        qid2set.update({question_id: set_name for question_id in current_questions})
        aid2set.update({answer_id: set_name for answer_id in current_answers})

    return questions, answers, qid2set, aid2set


def read_data_file(path):
    """
    Read a json data set from a gzipped file.
    """
    with gzip.open(path, "rb") as file:
        return json.loads(file.read())


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


def save_qa_vocab(qa_vocab, path):
    with open(path, "wb") as file:
        pickle.dump(qa_vocab, file)


def load_qa_vocab(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_qa_vectors(data_vecs, path, verbosity=0, convert=True):
    if verbosity > 0: print("Saving one hot vectors to pickle file...", end="", flush=True)
    with open(path, "wb") as file:
        if convert:
            pickle.dump([convert_qavectors_to_indices(vec_pair) for vec_pair in data_vecs], file)
        else:
            pickle.dump(data_vecs, file)
    if verbosity > 0: print("\rSaving one hot vectors to pickle file complete!")


def load_qa_vectors(path, image_features=None):
    with open(path, "rb") as file:
        raw_qavectors = pickle.load(file)
        return [convert_qavectors_to_vec(vec_pair, image_features) for vec_pair in raw_qavectors]


def convert_qavectors_to_indices(vec_pair):
    return (
        convert_vec_to_indices(vec_pair.question_vec),
        convert_vec_to_indices(vec_pair.answer_vec),
        vec_pair.image_id, vec_pair.question_id, vec_pair.answer_id
    )


def convert_qavectors_to_vec(vec_pair, image_features=None):
    # Add image features to tuple if available
    image_feature_func = lambda image_id: [] if image_features is None else image_features[image_id]

    return QAVectors(
        question_vec=convert_indices_to_vec(vec_pair[0]),
        answer_vec=convert_indices_to_vec(vec_pair[1]),
        image_vec=image_feature_func(vec_pair[2]),
        image_id=vec_pair[2], question_id=vec_pair[3], answer_id=vec_pair[4]
    )


def convert_vec_to_indices(vec):
    hot_indices = [len(vec)]
    hot_indices.extend(list(np.where(vec == 1)))  # Save "hot" indices, np.where is much fast than vanilla python loop

    return hot_indices


def convert_indices_to_vec(indices):
    length = indices.pop(0)  # Length of vector is first entry
    vec = np.zeros(length)

    for hot_index in indices:
        vec[hot_index] = 1

    return vec
