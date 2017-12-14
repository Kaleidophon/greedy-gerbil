# -*- coding: utf-8 -*-
"""
Preparing data for later experiments.
"""

# STD
from collections import defaultdict, namedtuple

# EXT
import numpy as np
from torch.utils.data import DataLoader

# PROJECT
from data_loading import (
    combine_data_sets, QAVectors, save_qa_vocab, save_qa_vectors, convert_qavectors_to_indices, VQADataset
)

# CONST
QAVocabulary = namedtuple(
    "QAVocabulary", [
        "questions",  # Question vocabulary (a word in the vocabulary is called an "entry")
        "answers",    # Answer vocabulary (a word in the vocabulary is called an "entry")
        "qe2i",       # Dict from question vocabulary entry to its index in the one hot vector
        "qi2e",       # Reverse of qe2i
        "ae2i",       # Dict from answer vocabulary entry to its index in the one hot vector
        "ai2e"        # Reverse of ae2i
    ]
)

# Hack to make these namedtuples pickle-able
globals()[QAVectors.__name__] = QAVectors
globals()[QAVocabulary.__name__] = QAVocabulary


class HotLookup:
    """
    Class to look up hot one vectors given a vocabulary.
    """
    def __init__(self, vocabulary, entry2index, key_func=lambda key: key):
        self.vocabulary = vocabulary
        self.vocab_length = len(vocabulary)
        self.key_func = key_func
        self.entry2index = entry2index

    def __getitem__(self, item):
        key = self.key_func(item)

        # Convert to indices
        one_hots = None  # "Hot" indices
        if type(key) == list:
            one_hots = [self.entry2index[el] for el in key]
        else:
            one_hots = [self.entry2index[key]]

        one_hot = np.zeros(shape=(self.vocab_length, 1))
        for hot_index in one_hots:
            one_hot[hot_index] = 1

        return one_hot


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


def get_data_hot_vectors(questions, answers, image_features=None, args_question_voc=None, args_answer_voc=None,
                         convert=False):
    """
    Take questions and answers of a data set and return the questions and answers
    as pairs of one-hot (or "multiple-hot") vectors (and image features, if they have been loaded).

    :param questions: List of Question objects for a data set.
    :param answers: List Answer objects for a data set.
    :param image_features: Dictionary of image ids and their corresponding image features.
    :param args_question_voc: Additional arguments for creating the question vocabulary.
    :param args_answer_voc: Additional arguments for creating the answer vocabulary.
    :param convert: Convert one-hot vectors into dense data format during creation.
    :return List of namedtuples
    """
    # Initialize default arguments to create the vocabulary
    args_question_voc = args_question_voc if args_question_voc is not None else {"index_vocab": True}
    args_answer_voc = args_answer_voc if args_answer_voc is not None else {"add_unk": False, "index_vocab": True}

    # Read data set, get vocabulary
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
    data_vecs = []
    for _, question in questions.items():
        question_vec = hots_questions[question.question]

        for answer in question:
            answer_vec = hots_answers[answer.answer]
            qa_vec = QAVectors(
                question_vec=question_vec, answer_vec=answer_vec, image_id=question.image_id,
                image_vec=[] if image_features is None else image_features[question.image_id],
                question_id=question.uid, answer_id=answer.uid
            )

            if convert:
                qa_vec = convert_qavectors_to_indices(qa_vec)

            data_vecs.append(qa_vec)

    return data_vecs, QAVocabulary(
        questions=question_vocabulary, answers=answer_vocabulary, qe2i=qe2i, qi2e=qi2e, ae2i=ae2i, ai2e=ae2i
    )


def save_vectors(data_vecs, target_dir, qid2set, convert=True):
    """
    Save a list of QAVectors namedtuples into pickle files.

    :param data_vecs: List of QAVectors.
    :param target_dir: Directory to save vectors in.
    :param qid2set: Dictionary mapping from a question's id to the split (train, valid, test) it original belonged to.
    :param convert: Convert one-hot vectors into dense data format during saving (not necessary if they have been
    converted during creation).
    """
    vector_sets = defaultdict(list)

    # Sort them back into their right sets
    for data_vec in data_vecs:
        if convert:
            vector_sets[qid2set[data_vec.question_id]].append(data_vec)
        else:
            vector_sets[qid2set[data_vec[3]]].append(data_vec)

    # Save the sets
    for set_name, contents in vector_sets.items():
        print("Saving data set {}...".format(set_name), end="", flush=True)
        save_qa_vectors(contents, "{}vqa_vecs_{}.pickle".format(target_dir, set_name), convert=convert)
        print("\rSaving data set {} complete!".format(set_name))

if __name__ == "__main__":
    # 1. Read in all the data sets to create a global question / answer vocabulary
    questions, answers, qid2set, aid2set = combine_data_sets("train", "valid", "test", unique_answers=True)

    # 2. Create one-hot vectors and the vocabulary
    data_vecs, qa_vocab = get_data_hot_vectors(questions, answers, convert=True)

    # 3. Save the vocabulary and the one-hot vectors using pickle
    save_qa_vocab(qa_vocab, "./data/qa_vocab.pickle")
    save_vectors(data_vecs, "./data/", qid2set, convert=False)

    # Example on how to load the pickled data and use it with the torch DataLoader class
    # vec_collection = VQADataset(
    #     load_path="./data/vqa_vecs_train.pickle",
    #     inflate_vecs=False,
    #     image_features_path="./data/VQA_image_features.h5",
    #     image_features2id_path="./data/VQA_img_features2id.json"
    # )
    # for vec_pair in vec_collection:
    #     print(len(vec_pair.question_vec), vec_pair.question_vec)

    #dataset_loader = DataLoader(vec_collection, batch_size=4, shuffle=True, num_workers=4)
    #for i_batch, sample_batched in enumerate(dataset_loader):
    #    print(i_batch, sample_batched)
