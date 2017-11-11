# -*- coding: utf-8 -*-
"""
Preparing data for later experiments.
"""

# STD
import json
import gzip
import os

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


def get_data_set(set_name, unique_answers=False):
    """
    Retrieve all the question and their respective answers from a data set (possible options are
    "test", "valid", and "train").

    :param unique_answers: Flag to indicate whether duplicate answers to a question should be kept.
    :type unique_answers: bool
    """
    questions = dict()

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
            answer_id = raw_answer["answer_id"]
            answer = raw_answer["answer"]

            if answer not in question or not unique_answers:
                question.answers[answer_id] = Answer(
                    uid="{}/{}".format(question_id, answer_id), image_id=question.image_id, answer=answer,
                    confidence=raw_answer["answer_confidence"], atype=answer_type
                )

    return questions


def read_data_file(path):
    """
    Read a json data set from a gzipped file.
    """
    with gzip.open(path, "rb") as file:
        return json.loads(file.read())


if __name__ == "__main__":
    for _, question in get_data_set("valid", unique_answers=True).items():
        print(question)
        for answer in question:
            print(answer)

