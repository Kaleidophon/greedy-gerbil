"""
Analyse question or answer vocabularies.
"""

# STD
from collections import defaultdict

# PROJECT
from data_loading import combine_data_sets
from one_hot import get_vocabulary


def get_freq_distribution(freqs):
    freq_distribution = defaultdict(int)

    for _, freq in freqs.items():
        freq_distribution[freq] += 1

    return freq_distribution


if __name__ == "__main__":
    questions, answers, _, _ = combine_data_sets("train", "valid", "test", unique_answers=True)

    question_vocabulary, _, _, question_freqs = get_vocabulary(
        questions.values(), entry_getter=lambda question: question.question.replace("?", "").split(" "),
        index_vocab=False
    )
    answer_vocabulary, _, _, answer_freqs = get_vocabulary(
        list(answers.values()), entry_getter=lambda answer: answer.answer,
        index_vocab=False
    )
    question_freq_distribution = get_freq_distribution(question_freqs)
    answer_freq_distribution = get_freq_distribution(answer_freqs)
    pass