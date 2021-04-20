from torch.utils.data import Dataset
from external.vqa.vqa import VQA

from student_code.vqa_dataset import VqaDataset

import os

def create_word_list(sentences):
    """
    Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
    Args:
        sentences: a list of str, sentences to be splitted into words
    Return:
        A list of str, words from the split, order remained.
    """

    lowerCaseList = []
    specChars = "!@#$%^&*;:,./<>?\|`~-=_+?()[]{}"

    for sentence in sentences:
        tempLowerSC = sentence.lower()
        tempLower = tempLowerSC
        for specChar in specChars:
            tempLower = tempLower.replace(specChar, " ")
        lowerCaseList.append(tempLower.split())

    return lowerCaseList

def create_id_map(word_list, max_list_length):
    """
    Find the most common str in a list, then create a map from str to id (its rank in the frequency)
    Args:
        word_list: a list of str, where the most frequent elements are picked out
        max_list_length: the number of strs picked
    Return:
        A map (dict) from str to id (rank)
    """

    word_counter = {}


    for word in word_list:
        if word in word_counter:
            word_counter[word] += 1
        else:
            word_counter[word] = 1

    top_words = sorted(word_counter, key=word_counter.get, reverse = True)

    if max_list_length < len(top_words):
        wordLen = max_list_length
    else:
        wordLen = len(top_words)

    returnDict = {}

    for i in range(wordLen):
        returnDict[i] = top_words[i]

    return returnDict


if __name__ == '__main__':
    annotation_json_file_path = "data/mscoco_train2014_annotations.json"
    question_json_file_path = "data/OpenEnded_mscoco_train2014_questions.json"
    vqa = VQA(annotation_json_file_path, question_json_file_path)

    questionID = 409380

    print(vqa.loadQA([questionID]))

    # Q1.4
    sentences = ["Welcome???@@##$ to#$% Geeks%$^for$%^&Geeks", "What is the color of this?"]

    listWords = create_word_list(sentences)

    # Q1.5
    word_list = listWords[0]

    idMap = create_id_map(word_list, 10)

    # Q1.6





