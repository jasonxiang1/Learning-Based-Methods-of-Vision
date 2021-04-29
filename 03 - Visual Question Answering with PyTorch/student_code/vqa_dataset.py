from torch.utils.data import Dataset
import sys
import os

sys.path.append(os.getcwd())

from external.vqa.vqa import VQA

from PIL import Image
from torchvision import transforms
import torch

import numpy as np
import pickle

import time

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        # import pdb; pdb.set_trace()
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        
        if question_word_to_id_map is None:
            ############ 1.6 TODO
            self._questions = self._vqa.questions['questions']
            self._listQuestionWords = []
            for i in self._questions:
                if self._listQuestionWords == []:
                    self._listQuestionWords = self._create_word_list([i['question']])
                else:
                    self._listQuestionWords.extend(self._create_word_list([i['question']]))
                



            # to implement this line of code -> add [] to i['question']
            # self._listQuestionWords = sum(self._listQuestionWords, [])

            self.question_word_to_id_map = self._create_id_map(self._listQuestionWords, self.question_word_list_length-1)
                
            ############
            # raise NotImplementedError()
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            ############ 1.7 TODO
            self._answers = self._vqa.qa
            self._listAnswerWords = []
            for i in self._answers:
                for j in self._answers[i]['answers']:
                    self._listAnswerWords.append(j['answer'])

            # self._listAnswerWords = sum(self._listAnswerWords, [])
            self.answer_to_id_map = self._create_id_map(self._listAnswerWords, self.answer_list_length-1)


            ############
            # raise NotImplementedError()
        else:
            self.answer_to_id_map = answer_to_id_map


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """

        ############ 1.4 TODO
        lowerCaseList = []
        specChars = "!@#$%^&*;:,./<>?\|`~-=_+?()[]{}"

        for sentence in sentences:
            tempLowerSC = sentence.lower()
            tempLower = tempLowerSC
            for specChar in specChars:
                tempLower = tempLower.replace(specChar, " ")
            # lowerCaseList.append(tempLower.split())
            if lowerCaseList == []:
                lowerCaseList = tempLower.split()
            else:
                lowerCaseList.extend(tempLower.split())

        # return sum(lowerCaseList, []) # convert list of lists into flattened list
        return lowerCaseList

        ############
        # raise NotImplementedError()


    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        ############ 1.5 TODO
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

        ############
        # raise NotImplementedError()


    def __len__(self):
        ############ 1.8 TODO

        # return the number of questions
        return len(self._vqa.questions['questions'])

        ############
        # raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ############ 1.9 TODO
        # figure out the idx-th item of dataset from the VQA API
        tempQuestionID = self._vqa.questions['questions'][idx]['question_id']
        tempQuestion = self._vqa.questions['questions'][idx]['question']
        tempAnswers = self._vqa.loadQA(tempQuestionID)[0]['answers']

        tempAnswersArr = []
        for answer in tempAnswers:
            tempAnswersArr.append(answer['answer'])

        ############
        tempImageID = self._vqa.questions['questions'][idx]['image_id']
        tempImageName = '0'*(12 - len(str(tempImageID))) + str(tempImageID)
        

        if self._cache_location is not None and self._pre_encoder is not None:
            ############ 3.2 TODO
            # implement your caching and loading logic here
            # cache_dir = os.path.join(os.getcwd(), self._cache_location)
            cache_dir= self._cache_location
            fileName = 'resnet_avgpool_'+tempImageName
            fileExists = os.path.isfile(os.path.join(cache_dir, fileName+'.pkl'))

            # check if file exists in the cache location directory
            if fileExists:
                # load the file and set that as the "image" output
                try:
                    with open(os.path.join(cache_dir, fileName+'.pkl'), 'rb') as f:
                        resnetFeatureVar = pickle.load(f)
                    imageTensor = resnetFeatureVar
                except:
                    import pdb; pdb.set_trace()
                    with open(os.path.join(cache_dir, fileName+'.pkl'), 'rb') as f:
                        resnetFeatureVar = pickle.load(f)
                    imageTensor = resnetFeatureVar

            else:
                # load the image
                tempImageIDString = os.path.join(self._image_dir, self._image_filename_pattern.format(tempImageName))
                #print(tempImageIDString)
                try:
                    imagePILVar = Image.open(tempImageIDString).convert('RGB')
                except:
                    # import pdb; pdb.set_trace()
                    import pdb; pdb.set_trace()
                    imagePILVar = Image.open(tempImageIDString).convert('RGB')
                if self._transform is not None:
                    imageTensorVar = self._transform(imagePILVar)
                else:
                    imageTensorVar = transforms.ToTensor()(imagePILVar)
                # pass through image encoder (i.e. resnet18)
                if len(imageTensorVar.size()) == 3:
                    imageTensorVar = imageTensorVar.unsqueeze(0)
                output = self._pre_encoder(imageTensorVar) # (1, 512)
                output = torch.squeeze(output)
                # pickle dump the variable
                with open(os.path.join(cache_dir, fileName + '.pkl'), 'wb') as f:
                    pickle.dump(output, f)

                imageTensor = output

                # save to cache directory

            # if file is in tmp_train:
            #     pkl.load ( file )
            #     image = file
            # else:
            #     load image
            #     output = resent(image)
            #     pkl.dump(output)
            #     image = output

            ############
            # raise NotImplementedError()
        else:
            ############ 1.9 TODO
            # load the image from disk, apply self._transform (if not None)
            # tempImageIDString = self._image_dir + '\\' + self._image_filename_pattern.format(tempImageName)
            tempImageIDString = os.path.join(self._image_dir, self._image_filename_pattern.format(tempImageName))
            #print(tempImageIDString)
            imagePIL = Image.open(tempImageIDString).convert('RGB')
            if self._transform is not None:
                imageTensor = self._transform(imagePIL)
            else:
                imageTensor = transforms.ToTensor()(imagePIL)


            ############
            # raise NotImplementedError()

        ############ 1.9 TODO
        # load and encode the question and answers, convert to torch tensors
        questionDict = self._create_id_map(self._create_word_list([tempQuestion]), self._max_question_length)
        
        questionOneHotArray = torch.zeros(self._max_question_length,len(self.question_word_to_id_map)+1)
        for i in range(len(questionDict.values())):
            # tempquestionOneHotVect = torch.zeros(len(self.question_word_to_id_map)+1)
            for j in range(len(self.question_word_to_id_map)):
                if self.question_word_to_id_map[j] == questionDict[i]:
                    # tempquestionOneHotVect[j] = 1
                    questionOneHotArray[i,j] = 1
                    break
            if 1 not in questionOneHotArray[i]:
                # tempquestionOneHotVect[-1] = 1
                questionOneHotArray[i,-1] = 1

        # set all indices after the question to be 1
        if len(questionDict) < self._max_question_length:
            questionOneHotArray[len(questionDict):, -1] = 0

            # questionOneHotArray = torch.cat((questionOneHotArray, tempquestionOneHotVect.unsqueeze(0)), 0)
            # questionOneHotArray.append(tempquestionOneHotVect)
        # questionOneHotArray = questionOneHotArray[1:]
        # create stacked one hot for answers
        # print("testing")

        # create torch vector of zeros
        answerOneHotArr = torch.zeros(len(tempAnswersArr),len(self.answer_to_id_map)+1)

        for i in range(len(tempAnswersArr)):
            for j in range(len(answerOneHotArr[0])-1):
                if self.answer_to_id_map[j] in tempAnswersArr[i]:
                    answerOneHotArr[i,j] = 1
            if 1 not in answerOneHotArr[i,:]:
                answerOneHotArr[i,-1] = 1

        return {'image': imageTensor, 'questions': questionOneHotArray, 'answers': answerOneHotArr}



        # if word from the list is in the 
        # for answer in tempAnswersArr:
        #     # create word list of the answer
        #     tempAnswerWordList = 





        ############
        # raise NotImplementedError()
