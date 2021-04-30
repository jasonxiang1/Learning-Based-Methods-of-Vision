import sys
import os

sys.path.append(os.getcwd())

from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch.optim

class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers, cache_location, lr, log_validation):

        ############ 2.3 TODO: set up transform

        # resize to 224 x 224
        # normalize

        transform = transforms.Compose([transforms.Resize((224,224)), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
                                        ])

        ############

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   transform=transform,
                                   ############ 2.4 TODO: fill in the arguments
                                   question_word_to_id_map=None,
                                   answer_to_id_map=None,
                                   ############
                                   )
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 transform=transform,
                                 ############ 2.4 TODO: fill in the arguments
                                 question_word_to_id_map=train_dataset.question_word_to_id_map,
                                 answer_to_id_map=train_dataset.answer_to_id_map,
                                 ############
                                 )

        model = SimpleBaselineNet(len(train_dataset.question_word_to_id_map)+1, len(train_dataset.answer_to_id_map)+1)

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

        # xavier initialize the model
        nn.init.xavier_uniform_(self._model.wordEmbedding.weight)
        nn.init.xavier_uniform_(self._model.combinedOutput[0].weight)

        momentum = 0.9
        weight_decay = 1e-4
        ############ 2.5 TODO: set up optimizer
        self.optimizer_word = torch.optim.SGD(self._model.wordEmbedding.parameters(), lr=0.8, weight_decay = weight_decay) # , momentum=momentum, weight_decay=weight_decay
        self.optimizer_softmax = torch.optim.SGD(self._model.combinedOutput.parameters(), lr=0.01, weight_decay = weight_decay) # , momentum=momentum, weight_decay=weight_decay
        ############


    def _optimize(self, predicted_answers, true_answer_ids):
        ############ 2.7 TODO: compute the loss, run back propagation, take optimization step.
        lossFunc = nn.CrossEntropyLoss()

        self.optimizer_word.zero_grad()
        self.optimizer_softmax.zero_grad()

        wordClamp = 1500.0
        softmaxClamp = 20.0
        grad_value = 20.0
        nn.utils.clip_grad_value_(self._model.wordEmbedding.parameters(), clip_value = grad_value)
        nn.utils.clip_grad_value_(self._model.combinedOutput.parameters(), clip_value = grad_value)


        # self._model.wordEmbedding.weight.data = self._model.wordEmbedding.weight.clamp(-wordClamp, wordClamp)
        # self._model.combinedOutput[0].weight.data = self._model.combinedOutput[0].weight.clamp(-softmaxClamp, softmaxClamp)


        loss = lossFunc(predicted_answers, true_answer_ids)


        # for name,param in self._model.wordEmbedding.named_parameters():
        #     if 'weight' in name:
        #         param = param.clamp(-wordClamp, wordClamp)
        #         self.
        #         # use getattr(self._model.wordEmbedding,name)()

        #     print("test")
        
        # for name,param in self._model.combinedOutput.named_parameters():
        #     print("test")

        loss.backward()

        self.optimizer_softmax.step()
        self.optimizer_word.step()


        ############
        # raise NotImplementedError()
        return loss.item()
