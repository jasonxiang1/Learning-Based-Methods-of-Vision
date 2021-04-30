from torch.utils.data import DataLoader
import torch

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 20  # Steps
        self._test_freq = 500  # Steps
        self._validation_maxSamples = 1500
        self.batch_size = batch_size


        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

        self.writer = SummaryWriter("runs/VQD_Simple_02")

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        ############ 2.8 TODO
        # Should return your validation accuracy
        num_val_samples = 0
        num_val_correct = 0
        num_val_batches = len(self._val_dataset_loader)

        softmaxFunc = nn.Softmax(dim=1)
        print("Begin Validation Loop")
        # import pdb; pdb.set_trace()

        
        for valbatch_id, valbatch_data in enumerate(self._val_dataset_loader):
            # print("Validation: ")
            # print(valbatch_id)
            current_step = valbatch_id * self.batch_size
            # print("Validation Step: ")
            # print(valbatch_id)

            image = valbatch_data['image'].to('cuda')
            
            questions = valbatch_data['questions'].to('cuda')

            answers = valbatch_data['answers'].to('cuda')

            num_val_samples += questions.size()[0]      

            predicted_answer_array = torch.sum(answers,1).float()
            predicted_answer_array = torch.argmax(predicted_answer_array, 1)

            output = self._model(image,questions)
            output = softmaxFunc(output)


            output_answer = torch.argmax(output,1)

            # import pdb; pdb.set_trace()
            num_batch_correct = predicted_answer_array.int() == output_answer.int()
            
            num_val_correct += torch.sum(num_batch_correct.int())

            if current_step > self._validation_maxSamples:
                break
            
                              
        ############
        # debugging log_validation
        # TODO: understand how to set log_validation in the main.py file to be true
        # import pdb; pdb.set_trace()
        self._log_validation = True
        if self._log_validation:
            ############ 2.9 TODO
            # you probably want to plot something here
            
            log_idx = 0
            questionDict = self._val_dataset_loader.dataset.question_word_to_id_map
            answerDict = self._val_dataset_loader.dataset.answer_to_id_map

            valImage = image[log_idx]
            normImageTransform = transforms.Compose([
                transforms.Normalize(mean=[0.,0.,0.], std=[1/0.229, 1/0.224, 1/0.225]), 
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
            ])
            valImage = normImageTransform(valImage)

            self.writer.add_image("Input Image", valImage)


            # convert question to string
            # import pdb; pdb.set_trace()
            questionString = self.hot_vector_to_string(questions[log_idx], questionDict)
            self.writer.add_text("Input Question", questionString)
            answerString = self.hot_vector_to_string(output_answer[log_idx], answerDict)
            self.writer.add_text("Predicted Answer", answerString)
            gtanswer = torch.sum(answers[log_idx], 0)
            gtanswer = torch.argmax(gtanswer)
            gtanswerString = self.hot_vector_to_string(gtanswer, answerDict)
            self.writer.add_text("Ground Truth Answer", gtanswerString)
            # self.writer.add_answer("")


            ############
        # return accuracy value
        
        # raise NotImplementedError()
        # import pdb; pdb.set_trace()
        return int(num_val_correct) / num_val_samples

    def hot_vector_to_string(self, hotVect, dictVar):

        # determine the length of the dictionary
        lenDict = len(dictVar)

        if len(hotVect.size()) > 1:
            # find indices where hotVect is non zero along dim=1
            hotVectIndices = torch.nonzero(hotVect, as_tuple=False)

            # remove all indices that are the

            hotVectIndices = hotVectIndices[hotVectIndices[:,1] != lenDict]
            hotVectIndices = hotVectIndices[:,1]
            
            lenHotVect = len(hotVectIndices)
        else:
            if int(hotVect) == lenDict:
                return 'N/A'
            else:
                return dictVar[int(hotVect)]

        stringVar = []

        for i in hotVectIndices:
            if len(stringVar) == 0:
                # import pdb; pdb.set_trace()
                stringVar = [dictVar[int(i)]]
            else:
                stringVar.extend([dictVar[int(i)]])

        stringVar = ' '.join(stringVar)

        return stringVar


    def train(self):
        
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)
            # import pdb; pdb.set_trace()
            print("Begin Training Loop for Epoch {}".format(epoch))

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                # print("Training: ")
                # print(batch_id)
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                image = batch_data['image'].to('cuda')
                questions = batch_data['questions'].to('cuda')
                answers = batch_data['answers'].to('cuda')

                # compute one hot for the ground truth answers
                predicted_answer_array = torch.sum(answers,1).float()
                predicted_answer_array = torch.argmax(predicted_answer_array, 1)

                output = self._model(image,questions)

                predicted_answer = output # TODO
                ground_truth_answer = predicted_answer_array # TODO

                ############

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar('Training Loss', loss, current_step)

                    # print(self._model.wordEmbedding.weight.data)

                    ############

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here
                    self.writer.add_scalar('Validation Accuracy', val_accuracy, current_step)

                    ############

        self.writer.close()