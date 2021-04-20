from torch.utils.data import DataLoader
import torch

import torch.nn as nn


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps


        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation

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

        softmaxFunc = nn.Softmax(dim=1)

        for batch_id, batch_data in enumerate(self._val_dataset_loader):

            image = batch_data['image'].to('cuda')
            questions = batch_data['questions'].to('cuda')
            answers = batch_data['answers'].to('cuda')

            num_val_samples += questions.size()[0]      

            predicted_answer_array = torch.sum(answers,1).float()
            predicted_answer_array = torch.argmax(predicted_answer_array, 1)

            output = self._model(image,questions)
            output = softmaxFunc(output)


            output_answer = torch.argmax(output,1)

            num_batch_correct = predicted_answer_array == output_answer
            
            num_val_correct += torch.sum(num_batch_correct.int())
            
                              
        ############

        if self._log_validation:
            ############ 2.9 TODO
            # you probably want to plot something here
            pass

            ############
        # return accuracy value
        
        # raise NotImplementedError()
        return num_val_correct / num_val_samples

    def train(self):

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
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

                    ############

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    ############ 2.9 TODO
                    # you probably want to plot something here

                    ############
