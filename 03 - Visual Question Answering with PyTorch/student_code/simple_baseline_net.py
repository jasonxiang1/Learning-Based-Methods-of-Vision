import torch.nn as nn


from external.googlenet.googlenet import *
import torch

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """

    # TODO: implement parallel GoogLENet and LSTM model here:
    # one part GoogLENet, one part LSTM
    # torch.nn.LSTM
    # self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=4, bidirectional=True, dropout = 0.1)

    def __init__(self, questionVocabSize, answerVocabSize):
        super().__init__()
	    ############ 2.2 TODO
        self.embeddingOutput = 1024
        self.questionVocabSize = questionVocabSize
        self.answerVocabSize = answerVocabSize

        self.imageModel = googlenet(pretrained=True)
        # self.linear1 = nn.Linear(self.questionVocabSize, 1024)
        self.wordEmbedding = nn.Embedding(self.questionVocabSize, self.embeddingOutput, padding_idx=self.questionVocabSize-1)
        
        self.combinedOutput = nn.Sequential(
            nn.Linear(1000+self.embeddingOutput, self.answerVocabSize)# , 
            # nn.Softmax(dim=1)
        )
	    ############

    def forward(self, image, question_encoding):
	    ############ 2.2 TODO
        imageOutput = self.imageModel(image)
        if len(imageOutput) == 3:
            imageOutput = imageOutput[2]
        batchSize, height, width = question_encoding.size()

        # instead of using bag of words, convert question_encoding to vector of array indices
        # questionEmbed = torch.sum(question_encoding, 1)
        questionEmbed = torch.nonzero(question_encoding, as_tuple=False)[:,2].reshape((batchSize, height))
        wordOutput = self.wordEmbedding(questionEmbed)
        wordOutput = wordOutput.mean(dim=1)

        # wordOutput = self.linear1(questionEmbed)
        combinedVect = torch.cat((imageOutput, wordOutput), 1)
        # output from model is a tuple of (3,2,1000)
        # think about resolving the per channel outputs together

        # output row-wise softmaxS
        output = self.combinedOutput(combinedVect)

        # nn.embedding(length, hidden,)




	    ############
        # raise NotImplementedError()
        return output

if __name__ == "__main__":
    print("testing")
    model = SimpleBaselineNet()