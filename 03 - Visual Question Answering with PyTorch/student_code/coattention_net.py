import torch.nn as nn
import torch.utils as utils
import torch


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, questionVocabSize=5747, answerVocabSize=5217, questionLength=26): # , questionVocabSize, answerVocabSize
        super().__init__()
        ############ 3.3 TODO
        self.questionVocabSize = questionVocabSize
        self.answerVocabSize = answerVocabSize
        # assume max question length of 26
        self.questionLength = questionLength

        self.wordEmbeddingSize = 512
        # self.linear1 = nn.Linear(self.questionVocabSize, self.wordEmbeddingSize)
        # TODO: implement word embedding for better performance
        self.wordembedding1 = nn.Embedding(self.questionVocabSize, self.wordEmbeddingSize, padding_idx = self.questionVocabSize-1)

        self.innerProdDim = 512
        self.conv_unigram = nn.Conv1d(self.wordEmbeddingSize, self.innerProdDim, kernel_size=1, padding=0)
        self.conv_bigram = nn.Conv1d(self.wordEmbeddingSize, self.innerProdDim, kernel_size=2, padding=1, dilation=2)
        self.conv_trigram = nn.Conv1d(self.wordEmbeddingSize, self.innerProdDim, kernel_size=3, padding=2, dilation=2)
        self.innerProdActivation = nn.Tanh()

        self.maxpool1 = nn.MaxPool2d(kernel_size = (3,1), stride = 1)

        # TODO: pack_padded_sequence
        self.lstm1 = nn.LSTM(input_size = self.innerProdDim, hidden_size = self.innerProdDim, num_layers = 3, bidirectional=False, batch_first = True, dropout=0.5)
        # TODO: pad_packed_sequence

        # alternating coattention functions and parameters
        self.hiddenDim = 512
        self.imageDim = 14*14

        # use nn.ModuleDict to create dictionary of values
        self.alternating_coatten = nn.ModuleDict([ 
            ['nonlin_tanh', nn.Tanh()], 
            ['nonlin_softmax', nn.Softmax(dim=1)], 
            ['step1_H_Wx', nn.Linear(self.hiddenDim, self.hiddenDim)], 
            ['step1_ax_Wh', nn.Linear(self.hiddenDim, 1)], 
            ['step2_H_Wx', nn.Linear(self.hiddenDim, self.hiddenDim)], 
            ['step2_H_Wg', nn.Linear(self.hiddenDim, self.hiddenDim)], 
            ['step2_ax_Wh', nn.Linear(self.hiddenDim, 1)], 
            ['step3_H_Wx', nn.Linear(self.hiddenDim, self.hiddenDim)], 
            ['step3_H_Wg', nn.Linear(self.hiddenDim, self.hiddenDim)], 
            ['step3_ax_Wh', nn.Linear(self.hiddenDim, 1)]
        ])

        # encoding for predicting
        self.linearOut1 = nn.Linear(self.hiddenDim, self.hiddenDim)
        self.linearOut2 = nn.Linear(self.hiddenDim*2, self.hiddenDim)
        self.linearOut3 = nn.Linear(self.hiddenDim*2, self.hiddenDim)
        self.linearOut4 = nn.Linear(self.hiddenDim, self.answerVocabSize)

        # self.phrase_alternating_coatten = nn.ModuleDict([ 
        #     ['nonlin_tanh', nn.Tanh()], 
        #     ['step1_H_Wx', nn.Linear(self.hiddenDim, self.questionLength)], 
        #     ['step1_ax_Wh', nn.Linear(self.questionLength, 1)], 
        #     ['step2_H_Wx', nn.Linear(self.hiddenDim, self.imageDim)], 
        #     ['step2_H_Wg', nn.Linear(self.hiddenDim, self.hiddenDim)], 
        #     ['step2_ax', nn.Linear(self.imageDim, 1)], 
        #     ['step3_H_Wx', nn.Linear(self.hiddenDim, self.questionLength)], 
        #     ['step3_H_Wg', nn.Linear(self.hiddenDim, self.hiddenDim)], 
        #     ['step3_ax', nn.Linear(self.questionLength, 1)]
        # ])

        # self.sentence_alternating_coatten = nn.ModuleDict([ 
        #     ['nonlin_tanh', nn.Tanh()], 
        #     ['step1_H_Wx', nn.Linear(self.hiddenDim, self.questionLength)], 
        #     ['step1_ax_Wh', nn.Linear(self.questionLength, 1)], 
        #     ['step2_H_Wx', nn.Linear(self.hiddenDim, self.imageDim)], 
        #     ['step2_H_Wg', nn.Linear(self.hiddenDim, self.hiddenDim)], 
        #     ['step2_ax', nn.Linear(self.imageDim, 1)], 
        #     ['step3_H_Wx', nn.Linear(self.hiddenDim, self.questionLength)], 
        #     ['step3_H_Wg', nn.Linear(self.hiddenDim, self.hiddenDim)], 
        #     ['step3_ax', nn.Linear(self.questionLength, 1)]
        # ])
    
        ############ 

    def forward(self, image, question_encoding):
        ############ 3.3 TODO
        # image: (batch size x 512 x 14 x 14) -> (batch size x 512 x 196 x 196)
        # question_encoding: (batch size x 26 x 5747)
        self.batchSize, self.height, self.width = question_encoding.size()
        imgBatch, imgDim, imageLen1, imageLen2 = image.size()
        image = torch.reshape(image, (imgBatch, imgDim, imageLen1*imageLen2))

        # word embedding for question_encoding input
        # padded with last index of the question vocab list i.e. 5746
        questionEmbed = question_encoding.argmax(-1) # (batch size x 26)
        questionEmbed = self.wordembedding1(questionEmbed) # (batch size x 26 x 512)
        wordLevel = questionEmbed # (batch size x 26 x 512)
        questionEmbed = questionEmbed.permute(0,2,1) # (batch size x 512 x 26)

        # 1d convolution for unigram, bigram, and trigram channels
        x_unigram = self.conv_unigram(questionEmbed) # (batch size x 512 x 26)
        x_unigram = torch.unsqueeze(self.innerProdActivation(x_unigram), 2) # (batch size x 512 x 1 x 26)
        x_bigram = self.conv_bigram(questionEmbed) # (batch size x 512 x 26)
        x_bigram = torch.unsqueeze(self.innerProdActivation(x_bigram), 2) # (batch size x 512 x 1 x 26)
        x_trigram = self.conv_trigram(questionEmbed) # (batch size x 512 x 26)
        x_trigram = torch.unsqueeze(self.innerProdActivation(x_trigram), 2) # (batch size x 512 x 1 x 26)

        # combine channels together
        x_threeChannel = torch.cat((x_unigram, x_bigram, x_trigram), 2) # (batch size x 512 x 3 x 26)

        # perform channel-size max pooling
        x = self.maxpool1(x_threeChannel) # (batch size x 512 x 1 x 26)
        x = x.squeeze(2) # (batch size x 512 x 26)
        phraseLevel = x.permute(0,2,1) # (batch size x 26 x 512)

        # TODO: what do you want the output to be?
        # TODO: what to do to modify the input to the lstm layer
        x = x.permute(0,2,1) # (batch size x 26 x 512)
        lstm_x = nn.utils.rnn.pack_padded_sequence(x, lengths = torch.ones(x.shape[0])*26, batch_first = True, enforce_sorted=False)
        hs, cs = self.lstm1(lstm_x) #
        sentenceLevel, lengths = nn.utils.rnn.pad_packed_sequence(hs, batch_first = True) # (batch size x 26 x 512)
        # sentenceLevel = sentenceLevel # (batch size x 26 x 512)
        # sentenceLevel = hs.permute(0,2,1) # (batch size x 512 x 26)

        # reshape image 

        # perform the same operations for word, phrase, and sentence level embeddings
        word_vhat, word_qoutput = self.alternating_coattention(image, wordLevel, self.alternating_coatten)
        phrase_vhat, phrase_qoutput = self.alternating_coattention(image, phraseLevel, self.alternating_coatten)
        sentence_vhat, sentence_qoutput = self.alternating_coattention(image, sentenceLevel, self.alternating_coatten)

        # encode for predicting answers
        hwOutput = self.linearOut1(word_qoutput + word_vhat) # (batch size x 1 x 512)
        hwOutput = self.innerProdActivation(hwOutput) # (batch size x 1 x 512)
        # hwOutput = nn.Dropout(0.5)(hwOutput)
        hpOutput = torch.cat((phrase_qoutput+phrase_vhat, hwOutput), 2) # (batch size x 1 x 1024)
        hpOutput = self.linearOut2(hpOutput) # (batch size x 1 x 512)
        hpOutput = self.innerProdActivation(hpOutput) # (batch size x 1 x 512)
        # hpOutput = nn.Dropout(0.5)(hpOutput)
        hsOutput = torch.cat((sentence_qoutput+sentence_vhat, hpOutput), 2) # (batch size x 1 x 1024)
        hsOutput = self.linearOut3(hsOutput) # (batch size x 1 x 512)
        hsOutput = self.innerProdActivation(hsOutput) # (batch size x 1 x 512)
        # hsOutput = nn.Dropout(0.5)(hsOutput)
        returnOutput = self.linearOut4(hsOutput)
        returnOutput = returnOutput.squeeze()


        ############ 
        # raise NotImplementedError()
        return returnOutput

    def alternating_coattention(self, image, attentionVar, moduleList):

       
        # step 1
        g = 0
        H = moduleList['step1_H_Wx'](attentionVar) # (batch size x channels x 512)
        H = moduleList['nonlin_tanh'](H) # (batch size x channels x 512)
        H = nn.Dropout(0.5)(H) # (batch size x channels x 512)
        ax = moduleList['step1_ax_Wh'](H) # (batch size x channels x 1)
        ax = moduleList['nonlin_softmax'](ax) # (batch size x channels x 1)
        g = torch.bmm(ax.permute(0,2,1), attentionVar) # (batch size x 1 x 512)

        # step 2
        H_1 = moduleList['step2_H_Wx'](image.permute(0,2,1)) # (batch size x channels x 512)
        H_2 = moduleList['step2_H_Wg'](g) # (batch size x 1 x 512)
        # H_2 = torch.bmm(H_2.permute(0,2,1), torch.ones(self.batchSize, 1, self.imageDim)).permute(0,2,1) # (batch size x channels x 512)
        H_2 = torch.bmm(H_2.permute(0,2,1), torch.ones(self.batchSize, 1, self.imageDim).to('cuda')).permute(0,2,1) # (batch size x channels x 512)
        H = H_1 + H_2 # (batch size x channels x 512)
        H = moduleList['nonlin_tanh'](H) # (batch size x channels x 512)
        H = nn.Dropout(0.5)(H) # (batch size x channels x 512)
        ax = moduleList['step2_ax_Wh'](H) # (batch size x channels x 1)
        ax = moduleList['nonlin_softmax'](ax) # (batch size x channels x 1)
        vhat = torch.bmm(ax.permute(0,2,1), image.permute(0,2,1)) # (batch size x 1 x 512)

        # step 3 
        # g = xhat.permute(0,2,1)
        H_1 = moduleList['step3_H_Wx'](attentionVar) # (batch size x channels x 512)
        H_2 = moduleList['step3_H_Wg'](vhat) # (batch size x 1 x 512)
        # H_2 = torch.bmm(H_2.permute(0,2,1), torch.ones(self.batchSize, 1, self.questionLength)).permute(0,2,1) # (batch size x channels x 512)
        H_2 = torch.bmm(H_2.permute(0,2,1), torch.ones(self.batchSize, 1, self.questionLength).to('cuda')).permute(0,2,1) # (batch size x channels x 512)
        H = H_1 + H_2 # (batch size x channels x 512)
        H = moduleList['nonlin_tanh'](H) # (batch size x channels x 512)
        H = nn.Dropout(0.5)(H) # (batch size x channels x 512)
        ax = moduleList['step3_ax_Wh'](H) # (batch size x channels x 1)
        ax = moduleList['nonlin_softmax'](ax) # (batch size x channels x 1)
        qoutput = torch.bmm(ax.permute(0,2,1), attentionVar) # (batch size x 1 x 512)

        return vhat, qoutput

if __name__ == "__main__":
    print("testing coattention net model")
    from student_code.vqa_dataset import VqaDataset
    from torch.utils.data import DataLoader
    import torch
    import os
    import pickle

    currDir = os.getcwd()
    default_train_image_dir = os.path.join(currDir, 'data', 'train2014')
    default_test_image_dir = os.path.join(currDir, 'data', 'val2014')
    default_data_dir = os.path.join(currDir, 'data')
    default_train_question_path = os.path.join(default_data_dir, 'OpenEnded_mscoco_train2014_questions.json')
    default_train_annotation_path = os.path.join(default_data_dir, 'mscoco_train2014_annotations.json')
    default_test_question_path = os.path.join(default_data_dir, 'OpenEnded_mscoco_val2014_questions.json')
    default_test_annotation_path = os.path.join(default_data_dir, 'mscoco_val2014_annotations.json')


    model = CoattentionNet()


    vqa_dataset = VqaDataset(question_json_file_path=default_train_question_path,
                                annotation_json_file_path=default_train_annotation_path,
                                image_dir=default_train_image_dir,
                                image_filename_pattern="COCO_train2014_{}.jpg")
    train_dataset_loader = DataLoader(vqa_dataset, batch_size=2)

    # load a batch of image pkl files
    with open(os.path.join(currDir,'tmp_train', 'resnet_avgpool_000000581686.pkl'), 'rb') as f:
        resnetFeatureVar01 = pickle.load(f)
    with open(os.path.join(currDir,'tmp_train', 'resnet_avgpool_000000581409.pkl'), 'rb') as f:
        resnetFeatureVar02 = pickle.load(f)

    # combine features maps together
    resnetFeatureVar01 = resnetFeatureVar01.unsqueeze(0)
    resnetFeatureVar02 = resnetFeatureVar02.unsqueeze(0)
    image = torch.cat((resnetFeatureVar01,resnetFeatureVar02), 0)

    for batch_id, batch_data in enumerate(train_dataset_loader):
        questions = batch_data['questions']
        answers = batch_data['answers']
        output = model(image, questions)
        break


    print("testing done")
