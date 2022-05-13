import torch
import torch.nn as nn
import torchvision.models as models
import pickle5 as pickle


class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features  # input size of feature vector
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])    # remove last fc layer

        self.model = model                              # loaded model without last fc layer
        self.fc = nn.Linear(in_features, embed_size)    # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size, bag_of_words_file):

        super(QstEncoder, self).__init__()
        self.bag_of_words_file = bag_of_words_file
        if bag_of_words_file is None:
            self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
            self.tanh = nn.Tanh()
            self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
            self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states
        else:
            with open(self.bag_of_words_file, 'rb') as handle:
                self.word_index_to_array_index =  pickle.load(handle)

    def forward(self, question):
        # question is a 30-length tensor
        
        if self.bag_of_words_file is None:
            qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
            qst_vec = self.tanh(qst_vec)
            qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
            _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
            qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
            qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
            qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
            qst_feature = self.tanh(qst_feature)
            qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]
        else:
            qst_feature = torch.zeros(10, 1024)
            for batch in range(question.shape[0]):
                new_feature = torch.zeros(1024)
                for val in question[batch]:
                    if val == 1:
                        new_feature[-1] += 1
                    elif val in self.word_index_to_array_index:
                        new_feature[self.word_index_to_array_index[val]] += 1
                qst_feature[batch] = new_feature

        return qst_feature


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, bag_of_words_file=None):

        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size, bag_of_words_file)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature
