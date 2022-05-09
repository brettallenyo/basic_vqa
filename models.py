
import torch
from torch._C import device
import torch.nn as nn
import torchvision.models as models
import numpy as np

k=512
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
        # (U, S, V) = torch.pca_lowrank(img_feature, q=None, center=True, niter=2)
        # feature_reduced=torch.matmul(img_feature, V[:, :k])
        # # print(np.linalg.matrix_rank(qst_feature.cpu().detach().numpy()))
        # print(feature_reduced.shape)
        # img_feature=feature_reduced
        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]
        # (U, S, V) = torch.pca_lowrank(qst_feature, q=None, center=True, niter=2)
        # feature_reduced=torch.matmul(qst_feature, V[:, :k])
        # print(qst_feature.cpu().detach().numpy().shape)
        # print(np.linalg.matrix_rank(qst_feature.cpu().detach().numpy()))
        # print(feature_reduced.shape)
        # qst_feature=feature_reduced
        return qst_feature


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        # print(img_feature.shape)
        # (U, S, V) = torch.pca_lowrank(img_feature1, q = 10, center=True, niter=2)
        (U, S, V) = torch.svd(img_feature)
        # print('u', U.shape)
        # print('s', S.shape)
        # print('v' ,V.shape)
        img_feature=torch.matmul(img_feature.T, V.T[:, :k])
        img_feature=torch.matmul(V.T[:, :k], img_feature.T)
        # U=U[:, :k]
        # S=S[:k]
        # img_feature=torch.mul(U, S)
        # print(img_feature.shape)
        # print('temp,', temp.shape)
        # img_feature=torch.matmul(img_feature, V.T[:, :k])
        # print('img', (img_feature1-img_feature).sum())
        
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        # (U, S, V) = torch.pca_lowrank(qst_feature2, q=10, center=True, niter=2)
        # qst_feature2=torch.nan_to_num(qst_feature2)
        (U, S, V) = torch.svd(qst_feature)
        # U=U[:, :k]
        # S=S[:k]
        qst_feature=torch.matmul(qst_feature.T, V.T[:, :k])
        qst_feature=torch.matmul(V.T[:, :k], qst_feature.T)
        # qst_feature=torch.mul(U, S)
        # qst_feature=torch.matmul(qst_feature, V.T[:, :k])
        # print('qst', (qst_feature2-qst_feature).sum())
       
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        # combined_feature= torch.mul(combined_feature, 0)
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]
        # print('CSF', combined_feature.shape)
        return combined_feature

