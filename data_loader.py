import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils import text_helper
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, transform=None):
        self.input_dir = input_dir
        #self.vqa = np.load(input_dir+'/'+input_vqa)
        self.vqa = np.load(input_dir+'/'+input_vqa, allow_pickle=True)
        self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
        self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
        self.max_qst_length = max_qst_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
        self.transform = transform

    def __getitem__(self, idx):

        vqa = self.vqa
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans

        reduceImage=True #change to False if want original model
        k=100 #how many singular values 

        image = vqa[idx]['image_path']
        image = Image.open(image).convert('RGB')
        if reduceImage:
          image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
          blue, green, red = cv2.split(image)

          bU, bs, bVt = np.linalg.svd(blue, full_matrices=False)
          bV = bVt.T
          bS = np.diag(bs)
          blueNew = np.dot(bU[:, :k], np.dot(bS[:k, :k], bV[:,:k].T))

          
          gU, gs, gVt = np.linalg.svd(green, full_matrices=False)
          gV = gVt.T
          gS = np.diag(gs)
          greenNew = np.dot(gU[:, :k], np.dot(gS[:k, :k], gV[:,:k].T))

          rU, rs, rVt = np.linalg.svd(red, full_matrices=False)
          rV = rVt.T
          rS = np.diag(rs)
          redNew = np.dot(rU[:, :k], np.dot(rS[:k, :k], rV[:,:k].T))
          img_reduced = (np.dstack((redNew, greenNew, blueNew))).astype(np.uint8)
          image=img_reduced
        # # # print(np.array(image).shape) #224, 224, 3

        # # # image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        # # #       #initialize PCA with first 20 principal components
        # pca = PCA(223)
        
        # #Applying to red channel and then applying inverse transform to transformed array.
        # red_transformed = pca.fit_transform(red)
        # red_inverted = pca.inverse_transform(red_transformed)
        
        # #Applying to Green channel and then applying inverse transform to transformed array.
        # green_transformed = pca.fit_transform(green)
        # green_inverted = pca.inverse_transform(green_transformed)
        
        # #Applying to Blue channel and then applying inverse transform to transformed array.
        # blue_transformed = pca.fit_transform(blue)
        # blue_inverted = pca.inverse_transform(blue_transformed)
        # img_reduced = (np.dstack((red_inverted, blue_inverted, green_inverted))).astype(np.uint8)

        # # print('image', np.array(image))
        # # print('red', img_reduced)
        # # print('diff', np.array(img_reduced)-np.array(image))
        # image=img_reduced
        
        # plt.imshow(image)
        # plt.show()
        # df_blue = blue/255
        # df_green = green/255
        # df_red = red/255
        # pca_b = PCA(n_components=224)
        # pca_b.fit(df_blue)
        # trans_pca_b = pca_b.transform(df_blue)
        # pca_g = PCA(n_components=100)
        # pca_g.fit(df_green)
        # trans_pca_g = pca_g.transform(df_green)
        # pca_r = PCA(n_components=100)
        # pca_r.fit(df_red)
        # trans_pca_r = pca_r.transform(df_red)
        # print(idx)
        # print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
        # print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
        # print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")


        # b_arr = pca_b.inverse_transform(trans_pca_b)
        # g_arr = pca_g.inverse_transform(trans_pca_g)
        # r_arr = pca_r.inverse_transform(trans_pca_r)

        # print(b_arr.shape, g_arr.shape, r_arr.shape)

        # img_reduced= (cv2.merge((b_arr, g_arr, r_arr)))
        # print('im reducd, ', img_reduced.shape)

        # image=Image.fromarray(np.array(img_reduced))
        # image=img_reduced
        # plt.imshow(image)
        # plt.show()




        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
        qst2idc[:len(vqa[idx]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]
        sample = {'image': image, 'question': qst2idc}

        if load_ans:
            ans2idc = [ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers']] #valid answers index
            ans2idx = np.random.choice(ans2idc) #random answer selected as right
            sample['answer_label'] = ans2idx         # for training, set answer

            mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
            sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice', setting 'answer_multi_choice'
                                                        #to valid answers index list
        if transform:
            sample['image'] = transform(sample['image'])

        return sample

    def __len__(self):

        return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):

    transform = {
        phase: transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))]) 
        for phase in ['train', 'valid']}

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['train']),
        'valid': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['valid'])}

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['train', 'valid']}

    return data_loader
