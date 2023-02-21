import torch
from torch.utils.data import Dataset, DataLoader
from .util import load_file
import os.path as osp
import numpy as np
import nltk
import h5py
import spacy
from tqdm import tqdm

class VidQADataset(Dataset):
    """load the dataset in dataloader"""

    def __init__(self, video_feature_path, video_feature_cache, sample_list_path, vocab, use_bert, mode, task):
        self.video_feature_path = video_feature_path
        self.vocab = vocab
        sample_list_file = osp.join(sample_list_path, '{}.csv'.format(mode))
        self.sample_list = load_file(sample_list_file)
        self.video_feature_cache = video_feature_cache
        self.max_qa_length = 37
        self.use_bert = use_bert
        if task == 'STVQA':
            self.use_frame = False  # False for STVQA
            self.use_mot = False  # False for STVQA
            self.use_spatial = True
        else:
            self.use_frame = True  # False for STVQA
            self.use_mot = True  # False for STVQA
            self.use_spatial = False  # True for STVQA
        self.mode = mode

        if self.use_bert:
            self.bert_file = osp.join(video_feature_path, 'qas_bert/bert_ft_{}.h5'.format(mode))

        if not self.use_spatial:
            vid_feat_file = osp.join(video_feature_path, 'vid_feat/app_mot_{}.h5'.format(mode))
            print('Load {}...'.format(vid_feat_file))
            self.frame_feats = {}
            self.mot_feats = {}
            with h5py.File(vid_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['feat']
                for id, (vid, feat) in enumerate(zip(vids, feats)):
                    if self.use_frame:
                        self.frame_feats[str(vid)] = feat[:, :2048]  # (16, 2048)
                    if self.use_mot:
                        self.mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)
        else:
            # if you don't have enough memory(>60G), you can read feature from hdf5 at each iteration
            self.vid_feat_file = osp.join(self.video_feature_path, 'spatial_feat/feat_maps_{}_new.h5'.format(self.mode))


    def __len__(self):
        return len(self.sample_list)


    def get_video_feature(self, video_name):
        """
        :param video_name:
        :return:
        """

        if self.use_spatial:
            self.spatial_feats = h5py.File(self.vid_feat_file, 'r')
            video_feature = np.array(self.spatial_feats[video_name])
        else:
            if self.use_frame:
                app_feat = self.frame_feats[video_name]
                video_feature = app_feat  # (16, 2048)
            if self.use_mot:
                mot_feat = self.mot_feats[video_name]
                video_feature = np.concatenate((video_feature, mot_feat), axis=1)  # (16, 4096)

        return torch.from_numpy(video_feature).type(torch.float32)


    def get_word_idx(self, text):
        """
        """
        tokens = nltk.tokenize.word_tokenize(str(text).lower())
        token_ids = [self.vocab(token) for i, token in enumerate(tokens) if i < 25]

        return token_ids

    def get_trans_matrix(self, candidates):

        qa_lengths = [len(qa) for qa in candidates]
        candidates_matrix = torch.zeros([5, self.max_qa_length]).long()
        for k in range(5):
            sentence = candidates[k]
            candidates_matrix[k, :qa_lengths[k]] = torch.Tensor(sentence)

        return candidates_matrix, qa_lengths


    def __getitem__(self, idx):
        """
        """
        cur_sample = self.sample_list.loc[idx]
        video_name, qns, ans, qid = str(cur_sample['video']), str(cur_sample['question']),\
                                    int(cur_sample['answer']), str(cur_sample['qid'])


        candidate_qas = []
        candidate_as = []
        candidate_qs = []
        qns2ids = [self.vocab('<start>')]+self.get_word_idx(qns)+[self.vocab('<end>')] # start:1, end:2

        try:
            qns_tag_word = cur_sample['Noun'].split() # string
            qns_tag = torch.zeros(self.max_qa_length).long()
            for idx, word in enumerate(qns_tag_word):
                qns_tag[idx] = self.get_word_idx(word)[0]
        except:
            qns_tag = torch.zeros(self.max_qa_length).long()

        for id in range(5):
            cand_ans = cur_sample['a'+str(id)]
            ans2id = self.get_word_idx(cand_ans) + [self.vocab('<end>')]
            candidate_qas.append(qns2ids+ans2id)
            candidate_as.append(ans2id)
            candidate_qs.append(qns2ids)

        candidate_qas, qa_lengths = self.get_trans_matrix(candidate_qas)
        candidate_as, a_lengths = self.get_trans_matrix(candidate_as)
        # candidate_qs, q_lengths = self.get_trans_matrix(candidate_qs)
        if self.use_bert:
            with h5py.File(self.bert_file, 'r') as fp:
                temp_feat = fp['feat'][idx]
                candidate_qas = torch.from_numpy(temp_feat).type(torch.float32)

            for i in range(5):
                valid_row = nozero_row(candidate_qas[i])
                qa_lengths[i] = valid_row
                valid_ans_row = nozero_row(candidate_as[i])
                a_lengths[i] = valid_ans_row

        video_feature = self.get_video_feature(video_name)
        qns_key = video_name + '_' + qid
        qa_lengths = torch.tensor(qa_lengths)
        # a_lengths = torch.tensor(a_lengths)
        # q_lengths = torch.tensor(q_lengths)
        

        return video_feature, candidate_qas, qa_lengths, ans , qns_key
        # return video_feature, candidate_qas, qa_lengths, ans , qns_key , candidate_as, a_lengths , candidate_qs, q_lengths#, qns_tag #, cand_as_raw


def nozero_row(A):
    i = 0
    for row in A:
        if row.sum()==0:
            break
        i += 1

    return i



class QALoader():
    def __init__(self, batch_size, num_worker, video_feature_path, video_feature_cache,
                 sample_list_path, vocab, use_bert, task, train_shuffle=True, val_shuffle=False):
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.video_feature_path = video_feature_path
        self.video_feature_cache = video_feature_cache
        self.sample_list_path = sample_list_path
        self.vocab = vocab
        self.use_bert = use_bert

        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.task = task


    def run(self, mode=''):
        if mode != 'train':
            train_loader = ''
            val_loader = self.validate(mode)
            test_loader = self.validate(mode)
        else:
            train_loader = self.train('train')
            val_loader = self.validate('val')
            test_loader = self.validate('test')
        return train_loader, val_loader, test_loader


    def train(self, mode):

        training_set = VidQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                       self.vocab, self.use_bert, mode, self.task)

        print('Eligible video-qa pairs for training : {}'.format(len(training_set)))
        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_worker
            )

        return train_loader


    def validate(self, mode):

        validation_set = VidQADataset(self.video_feature_path, self.video_feature_cache, self.sample_list_path,
                                         self.vocab, self.use_bert, mode, self.task)

        print('Eligible video-qa pairs for validation : {}'.format(len(validation_set)))
        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_worker
            )

        return val_loader


if __name__ == '__main__':
    mode = 'train'
    if mode == 'train':
        batch_size = 1
        num_worker = 1

    spatial = False
    if spatial:
        # STVQA
        video_feature_path = '../data/feats/spatial/'
        video_feature_cache = '../data/feats/cache/'
    else:
        video_feature_cache = '../data/feats/cache/'
        video_feature_path = '../data/feats/'

    dataset = 'nextqa'

    sample_list_path = 'dataset/{}/'.format(dataset)
    use_bert = True  # True #Otherwise GloVe
    vocab = ''

    model_type = 'HGA'  # (EVQA, CoMem, HME, HGA)

    vis_step = 106  # visual step
    lr_rate = 5e-5 if use_bert else 1e-4
    epoch_num = 80

    data_loader = QALoader(batch_size, num_worker, video_feature_path, video_feature_cache,
                                      sample_list_path, vocab, use_bert, True, False)
    train_loader, val_loader, test_loader = data_loader.run()

    for i in tqdm(range(len(train_loader))):
        res = train_loader[i]
