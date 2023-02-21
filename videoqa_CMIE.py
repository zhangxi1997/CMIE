from networks import EncoderRNN, embed_loss
from networks.VQAModel import HGA, HGA_CMIE

from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import time
from tqdm import tqdm


class VideoQA():
    def __init__(self, vocab, train_loader, val_loader, test_loader, glove_embed, use_bert, checkpoint_path, model_type,
                 model_prefix, vis_step, lr_rate, batch_size, epoch_num):
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.glove_embed = glove_embed
        self.use_bert = use_bert
        self.model_dir = checkpoint_path
        self.model_type = model_type
        self.model_prefix = model_prefix
        self.vis_step = vis_step
        self.lr_rate = lr_rate
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self):

        vid_dim = 2048 + 2048
        hidden_dim = 256
        word_dim = 300
        vocab_size = len(self.vocab)
        max_ans_len = 7
        max_vid_len = 16
        max_qa_len = 37

        if self.model_type == 'HGA':
            #AAAI20
            hidden_dim = 256
            # vid_dim_new = 2048+2048+2048
            vid_encoder = EncoderRNN.EncoderVidHGA(vid_dim, hidden_dim, input_dropout_p=0.3,
                                                     bidirectional=False, rnn_cell='gru')

            qns_encoder = EncoderRNN.EncoderQns(word_dim, hidden_dim, vocab_size, self.glove_embed, self.use_bert, n_layers=1,
                                                rnn_dropout_p=0, input_dropout_p=0.3, bidirectional=False,
                                                rnn_cell='gru') # n_layers = 2

            self.model = HGA_CMIE.HGA(vid_encoder, qns_encoder, self.device) # for HGA+CMIE

        params = [{'params':self.model.parameters()}]

        self.optimizer = torch.optim.Adam(params = params, lr=self.lr_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', factor=0.5, patience=5, verbose=True)
        self.model.to(self.device)
        self.criterion = embed_loss.MultipleChoiceLoss().to(self.device)


    def save_model(self, epoch, acc):
        torch.save(self.model.state_dict(), osp.join(self.model_dir, '{}-{}-{}-{:.2f}.ckpt'
                                                     .format(self.model_type, self.model_prefix, epoch, acc)))
    def save_model_test(self, epoch, acc):
        torch.save(self.model.state_dict(), osp.join(self.model_dir, 'test-{}-{}-{}-{:.2f}.ckpt'
                                                     .format(self.model_type, self.model_prefix, epoch, acc)))

    def resume(self, model_file):
        """
        initialize model with pretrained weights
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        print(f'Warm-start (or test) with model: {model_path}')
        model_dict = torch.load(model_path)
        new_model_dict = {}
        for k, v in self.model.state_dict().items():
            if k in model_dict:
                v = model_dict[k]
            else:
                pass
                # print(k)
            new_model_dict[k] = v
        self.model.load_state_dict(new_model_dict)

    def run(self, model_file, pre_trained=False):
        self.build_model()
        best_eval_score = 0.0
        best_test_score = 0.0
        start_epoch_num = 0
        eval_score_all = []
        test_score_all = []
        if pre_trained:
            if os.path.exists(model_file):
                self.resume(model_file)
                start_epoch_num = int(model_file.split('-')[-2])
                best_eval_score = self.eval(start_epoch_num)
                print('Resume: Initial Acc {:.2f}'.format(best_eval_score))
            else:
                print('Resume: cannot find the model file')

        for epoch in range(1 + start_epoch_num, self.epoch_num):
            train_loss, train_acc, train_acc_counter = self.train(epoch)
            eval_score, eval_score_counter = self.eval(self.val_loader, epoch)
            print("==>Epoch:[{}/{}][Train Loss: {:.4f} Train acc: {:.2f} Val acc: {:.2f} Val acc counter: {:.2f}]".
                  format(epoch, self.epoch_num, train_loss, train_acc, eval_score, eval_score_counter))
            self.scheduler.step(eval_score)

            self.save_model(epoch, eval_score)  # save models
            eval_score_all.append(eval_score)
        
        # test for all
        print("\n\n=================== Testing phase ================")
        for epoch in range(1 + start_epoch_num, self.epoch_num):
            model_file = osp.join(self.model_dir,
                                  '{}-{}-{}-{:.2f}.ckpt'.format(self.model_type, self.model_prefix, epoch,
                                                                eval_score_all[epoch - 1]))
            if os.path.exists(model_file):
                self.resume(model_file)
                start_epoch_num = int(model_file.split('-')[-2])
                test_score, _ = self.eval(self.test_loader, epoch)
                test_score_all.append(test_score)
                print("Epoch:", epoch, " Test acc:", test_score)
        print("Best Val Acc:", max(eval_score_all))
        print("Best Test Acc:", max(test_score_all))


    def train(self, epoch):
        print('==>Epoch:[{}/{}][lr_rate: {}]'.format(epoch, self.epoch_num, self.optimizer.param_groups[0]['lr']))
        self.model.train()
        total_step = len(self.train_loader)
        epoch_loss = 0.0
        prediction_list = []
        answer_list = []
        precition_list_counter = []
        for iter, inputs in enumerate(tqdm(self.train_loader)):
            videos, qas, qas_lengths, answers, _ = inputs
            video_inputs = videos.to(self.device)
            qas_inputs = qas.to(self.device)
            ans_targets = answers.to(self.device)

            out, out_counterfactual, prediction, prediction_counter = self.model(video_inputs, qas_inputs, qas_lengths)

            self.model.zero_grad()
            loss_origin = self.criterion(out, ans_targets)

            # counterfactual interaction learning / interactive loss
            loss_counterfactual = self.criterion(out_counterfactual, ans_targets)

            lambda_1 = float(1.7)  # or 1.5
            loss = loss_origin + lambda_1 * loss_counterfactual

            loss.backward()
            self.optimizer.step()
            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            if iter % self.vis_step == 0:
                print('\t[{}/{}]-{}  loss: {:.4f} loss_counter1: {:.4f}'.\
                      format(iter, total_step, cur_time, loss.item(), loss_counterfactual.item()))

            epoch_loss += loss.item()

            prediction_list.append(prediction)
            answer_list.append(answers)
            precition_list_counter.append(prediction_counter)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers==ref_answers).numpy()

        predict_answers_counter = torch.cat(precition_list_counter, dim=0).long().cpu()
        acc_num_counter1 = torch.sum(predict_answers_counter == ref_answers).numpy()

        return epoch_loss / total_step, acc_num*100.0 / len(ref_answers), acc_num_counter1*100.0 / len(ref_answers)


    def eval(self, eval_loader, epoch):
        print('==>Epoch:[{}/{}][validation stage]'.format(epoch, self.epoch_num))
        self.model.eval()
        total_step = len(eval_loader)
        acc_count = 0
        prediction_list = []
        prediction_list_counterfactual = []
        # prediction_list_counterfactual2 = []
        answer_list = []
        with torch.no_grad():
            for iter, inputs in enumerate(tqdm(eval_loader)):
                videos, qas, qas_lengths, answers, _ = inputs
                video_inputs = videos.to(self.device)
                qas_inputs = qas.to(self.device)

                out, out_counterfactual, prediction, prediction_counterfactual = self.model(video_inputs, qas_inputs, qas_lengths)

                prediction_list.append(prediction)
                prediction_list_counterfactual.append(prediction_counterfactual)
                answer_list.append(answers)

        predict_answers = torch.cat(prediction_list, dim=0).long().cpu()
        predict_answers_counter = torch.cat(prediction_list_counterfactual, dim=0).long().cpu()

        ref_answers = torch.cat(answer_list, dim=0).long()
        acc_num = torch.sum(predict_answers == ref_answers).numpy()
        acc_num_counter = torch.sum(predict_answers_counter == ref_answers).numpy()
        return acc_num*100.0 / len(ref_answers), acc_num_counter*100.0 / len(ref_answers) #, acc_num_counter2*100.0 / len(ref_answers)


    def predict(self, model_file, result_file):
        """
        predict the answer with the trained model
        :param model_file:
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        self.build_model()
        if self.model_type in ['HGA', 'STVQA']:
            self.resume(model_file)
        else:
            old_state_dict = torch.load(model_path)
            self.model.load_state_dict(old_state_dict)

        self.model.eval()
        results = {}
        with torch.no_grad():
            print("Now predict")
            for iter, inputs in enumerate(tqdm(self.val_loader)):
                videos, qas, qas_lengths, answers, qns_keys = inputs
                video_inputs = videos.to(self.device)
                qas_inputs = qas.to(self.device)
                out, _, prediction, _ = self.model(video_inputs, qas_inputs, qas_lengths)
                prediction = prediction.data.cpu().numpy()
                answers = answers.numpy()
                for qid, pred, ans in zip(qns_keys, prediction, answers):
                    results[qid] = {'prediction': int(pred), 'answer': int(ans)}

        print(len(results))
        save_file(results, result_file)

    def predict_test(self, model_file, result_file):
        """
        predict the answer with the trained model
        :param model_file:
        :return:
        """
        model_path = osp.join(self.model_dir, model_file)
        self.build_model()
        if self.model_type in['HGA', 'STVQA']:
            self.resume(model_file)
        else:
            old_state_dict = torch.load(model_path)
            self.model.load_state_dict(old_state_dict)

        self.model.eval()
        results = {}
        video_id_list = []
        pred_as_list = []

        with torch.no_grad():
            for iter, inputs in enumerate(tqdm(self.test_loader)):
                videos, qas, qas_lengths, answers, qns_keys = inputs
                video_inputs = videos.to(self.device)
                qas_inputs = qas.to(self.device)
                out, _, prediction, _ = self.model(video_inputs, qas_inputs, qas_lengths)
                prediction = prediction.data.cpu().numpy()
                answers = answers.numpy()
                for qid, pred, ans in zip(qns_keys, prediction, answers):
                    results[qid] = {'prediction': int(pred), 'answer': int(ans)}

        print(len(results))
        save_file(results, result_file)






