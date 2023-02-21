from videoqa import *
import dataloader
from build_vocab import Vocabulary
from utils import *
import argparse
import eval_mc


NUM_THREADS = 1
torch.set_num_threads(NUM_THREADS)

def main(args):

    mode = args.mode
    if mode == 'train':
        batch_size = 64
        num_worker = 8
    else:
        batch_size = 64 #you may need to change to a number that is divisible by the size of test/val set, e.g., 4
        num_worker = 8

    model_type = 'HGA' #(EVQA, STVQA, CoMem, HME, HGA)
    if model_type == 'STVQA':
        spatial = True
    else:
        spatial = False # True for STVQA

    if spatial:
        #STVQA
        video_feature_path = '../data/feats/'
        video_feature_cache = '../data/feats/cache/'
    else:
        video_feature_cache = '../data/feats/cache/'
        video_feature_path = '../data/feats/'

    dataset = 'nextqa'

    sample_list_path = 'dataset/{}/'.format(dataset)
    vocab = pkload('dataset/{}/vocab.pkl'.format(dataset))

    glove_embed = 'dataset/{}/glove_embed.npy'.format(dataset)
    use_bert = args.bert #True #Otherwise GloVe

    model_prefix= args.checkpoint + '-bert-ft-h256'

    checkpoint_path = 'videoqa_saves/' + args.checkpoint
    if os.path.exists(checkpoint_path) == False:
        os.mkdir(checkpoint_path)

    vis_step = 106 # visual step
    lr_rate = 5e-5 if use_bert else 1e-4
    epoch_num = 50


    data_loader = dataloader.QALoader(batch_size, num_worker, video_feature_path, video_feature_cache,
                                      sample_list_path, vocab, use_bert, model_type, True, False)

    train_loader, val_loader, test_loader = data_loader.run(mode=mode)

    vqa = VideoQA(vocab, train_loader, val_loader, test_loader, glove_embed, use_bert, checkpoint_path, model_type, model_prefix,
                  vis_step,lr_rate, batch_size, epoch_num)


    ep = 42
    acc = 49.42
    model_file = f'{model_type}-{model_prefix}-{ep}-{acc:.2f}.ckpt'

    if mode != 'train':
        if mode == 'val':
            result_file = f'results/{model_type}-{model_prefix}-{mode}.json'
            vqa.predict(model_file, result_file)
            eval_mc.main(result_file, mode)
        if mode == 'test':
            result_file = f'results/{model_type}-{model_prefix}-{mode}.json'
            print("======= Loading:", model_file)
            vqa.predict_test(model_file, result_file)
            eval_mc.main(result_file, mode)
    else:
        #Model for resume-training.
        model_file = checkpoint_path + f'/{model_type}-{model_prefix}-43-39.67.ckpt'
        vqa.run(model_file, pre_trained=False)



if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', type=int,
                        default=0, help='gpu device id')
    parser.add_argument('--mode', dest='mode', type=str,
                        default='train', help='train or val')
    parser.add_argument('--bert', dest='bert', action='store_true',
                        help='use bert or glove')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str,
                        default='ck', help='checkpoint name')
    args = parser.parse_args()

    main(args)
