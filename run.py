import argparse
import copy, json, os
from dataset import BRCDataset
from vocab import Vocab
import pickle
import time, math, torch
import tqdm
from model.bidaf import BiDAF
from torch import nn, optim
from eval_utils import *


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    # action='store_true'，只要运行时该变量有传参就将该变量设为True
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=int, default='1',
                        help='specify gpu device')
    parser.add_argument('--cuda', type=int, default='1',
                        help='1 use cuda')
    parser.add_argument('--dropout', type=float, default='0.1')
    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.01,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')

    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epoch', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/train_preprocessed/trainset/search.train.cut.json'],
                               help='list of files that contain the preprocessed train data')
    # path_settings.add_argument('--dev_files', nargs='+',
    #                            default=['../data/demo/devset/search.dev.json'],
    #                            help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/train_preprocessed/trainset/search.train.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='./data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./data/results/',
                               help='the dir to output the results')
    # path_settings.add_argument('--summary_dir', default='../data/summary/',
    #                            help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    print('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # logger.info('Building vocabulary...')
    print('Building vocabulary...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files)
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    # logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
    #                                                                         vocab.size()))
    print('After filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.size()))
    # logger.info('Assigning embeddings...')
    print('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size)

    print('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.cut.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    # logger.info('Done with preparing!')
    print('Done with preparing!')


def train(args):
    print('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.cut.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files)
    # logger.info('Converting text into ids...')
    print('Converting text into ids...')
    brc_data.convert_to_ids(vocab)

    print('Initialize the model...')
    model = BiDAF(args, vocab)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    device = torch.device('cuda:{}'.format(args.gpu) if args.cuda else 'cpu')
    # logger.info('Training the model...')
    print('Training the model...')
    pad_id = vocab.get_id(vocab.pad_token)

    for epoch_i in range(args.epoch):
        print('[ Epoch', epoch_i, ']')
        train_batches = brc_data.gen_mini_batches('train', args.batch_size, pad_id, shuffle=True)
        start = time.time()
        train_loss = train_epoch(
            model, train_batches, optimizer, device)
        print('Average train loss for epoch {} is {}'.format(epoch_i, train_loss))

        if epoch_i == 5:
            with torch.no_grad():
                test_batches = brc_data.gen_mini_batches('test', args.batch_size, pad_id, shuffle=True)
                bleu_rouge = evaluate(model, test_batches, device, args.result_dir, epoch_i)
                print(bleu_rouge)
    print('Done with model training!')


def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''

    model.train()

    model.to(device)
    total_loss, num_of_batch = 0, 0
    log_every_n_batch, n_batch_loss = 50, 0
    for bitx, batch in enumerate(training_data, 1):
        num_of_batch += 1
        p = torch.LongTensor(batch['passage_token_ids']).to(device)
        # batch_size x padded_q_len
        q = torch.LongTensor(batch['question_token_ids']).to(device)
        # batch_size
        start_label = torch.LongTensor(batch['start_id']).to(device)
        # batch_size
        end_label = torch.LongTensor(batch['end_id']).to(device)

        optimizer.zero_grad()
        model.zero_grad()

        # batch_size x padded_p_len x 2
        answer_prob = model(p, q, device)

        # batch_size x padded_p_len
        answer_begin_prob = answer_prob[0]
        # batch_size x padded_p_len
        answer_end_prob = answer_prob[1]

        # batch_size
        answer_begin_prob = torch.log(answer_begin_prob[range(start_label.size(0)),
                                                        start_label.data] + 1e-6)
        # batch_size
        answer_end_prob = torch.log(answer_end_prob[range(end_label.size(0)),
                                                    end_label.data] + 1e-6)
        # batch_size
        total_prob = -(answer_begin_prob + answer_end_prob)
        loss = torch.mean(total_prob)
        total_loss += loss.data
        n_batch_loss += loss.data
        if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
            print('Average loss from batch {} to {} is {}'.format(
                bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
            n_batch_loss = 0
        loss.backward()
        optimizer.step()
    return 1.0 * total_loss / num_of_batch


def evaluate(model, eval_batches, device, result_dir=None, result_prefix=None, save_full_info=False):
    """
    Evaluates the model performance on eval_batches and results are saved if specified
    Args:
        eval_batches: iterable batch data
        result_dir: directory to save predicted answers, answers will not be saved if None
        result_prefix: prefix of the file for saving predicted answers,
                       answers will not be saved if None
        save_full_info: if True, the pred_answers will be added to raw sample and saved
    """
    pred_answers, ref_answers = [], []
    total_num, num_of_batch, correct_p_num, select_total_num, select_true_num = 0, 0, 0, 0, 0
    model.eval()
    for b_itx, batch in enumerate(eval_batches):
        num_of_batch += 1
        # print("now is batch: ", b_itx)
        # batch_size * max_passage_num x padded_p_len
        p = torch.LongTensor(batch['passage_token_ids']).to(device)
        # batch_size * max_passage_num x padded_q_len
        q = torch.LongTensor(batch['question_token_ids']).to(device)
        # batch_size

        start_label = torch.LongTensor(batch['start_id']).to(device)
        end_label = torch.LongTensor(batch['end_id']).to(device)
        start, end = model(p, q, device)

        total_num += len(batch['raw_data'])
        # padded_p_len = len(batch['passage_token_ids'][0])
        # max_passage_num = p.size(0) // start_label.size(0)
        valid_answer_cnt = []
        cnt = -1
        for i, k in enumerate(batch['question_length']):
            if k != 0:
                cnt += 1
            if i % 5 == 0:
                valid_answer_cnt.append(cnt)
        valid_answer_cnt.append(cnt + 1)
        for idx, sample in enumerate(batch['raw_data']):
            select_total_num += 1
            # max_passage_num x padded_p_len
            start_prob = start[valid_answer_cnt[idx]: valid_answer_cnt[idx + 1], :]
            end_prob = end[valid_answer_cnt[idx]: valid_answer_cnt[idx + 1], :]

            best_answer, best_p_idx = find_best_answer(sample, start_prob, end_prob)
            if best_p_idx in sample['answer_passages']:
                correct_p_num += 1
            if sample['passages'][best_p_idx]['is_selected']:
                select_true_num += 1
            print('question:{},answer:{}'.format(sample['question'], best_answer))
            if save_full_info:
                sample['pred_answers'] = [best_answer]
                pred_answers.append(sample)
            else:
                pred_answers.append({'question_id': sample['question_id'],
                                     'question_type': sample['question_type'],
                                     'answers': [best_answer],
                                     'entity_answers': [[]],
                                     'yesno_answers': []})
            if 'answers' in sample:
                ref_answers.append({'question_id': sample['question_id'],
                                    'question_type': sample['question_type'],
                                    'answers': sample['answers'],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})

    if result_dir is not None and result_prefix is not None:
        result_file = os.path.join(result_dir, str(result_prefix) + '.json')
        with open(result_file, 'w') as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

        print('Saving {} results to {}'.format(result_prefix, result_file))

    # this average loss is invalid on test set, since we don't have true start_id and end_id
    # ave_loss = 1.0 * total_loss / num_of_batch
    # compute the bleu and rouge scores if reference answers is provided
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                ref_dict[question_id] = normalize(ref['answers'])
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
    else:
        bleu_rouge = None
    print('correct selected passage num is {} in {}'.format(select_true_num, select_total_num))
    print('correct passage num is {} in {}'.format(correct_p_num, total_num))
    return bleu_rouge


def predict(args):
    """
    predicts answers for test files
    """
    # logger = logging.getLogger("brc")
    # logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    # logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    # logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    # logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    # logger.info('Running with args : {}'.format(args))
    print('Running with args : {}'.format(args))

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)


if __name__ == '__main__':
    run()
