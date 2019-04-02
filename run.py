import argparse
import copy, json, os
from dataset import BRCDataset
from vocab import Vocab
import pickle
import time, math, torch
import tqdm
from model.bidaf import BiDAF
from torch import nn, optim


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
    parser.add_argument('--gpu', type=int, default='0',
                        help='specify gpu device')
    parser.add_argument('--cuda', type=int, default='1',
                        help='1 use cuda')
    parser.add_argument('--dropout', type=float, default='0.1')
    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
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

        # print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
        #       'elapse: {elapse:3.3f} min'.format(ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
        #                                          elapse=(time.time() - start) / 60))
        print('Average train loss for epoch {} is {}'.format(epoch_i, train_loss))
        # start = time.time()
        # valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        # print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
        #       'elapse: {elapse:3.3f} min'.format(
        #     ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
        #     elapse=(time.time() - start) / 60))
        #
        # valid_accus += [valid_accu]
        #
        # model_state_dict = model.state_dict()
        # checkpoint = {
        #     'model': model_state_dict,
        #     'settings': args,
        #     'epoch': epoch_i}
        #
        # if args.save_model:
        #     if args.save_mode == 'all':
        #         model_name = args.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
        #         torch.save(checkpoint, model_name)
        #     elif args.save_mode == 'best':
        #         model_name = args.save_model + '.chkpt'
        #         if valid_accu >= max(valid_accus):
        #             torch.save(checkpoint, model_name)
        #             print('    - [Info] The checkpoint file has been updated.')

    # logger.info('Done with model training!')
    print('Done with model training!')


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''

    model.train()

    model.to(device)
    total_loss, num_of_batch = 0, 0
    log_every_n_batch, n_batch_loss = 50, 0
    for bitx, batch in enumerate(training_data, 1):
        num_of_batch += 1
        # batch_size x padded_p_len
        # answers_doc_index = []
        # new_passsages_token = []
        # new_question_token = []
        # for ind in batch['raw_data']:
        #     answers_doc_index.append(len(answers_doc_index) * 5 + int(ind['answer_docs'][0]))
        # for i in answers_doc_index:
        #     new_passsages_token.append(batch['passage_token_ids'][i])
        #     new_question_token.append(batch['question_token_ids'][i])
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
        answer_prob = model(p, q)

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


# def evaluate(args):
#     """
#     evaluate the trained model on dev files
#     """
#     # logger = logging.getLogger("brc")
#     # logger.info('Load data_set and vocab...')
#     print('Load data_set and vocab...')
#     with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
#         vocab = pickle.load(fin)
#     assert len(args.dev_files) > 0, 'No dev files are provided.'
#     brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)
#     # logger.info('Converting text into ids...')
#     print('Converting text into ids...')
#     brc_data.convert_to_ids(vocab)
#     # logger.info('Restoring the model...')
#     rc_model = RCModel(vocab, args)
#     rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
#     # logger.info('Evaluating the model on dev set...')
#     dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
#                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
#     dev_loss, dev_bleu_rouge = rc_model.evaluate(
#         dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
#     # logger.info('Loss on dev set: {}'.format(dev_loss))
#     # logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
#     # logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


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
