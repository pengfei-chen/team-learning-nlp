import logging
import random

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# set cuda
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

# split data to 10 fold
fold_num = 10
data_file = './data/sample_train.csv'
import pandas as pd

# 切分数据到10折
def all_data2fold(fold_num, num=10000):
    fold_data = []
    f = pd.read_csv(data_file,encoding='UTF-8')
    texts = f['text'].tolist()[:num]
    labels = f['label'].tolist()[:num]
    total = len(labels)
    index = list(range(total))
    np.random.shuffle(index)
    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])
    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)
    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)
    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]
        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels
        assert batch_size == len(fold_labels)
        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)
        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])
        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)
    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))
    return fold_data

fold_data = all_data2fold(10)

# print(fold_data)

# build train, dev, test data

fold_id = 9
# dev
dev_data = fold_data[fold_id]

# train
train_texts = []
train_labels = []
for i in range(0, fold_id):
    data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])

train_data = {'label': train_labels, 'text': train_texts}

# test
test_data_file = './data/sample_test.csv'
f = pd.read_csv(test_data_file,encoding='UTF-8')
texts = f['text'].tolist()
test_data = {'label': [0] * len(texts), 'text': texts}


# build vocab
from collections import Counter
from transformers import BasicTokenizer
basic_tokenizer = BasicTokenizer()

class Vocab():
    def __init__(self, train_data):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']
        self._id2label = []
        self.target_names = []
        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))
        
    def build_vocab(self, data):
        self.word_counter = Counter()
        # 统计每个 词 出现的次数
        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1

        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        self.label_counter = Counter(data['label'])

        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])
    
    def load_pretrained_embs(self, embfile):
        # 打开word2vec向量
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            # 5976, 100
            word_count, embedding_dim = int(items[0]), int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            # embeddings[self.unk] 这个向量，是多个向量的累加，后来又求了平均值；当其他词向量不存在时，也是用这个替代
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1
        # 这里有些没看懂
        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)  #为何要除以标准差
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)
        assert len(set(self._id2extword)) == len(self._id2extword)
        return embeddings
    
    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)
    
vocab = Vocab(train_data)


# build module
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        # 详情参见：https://www.jianshu.com/p/d8b77cc02410
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        # tensor.copy_(src)
        # 将src中的元素复制到tensor中并返回这个tensor; 两个tensor应该有相同shape
        self.bias.data.copy_(torch.from_numpy(b))
        # torch.from_numpy()方法把数组转换成张量，且二者共享内存，
        # 对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。

        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)
        
    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden

        # compute attention
        outputs = torch.matmul(key, self.query)  # b x len
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))
        # masked_fill 将 batch_masks 中原值为0（现在为1)的位置用float(-1e32)来填充
        attn_scores = F.softmax(masked_outputs, dim=1)  # b x len

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        # sum weighted sources
        # torch.bmm()强制规定维度和大小相同
        # 相乘的两个矩阵，要满足一定的维度要求：input（p,m,n) * mat2(p,n,a) ->output(p,m,a)。这个要求，
        # 可以类比于矩阵相乘。前一个矩阵的列等于后面矩阵的行才可以相乘。

        # key:b x len x hidden

        #torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，
        # 默认是将a中所有为1的维度删掉。也可以通过dim指定位置，删掉指定位置的维数为1的维度。
        #torch.unsqueeze()这个函数主要是对数据维度进行扩充。
        # 需要通过dim指定位置，给指定位置加上维数为1的维度。
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores
    
# build word encoder
word2vec_path = '../Mycodes/data/word2vec.txt'
dropout = 0.15

class WordCNNEncoder(nn.Module):
    def __init__(self, vocab):
        super(WordCNNEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = 100

        self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)

        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size, word_dims = extword_embed.shape
        logging.info("Load extword embed: words %d, dims %d." % (extword_size, word_dims))
        # 通过word2vec初始化了 Embedding的权重嘛？
        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.filter_sizes = [2, 3, 4]  # n-gram window
        self.out_channel = 100
        #这里面是有3个卷积嘛？ 这里和下面的用处，我有点没看懂
        self.convs = nn.ModuleList([nn.Conv2d(1, self.out_channel, (filter_size, input_size), bias=True)
                                    for filter_size in self.filter_sizes])
        
    def forward(self, word_ids, extword_ids):
        # word_ids: sen_num x sent_len  #多少个句子 * 每个句子的长度？
        # extword_ids: sen_num x sent_len
        # batch_masks: sen_num x sent_len
        sen_num, sent_len = word_ids.shape
        word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed
        # 这两部分汇总后的Embedding作为训练输入
        if self.training:
            batch_embed = self.dropout(batch_embed)
        batch_embed.unsqueeze_(1)  # sen_num x 1 x sent_len x 100 #在1这里，增加了一个维度
        pooled_outputs = []
        for i in range(len(self.filter_sizes)):
            filter_height = sent_len - self.filter_sizes[i] + 1  #卷积后的高度和宽度， 默认核为正方形
            conv = self.convs[i](batch_embed)
            hidden = F.relu(conv)  # sen_num x out_channel x filter_height x 1
            mp = nn.MaxPool2d((filter_height, 1))  # (filter_height, filter_width) # kernel 长和宽= filter_height 、1
            pooled = mp(hidden).reshape(sen_num,
                                        self.out_channel)  # sen_num x out_channel x 1 x 1 -> sen_num x out_channel
            pooled_outputs.append(pooled)
        reps = torch.cat(pooled_outputs, dim=1)  # sen_num x total_out_channel
        if self.training:
            reps = self.dropout(reps)
        return reps
        

# build sent encoder
sent_hidden_size = 256
sent_num_layers = 2

# 这个类用来干嘛的，我还是不清楚
class SentEncoder(nn.Module):
    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 这里为啥要用到LSTM？
        self.sent_lstm = nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)
        # 返回隐藏层状态，用来干啥？
        return sent_hiddens

# build model
class Model(nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.sent_rep_size = 300
        self.doc_rep_size = sent_hidden_size * 2  # sent_hidden_size =256
        self.all_parameters = {}
        parameters = []
        self.word_encoder = WordCNNEncoder(vocab)  #CNNtext编码后的向量
        # 满足参数梯度 = True 的，就选出来放到 parameters 里面去。
        # parameters ： nn.Module自带的属性
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))

        self.sent_encoder = SentEncoder(self.sent_rep_size)  #返回隐藏层状态
        self.sent_attention = Attention(self.doc_rep_size) # 返回 batch_outputs, attn_scores
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))

        self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))
        if use_cuda:
            self.to(device)
        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters
        logging.info('Build model with cnn word encoder, lstm sent encoder.')
        # numpy.prod()这个函数是连乘操作，将里面所有的元素相乘
        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        logging.info('Model param num: %.2f M.' % (para_num / 1e6))  #统计一共有多少个需要训练、调整的参数

    def forward(self, batch_inputs):
        # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
        # batch_masks : b x doc_len x sent_len
        """
        所以 b 一直是 batch_size 的意思吗？
        """
        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)  # sen_num x sent_len

        sent_reps = self.word_encoder(batch_inputs1, batch_inputs2)  # sen_num x sent_rep_size

        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        return batch_outputs


model = Model(vocab)

