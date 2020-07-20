import random
import torch.nn.functional as F
import torch
import torchtext
from torchtext import data
import torch.nn as nn
import os
import re
import spacy
import torch.optim as optim
import time
import math

spacy_en = spacy.load('en_core_web_sm')  # 英文文本处理库

# 分词


def tokenize_query(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_api(text):
    return text.split()


SRC = torchtext.data.Field(tokenize=tokenize_query,
                           init_token='<sos>', eos_token='<eos>', lower=True)
TRG = torchtext.data.Field(tokenize=tokenize_api,
                           init_token='<sos>', eos_token='<eos>', lower=False)


dataset = torchtext.data.TabularDataset('merge.csv', format='csv', fields=[
                                        (',', None), ('src', SRC), ('trg', TRG)])
SRC.build_vocab(dataset, min_freq=1)
TRG.build_vocab(dataset, min_freq=1)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src=[src,batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded=[src len,batch size,emb dim]
        outputs, hidden = self.rnn(embedded)
        # outputs=[src len,batch size,hid dim*n directions]
        # hidden=[n layers*m directions,batch size,hid dim]
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim*2)+dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(
            1, src_len, 1)  # 增加一维，并对相应维度*！，*src_len，*1的操作
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # permute换维

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear(
            (enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)
        # input=[1,batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio 是使用teacher forcing的概率
        # 如果teacher_forcing_ratio是0.75 我们将有75%概率输入ground_truth

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        encoder_outputs, hidden = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):

            output, hidden = self.decoder(input, hidden, encoder_outputs)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            # 获得概率最高的预测
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
train_iterator = data.BucketIterator(
    dataset, batch_size=BATCH_SIZE, device=device)
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM,
                      ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM,
                      DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)

# 权重初始化


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

# 计算网络中共多少可训练的参数


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


optimizer = optim.Adam(model.parameters())
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]  # stoi 返回单词与其对应的下标
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# 训练函数


# def train(model, iterator, optimizer, criterion, clip):
#     model.train()

#     epoch_loss = 0

#     for i, batch in enumerate(iterator):
#         src = batch.src
#         trg = batch.trg

#         optimizer.zero_grad()
#         output = model(src, trg)
#         # trg = [trg len, batch size]
#         # output = [trg len, batch size, output dim]

#         output_dim = output.shape[-1]
#         output = output[1:].view(-1, output_dim)
#         trg = trg[1:].view(-1)
#         # trg = [(trg len - 1) * batch size]
#         # output = [(trg len - 1) * batch size, output dim]
#         loss = criterion(output, trg)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # 梯度裁剪，应对梯度爆炸
#         optimizer.step()
#         epoch_loss += loss.item()
#     return epoch_loss/len(iterator)

# # 评估函数


# def evaluate(model, iterator, criterion):
#     model.eval()
#     epoch_loss = 0
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#             src = batch.src
#             trg = batch.trg

#             output = model(src, trg, 0)
#             # trg = [trg len, batch size]
#             # output = [trg len, batch size, output dim]

#             output_dim = output.shape[-1]
#             output = output[1:].view(-1, output_dim)
#             trg = trg[1:].view(-1)
#             # trg = [(trg len - 1) * batch size]
#             # output = [(trg len - 1) * batch size, output dim]

#             loss = criterion(output, trg)

#             epoch_loss += loss.item()

#     return epoch_loss / len(iterator)

# # epoch花费时间


# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs

# N_EPOCHS = 30
# CLIP = 1

# best_train_loss = float('inf')#float('inf)正负无穷

# for epoch in range(N_EPOCHS):

#     start_time = time.time()

#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     # valid_loss = evaluate(model, train_iterator, criterion)

#     end_time = time.time()

#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)

#     if train_loss < best_train_loss:
#         best_train_loss = train_loss
#         torch.save(model.state_dict(), 'tut1-model.pt')


#     print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')

# torch.save(model,'seq2seq.pth')


model = torch.load('seq2seq.pth')
# 测试


def searchApi(sentence, src_field, trg_field, model, device, max_len=25):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en_core_web_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]


    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
    for i in range(max_len):

        

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        if pred_token in trg_indexes:
            continue
        else:
            trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]

def test():
    sear=""
    print("输入exit()退出")
    while True:
        str = input("请输入：")
        if str == 'exit()':
            break
        else:
            src = str
            search = searchApi(src, SRC, TRG, model, device)
            if search[-1]=="<eos>":
                search=search[:-1]
            for i in search:
                sear+=i
            print("你可能需要：", sear)


test()
