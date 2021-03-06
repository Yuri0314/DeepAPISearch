
import collections
import io
import re
import torch
import torchtext.vocab as Vocab
import torch.utils.data as Data
import pickle

PAD,EOS,UNK='<pad>','<eos>','<unk>'

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



#处理数据对
def processline(x,y,z):
    # 去除各种杂乱的标点符号
    x=re.sub('[.(),->]',' ',x)
    #使用strip去除y尾部的换行符
    y=re.sub(r'[()]',' ',y.strip())
    z=z.strip()
    #  把#部分与前面类名用空格分开,其中line为字符串
    x=re.sub('[#]',' #',x)
    #使用line1存储处理过后的x字符串，line2存储处理过后的y字符串
    line1=''
    line2=''
    line3=''
    for iter,(i) in enumerate(x.split(' ')):
        if i=='':
            continue
        if iter==0:
            line1+=re.sub("[A-Z]",lambda x:" "+x.group(0),i)+' '
            continue
        if not re.match('#.*',i):

            line1+=i+' '
        else:
            #下面语句是将方法名按照大写字母分割开,对于y为空的语句将#后面切割之后的方法名替代(源数据集中已经对无描述部分进行了处理，此处代码注释)

            line1+=(re.sub("[A-Z]",lambda x:" "+x.group(0),i[1:]))+' '
#         将line1中多个空格在一块的改为一个空格
        line1=re.sub(r'\s+',' ',line1)
    for j in y.split(' '):
        if j=='':
            continue
        else:
            line2+=j+' '
    # 返回处理之后的x与y与z字符串
    for l in z.split('->'):
        line3+=l+' '
    return line1.lower().strip(),line2.lower().strip(),line3.lower().strip()

# 将一个序列中所有的词记录在all_tokens中以便之后构造词典，然后在该序列后面添加PAD直到序列
# 长度变为max_seq_len，然后将序列保存在all_seqs中
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造Tensor
def build_data(all_tokens, all_seqs):
    
    vocab = Vocab.Vocab(collections.Counter(all_tokens),
                        specials=[PAD, EOS,UNK])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)
def build_method_data(vocab, all_seqs):

    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return torch.tensor(indices)


# 开始读取数据，构建词典，以及数据集
def read_data(path,query_max_length,api_max_length,method_max_length,api_dict):
#     使用raw_apis列表装所有的原生api
    in_tokens,out_tokens,in_seqs,out_seqs,method_seqs,raw_apis=[],[],[],[],[],[]
    with io.open(path) as f:
        lines=f.readlines()
    flag=0
    for line in lines:
        #将每一行语句按照'：'将api与对应的query查询分开
#         下面的out_seq代表api序列，in_seq代表查询语句，method_seq代表方法调用序列
        out_seq,in_seq,method_seq=line.split(':::')
        raw_apis.append(out_seq)
        #使用processline处理输入输出数据的各种标点，输出干净字符串
        out_seq,in_seq,method_seq=processline(out_seq,in_seq.strip(),method_seq)

        in_seq_tokens,out_seq_tokens,method_seq_tokens=in_seq.split(' '),out_seq.split(' '),method_seq.split(' ')

       #针对描述语句过长的进行截断，下面减一操作是因为要对句末添加EOS
        if len(in_seq_tokens)>query_max_length-1:
            in_seq_tokens=in_seq_tokens[:query_max_length-1]
        if len(out_seq_tokens)>api_max_length-1:
            out_seq_tokens=out_seq_tokens[:api_max_length-1]
        if len(method_seq_tokens)>method_max_length-1:
            method_seq_tokens=method_seq_tokens[:method_max_length-1]

        process_one_seq(in_seq_tokens,out_tokens,in_seqs,query_max_length)
        process_one_seq(out_seq_tokens,out_tokens,out_seqs,api_max_length)
        process_one_seq(method_seq_tokens,out_tokens,method_seqs,method_max_length)

    # 把三个输入都放进一个词表中
    in_vocab,in_data=build_data(out_tokens,in_seqs)
    out_data=build_method_data(in_vocab,out_seqs)
    method_data=build_method_data(in_vocab,method_seqs)


    #构建一个词典
    for i in range(len(out_data)):
         api_dict[tuple(out_data[i].numpy().tolist())] = raw_apis[i]

    pickle.dump(in_vocab, open("./vocab.pkl", "wb"))
    print(len(in_vocab))
    return in_vocab,Data.TensorDataset(in_data,out_data,method_data)

def read_data_val(path,query_max_length,api_max_length,method_max_length,in_vocab):
#     使用raw_apis列表装所有的原生api
    out_tokens,in_seqs,out_seqs,method_seqs=[],[],[],[]
    with io.open(path) as f:
        lines=f.readlines()
    for line in lines:
        #将每一行语句按照'：'将api与对应的query查询分开
#         下面的out_seq代表api序列，in_seq代表查询语句，method_seq代表方法调用序列
        out_seq,in_seq,method_seq=line.split(':::')
        
        #使用processline处理输入输出数据的各种标点，输出干净字符串
        out_seq,in_seq,method_seq=processline(out_seq,in_seq.strip(),method_seq)

        in_seq_tokens,out_seq_tokens,method_seq_tokens=in_seq.split(' '),out_seq.split(' '),method_seq.split(' ')

       #针对描述语句过长的进行截断，下面减一操作是因为要对句末添加EOS
        if len(in_seq_tokens)>query_max_length-1:
            in_seq_tokens=in_seq_tokens[:query_max_length-1]
        if len(out_seq_tokens)>api_max_length-1:
            out_seq_tokens=out_seq_tokens[:api_max_length-1]
        if len(method_seq_tokens)>method_max_length-1:
            method_seq_tokens=method_seq_tokens[:method_max_length-1]

        process_one_seq(in_seq_tokens,out_tokens,in_seqs,query_max_length)
        process_one_seq(out_seq_tokens,out_tokens,out_seqs,api_max_length)
        process_one_seq(method_seq_tokens,out_tokens,method_seqs,method_max_length)

    # 把三个输入都放进一个词表中
    in_data=build_method_data(in_vocab,in_seqs)
    out_data=build_method_data(in_vocab,out_seqs)
    method_data=build_method_data(in_vocab,method_seqs)

    return Data.TensorDataset(in_data,out_data,method_data)











