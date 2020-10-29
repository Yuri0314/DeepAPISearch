import torch
from torch import nn
import torch.utils.data as Data
import numpy as np
import json
import pickle
#根据build_data函数更改，转化字符串为index序列
def build_query(vocab,query_seq):
    query_idx=[vocab.stoi[w] for w in query_seq]
    return torch.tensor(query_idx)

def caculate(vocab,rnn_q,api_output,apis,query,k,query_max_length,api_dict,hidden_size,device,temperature=0.1):
    with torch.no_grad():
        query_seq=query.split(' ')
        if len(query_seq)>query_max_length-1:
                query_seq=query_seq[:query_max_length-1]
        query_data=build_query(vocab,query_seq).unsqueeze(0)
        query_data=query_data.to(device)
        # 初始化网络的隐藏层
        w = torch.empty(4,1,hidden_size).to(device)
        init_state=torch.nn.init.orthogonal_(w, gain=1)
        query_state=init_state
        q_output=rnn_q(query_data,query_state).unsqueeze(0)
        print(q_output.shape)
        #q_output形状应为（1，output_size）
        q_output=q_output/torch.norm(q_output,dim=1).unsqueeze(1)

        #此处传进来的api_output形状应为(所有api个数，output_size)
        api_output=torch.from_numpy(api_output)
        api_output=api_output.to(device)
        pre_score=torch.mm(q_output,api_output.permute(1,0))
        #此处进行sigmoid方程与结果的概率进行（0-1）匹配。
        score=torch.sigmoid(pre_score/temperature)
        #之后针对分数最高的那个API进行输出序列
        _,index=torch.topk(score,k,dim=1)
        index=index.cpu().numpy().tolist()
        index=index[0]
        for idx in index:
            print(api_dict[tuple(apis[idx].tolist())])


if __name__ == "__main__":
    with torch.no_grad():
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        api_dict=np.load('./api_dict.npy',encoding='bytes',allow_pickle=True)
        api_dict=api_dict.item()
        api_output=np.load('./api_output.npy',encoding='bytes',allow_pickle=True)
        apis=np.load('./apis.npy',encoding='bytes',allow_pickle=True)

        method_max_length=20
        #假设此时的batch_size为2，应该是一个对角矩阵（对角线均为1.0）
        query_max_length=20
        #需要提前计算api数据集中api序列的最大长度
        api_max_length=10
        fr = open("./vocab.pkl", "rb")
        in_vocab = pickle.load(fr)
        print(len(in_vocab))
        model=torch.load('model.pth')
        rnn_q=model.rnn_query
        hidden_size = 1024
        k=20
        query=""
        print("输入quit()退出")
        while True:
            str = input("请输入：")
            if str == 'quit()':
                break
            else:
                query = str
                caculate(in_vocab,rnn_q,api_output,apis,query.lower(),k,query_max_length,api_dict,hidden_size,device)