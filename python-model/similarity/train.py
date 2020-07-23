from prepareData import *
from model import *
from predict import *



#小批量计算损失
def batch_loss(rnn_q,rnn_api,query,api,tag,loss,hidden_size, temperature = 0.3):
#   初始化GRU的隐藏层状态
    w = torch.empty(3,query.shape[0],hidden_size).to(device)
    init_state=torch.nn.init.orthogonal_(w, gain=1)
    q_state=init_state
    api_state=init_state
    q_output=rnn_q(query,q_state)
    api_output=rnn_api(api,api_state)
    #此时输出的维度猜测应该是（batch_size,hidden_size）
    #下方为AI研习社语义任务之后修改的向量维度
    #
    # print(q_output.shape)
    # print(api_output.shape)
    q_output=q_output/torch.norm(q_output,dim=1).unsqueeze(1)
    api_output=api_output/torch.norm(api_output,dim=1).unsqueeze(1)

    pre_score=torch.mm(q_output,api_output.permute(1,0))
    # cls_score = torch.exp((q_output * api_output).sum(axis=1) / temperature)
    # exp_mat = torch.exp(pre_score / temperature)
    # prob = cls_score / exp_mat.sum(axis=1)
    #此处进行sigmoid方程与结果的概率进行（0-1）匹配。可能存在一部分问题，待确定
    score=torch.sigmoid(torch.flatten(pre_score / temperature))
    # score=torch.flatten((pre_score))#与下方损失函数相对应
    # score=(score+1)/2
  #！！！！！这个l的初始化可能有问题！！！！
    #tag应该是一个1维向量（由对角矩阵拉直）
    l=loss(score,tag).mean()
    # l = loss(prob, tag)
    return l

#训练过程
def train(rnn_q,rnn_api,dataset,lr,batch_size,num_epochs,hidden_size,device):
    query_optimizer=torch.optim.SGD(rnn_q.parameters(),lr=lr,momentum=0.9)
    api_optimizier=torch.optim.SGD(rnn_api.parameters(),lr=lr,momentum=0.9)
    # query_optimizer=torch.optim.Adam(rnn_q.parameters(),lr=lr,weight_decay=10.0)
    # api_optimizier=torch.optim.Adam(rnn_api.parameters(),lr=lr,weight_decay=10.0)

    #将损失函数由交叉熵损失改为MSE损失
    # loss=nn.MSELoss()

    # 损失尝试换为二分类交叉熵损失
    loss=nn.BCELoss()

    # 也是二分类，但将sigmoid函数包含住，比单纯二分类交叉熵损失更稳定（此改动与batch_loss中最后使用sigmoid联动）（尝试一下）
    # loss=nn.BCEWithLogitsLoss()
    print('开始训练~~~~~~~~~~~~~')
    for epoch in range(num_epochs):
        data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)
        ##print("epoch为：{}, iter: {}".format(epoch, len(data_iter)))
        l_sum=0.0
        for iter, (query,api) in enumerate(data_iter):
            query=query.to(device)
            api=api.to(device)
            query_optimizer.zero_grad()
            api_optimizier.zero_grad()
            # res_tag=torch.flatten(torch.ones(query.shape[0]).to(device))
            res_tag=torch.flatten(torch.eye(query.shape[0], query.shape[0]).to(device))
            l=batch_loss(rnn_q,rnn_api,query,api,res_tag,loss,hidden_size)
            l.backward()
            query_optimizer.step()
            api_optimizier.step()
            l_sum+=l.item()
            if (iter + 1) % 10 == 0:
                print("iter[{}/{}], loss {:.5f}".format((iter+1)*batch_size, len(dataset), l_sum / (iter+1)))
        print("epoch[{}/{}], loss {:.4f}".format(epoch, num_epochs, l_sum/len(data_iter)))


if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api_dict = {}
    #设置相关超参数
    hidden_size,out_size = 512, 128
    lr, batch_size, num_epochs =1, 2, 1

    #注意此时 res_tag的形状即为（batch_size,batch_size）


    #假设此时的batch_size为2，应该是一个对角矩阵（对角线均为1.0）
    query_max_length=15
    #需要提前计算api数据集中api序列的最大长度
    api_max_length=50
    path="./test.txt"
    in_vocab,out_vocab,dataset=read_data(path,query_max_length,api_max_length,api_dict)

    print(in_vocab)
    rnn_q = GRU(len(in_vocab),hidden_size,out_size).to(device)
    rnn_api = GRU(len(in_vocab),hidden_size,out_size).to(device)
    train(rnn_q, rnn_api, dataset, lr, batch_size, num_epochs,hidden_size,device)

    print('训练阶段完毕。。。。。。。')
    print("开始预测部分")
    data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)
    # rnn_q=rnn_q.cpu()
    query='See the general contract of the skip method of InputStream'
    api_output,apis=collection(rnn_api,data_iter,batch_size, hidden_size,dataset,device)
    # api_output=api_output.cpu()
    caculate(in_vocab,rnn_q,api_output,apis,query.lower(),query_max_length,api_dict,hidden_size,device)

    # 对collection函数进行调用，得到api_output(特征向量)，api（api的字符串序列）
    print('收集结束。。。。。')
