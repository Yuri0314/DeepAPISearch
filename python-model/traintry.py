from prepareDatatry import *
from modeltry import *
from predicttry import *
import torch.nn as nn



#小批量计算损失
def batch_loss(model,query,api,method,tag,loss,hidden_size, temperature = 0.05):
#   初始化GRU的隐藏层状态
    
    
    q_output,api_output=model(query,api,method,hidden_size,device)
    #此时输出的维度猜测应该是（batch_size,hidden_size）
    #下方为AI研习社语义任务之后修改的向量维度

    q_output=q_output/torch.norm(q_output,dim=1).unsqueeze(1)
    api_output=api_output/torch.norm(api_output,dim=1).unsqueeze(1)
    # pre_score=torch.sum(q_output*api_output,dim=1)
    # 使用batch_size-1分类任务，使用softmax计算概率
    # pre_score=torch.mm(q_output,api_output.permute(1,0))
    # exp_score = torch.exp(pre_score / temperature)
    # res_tag=torch.eye(query.shape[0]).to(device)
    # cls_score = (exp_score * res_tag).sum(axis=0)
    # cls_prob = cls_score / exp_score.sum(axis=0)
    # tag = torch.ones(query.shape[0]).to(device)

    # 使用二分类任务
    pre_score=torch.mm(q_output,api_output.permute(1,0))
    # cls_prob=torch.sigmoid(pre_score/temperature)
    m = nn.Softmax(dim=1)
    cls_prob=m(pre_score)

    # cls_score = torch.exp((q_output * api_output).sum(axis=1) / temperature)
    # exp_mat = torch.exp(pre_score / temperature)
    # prob = cls_score / exp_mat.sum(axis=1)
    #此处进行sigmoid方程与结果的概率进行（0-1）匹配。可能存在一部分问题，待确定
    # score=torch.sigmoid(pre_score / temperature)
    # score=torch.flatten((pre_score))#与下方损失函数相对应
    # score=(score+1)/2
    #！！！！！这个l的初始化可能有问题！！！！
    #tag应该是一个1维向量（由对角矩阵拉直）
    l=loss(cls_prob,tag).mean()
    # l = loss(prob, tag)
    return l, torch.diag(cls_prob,0)

#训练过程
def train(model,dataset,lr,batch_size,num_epochs,hidden_size,device):

    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    step_decay = 0.1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=8,gamma = step_decay)
    # query_optimizer=torch.optim.Adam(rnn_q.parameters(),lr=lr,weight_decay=10.0)
    # api_optimizier=torch.optim.Adam(rnn_api.parameters(),lr=lr,weight_decay=10.0)

    #将损失函数由交叉熵损失改为MSE损失
    # loss=nn.MSELoss()

    # 损失尝试换为二分类交叉熵损失
    loss=nn.BCELoss()
    # 交叉熵损失
    # loss=nn.CrossEntropyLoss()

    # 也是二分类，但将sigmoid函数包含住，比单纯二分类交叉熵损失更稳定（此改动与batch_loss中最后使用sigmoid联动）（尝试一下）
    # loss=nn.BCEWithLogitsLoss()
    print('开始训练~~~~~~~~~~~~~')
    model.train()
    for epoch in range(num_epochs):
        data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)

        ##print("epoch为：{}, iter: {}".format(epoch, len(data_iter)))
        l_sum=0.0
        for iter, (query,api,method) in enumerate(data_iter):
            # data_random=Data.DataLoader(dataset,query.shape[0],shuffle=True)
            # for  iter2,(query,_) in enumerate(data_random):
            #     if iter2 > 0:
            #         break
            #     query_random=query
            # api=torch.cat((api,api))
            # query=torch.cat((query,query_random))
            query=query.to(device)
            api=api.to(device)
            method=method.to(device)
            # api=api.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            # res_tag=torch.flatten(torch.ones(query.shape[0]).to(device))
            # res_tag=torch.cat((torch.ones(int(api.shape[0]/2)),torch.zeros(int(api.shape[0]/2)))).to(device)

            # batch_size-1分类使用tag
            # res_tag=torch.eye(query.shape[0]).to(device)
            #二分类使用tag
            res_tag=torch.eye(query.shape[0]).to(device)

            l, cls_prob=batch_loss(model,query,api,method,res_tag,loss,hidden_size)
            l.backward()
            optimizer.step()
            l_sum+=l.item()
            if (iter + 1) % 300 == 0:
                # print("iter[{}/{}], loss {:.5f}, prob {}".format((iter+1)*batch_size, len(dataset), l_sum / (iter+1), cls_prob))
                print("iter[{}/{}], loss {:.5f}, prob {}".format((iter+1)*batch_size, len(dataset), l_sum / (iter+1), cls_prob))
        if (epoch+1)%20==0:
            torch.save(model, '%s.pth'%(epoch+1))
        print("epoch[{}/{}], loss {:.4f},lr{}".format(epoch, num_epochs, l_sum/len(data_iter),optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler.step()

if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api_dict = {}
    #设置相关超参数
    hidden_size,out_size = 256, 256
    lr, batch_size, num_epochs =0.1, 8, 100
    method_max_length=10

    #注意此时 res_tag的形状即为（batch_size,batch_size）


    #假设此时的batch_size为2，应该是一个对角矩阵（对角线均为1.0）
    query_max_length=20
    #需要提前计算api数据集中api序列的最大长度
    api_max_length=8
    path="./all.txt"
    in_vocab,dataset=read_data(path,query_max_length,api_max_length,method_max_length,api_dict)

    print(in_vocab)
    rnn_q = GRU_QUERY(batch_size,len(in_vocab),hidden_size,out_size).to(device)
    rnn_api = GRU_API(batch_size,len(in_vocab),hidden_size,out_size).to(device)
    model=Similarity(rnn_api,rnn_q)
    train(model, dataset, lr, batch_size, num_epochs,hidden_size,device)

    

    print('训练阶段完毕。。。。。。。')
    print("开始预测部分")
    data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)
    # # rnn_q=rnn_q.cpu()
    # query='See the general contract of the skip method of InputStream'
    # api_output,apis=collection(rnn_api,data_iter,batch_size, hidden_size,dataset,device)
    # # api_output=api_output.cpu()
    # caculate(in_vocab,rnn_q,api_output,apis,query.lower(),query_max_length,api_dict,hidden_size,device)
    api_output,apis=collection(model.rnn_api,data_iter,batch_size, hidden_size,dataset,device)
    np.save('./api_dict.npy',api_dict)
    np.save('./api_output.npy',api_output)
    np.save('./apis.npy',apis)
    # 对collection函数进行调用，得到api_output(特征向量)，api（api的字符串序列）
    print('收集结束。。。。。')
