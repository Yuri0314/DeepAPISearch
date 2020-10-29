from prepareData import *
from model import *
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

#小批量计算损失
def batch_loss(model,query,api,method,tag,loss,hidden_size,a_output,apis2,pos_n,temperature = 0.1):
#   初始化GRU的隐藏层状态
    
    
    q_output,api_output=model(query,api,method,hidden_size,device)
    #此时输出的维度猜测应该是（batch_size,hidden_size）
    #下方为AI研习社语义任务之后修改的向量维度
    
    # 对输出进行范式计算（可替换
    q_output=q_output/torch.norm(q_output,p=None,dim=1).unsqueeze(1)
    api_output=api_output/torch.norm(api_output,p=None,dim=1).unsqueeze(1)
# 下面两个变量表用于测试准确度
    apis2.append(api[:pos_n].cpu().detach().numpy())
    a_output.append(api_output[:pos_n].cpu().detach().numpy())
    
    # 使用word2vec的损失计算方式
    pre_score=torch.sum(q_output*api_output,dim=1)

    # 对正样本乘上1，对于负样本乘上（-1）
    cls_prob=torch.mul(pre_score,tag)

    # 下面的l即为训练损失
    cls_prob=F.logsigmoid(cls_prob/temperature)
    l=-1*torch.sum(cls_prob)
    l=l/float(query.shape[0])

    return l, cls_prob

# 对dataset中取出的数据拼接负样本，其中n为一个正样本对应的负样本数
def negCat(query,api,method,n):
    method1,query1,api1=method,query,api
    while n>0:
        method1=torch.cat((method1,method),0)
        api1=torch.cat((api1,api),0)
        query=torch.cat((query[1:],query[0].unsqueeze(0)),0)
        query1=torch.cat((query1,query),0)
        n-=1
    return query1,api1,method1
#训练过程
def train(model,dataset,lr,batch_size,num_epochs,hidden_size,device,neg,file_path='./1.txt'):
    file=open(file_path,'w')
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    step_decay = 0.8
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
 

    # 损失尝试换为二分类交叉熵损失
    weight= torch.ones(batch_size,batch_size)+torch.eye(batch_size,batch_size)*2
    weight=weight.to(device)
    loss=nn.BCELoss(weight=weight)

    print('开始训练~~~~~~~~~~~~~')
    model.train()
    best_loss=float('inf')
    for epoch in range(num_epochs):
        data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)
        # 为测试做准备
        a_output = []
        apis2=[]
        ##print("epoch为：{}, iter: {}".format(epoch, len(data_iter)))
        l_sum=0.0
        for iter, (query,api,method) in enumerate(data_iter):
            query,api,method=negCat(query,api,method,neg)
            query=query.to(device)
            api=api.to(device)
            method=method.to(device)
            optimizer.zero_grad()
            
            # batch中正负样本行数为pos_n,neg_n
            pos_n=int(api.shape[0]/(neg+1))
            neg_n=int(api.shape[0]/(neg+1)*neg)

            a=np.hstack((np.ones(pos_n),np.ones(neg_n)*(-1)))
            res_tag=torch.from_numpy(a).float().to(device)

            l, cls_prob=batch_loss(model,query,api,method,res_tag,loss,hidden_size,a_output,apis2,pos_n)
            l.backward()
            optimizer.step()
            l_sum+=l.item()
            
            if (iter + 1) % 400 == 0:
                print("epoch{},iter[{}/{}], loss {:.5f}, sum_loss {:.5f}".format(epoch,(iter+1)*batch_size, len(dataset), l,l_sum / (iter+1)))

                file.writelines("epoch{},iter[{}/{}], loss {:.5f}\n".format(epoch,(iter+1)*batch_size, len(dataset), l_sum / (iter+1)))       
        writer.add_scalar('data/loss',l_sum,epoch)
        apis2=np.concatenate(apis2,axis=0)
        a_output=np.concatenate(a_output, axis=0)
        print("epoch[{}/{}], loss {:.4f},lr{}".format(epoch, num_epochs, l_sum/len(data_iter),optimizer.state_dict()['param_groups'][0]['lr']))
        file.writelines("epoch[{}/{}], loss {:.4f},lr{}\n".format(epoch, num_epochs, l_sum/len(data_iter),optimizer.state_dict()['param_groups'][0]['lr']))
        if l_sum<best_loss:
            best_loss=l_sum
            torch.save(model,'model.pth')
            print('best_loss已更新为：{}'.format(best_loss))     
        # 10个epoch测试一次准确度
        if (epoch+1)%10==0:
            with torch.no_grad():
                # 训练集测试部分
                print("开始在第{}轮测试".format(epoch))
                test_model=torch.load('model.pth')
                test_rnn_q=test_model.rnn_query
                val_rnn_api=test_model.rnn_api
                test_batch=128

                # 训练集部分测试
                print("注意！！！！开始训练集部分测试")
                data_iter1=Data.DataLoader(dataset,test_batch)
                correct1=0
                correct2=0
                correct3=0

                length=len(dataset)
                for iter, (query,api,method) in enumerate(data_iter1):
                    query=query.to(device)
                    topk1=test(test_rnn_q,a_output,apis2,query,20,hidden_size,device)
                    topk2=test(test_rnn_q,a_output,apis2,query,10,hidden_size,device)
                    topk3=test(test_rnn_q,a_output,apis2,query,5,hidden_size,device)
                    circle=api.size()[0]
                    api=api.tolist()
                    for i in range(circle):

                        if  api[i] in topk1[i]:
                            correct1+=1
                        if  api[i] in topk2[i]:
                            correct2+=1
                        if  api[i] in topk3[i]:
                            correct3+=1
                    if (iter + 1) % 100 == 0:
                        print("top20准确度为[{}/{}]".format(correct1,(iter + 1)*query.shape[0]))
                        print("top10准确度为[{}/{}]".format(correct2,(iter + 1)*query.shape[0]))
                        print("top5准确度为[{}/{}]".format(correct3,(iter + 1)*query.shape[0]))
                print('top20正确个数为{}'.format(correct1))
                print('top10正确个数为{}'.format(correct2))
                print('top5正确个数为{}'.format(correct3))
 
                file.writelines('正确个数为top20: {} top10: {} top5: {}\n'.format(correct1,correct2,correct3))
                print('api个数为{}'.format(length))
                file.writelines('api个数为{}\n'.format(length))
                rate1=correct1/length
                rate2=correct2/length
                rate3=correct3/length
                print('查询准确率为top20: {} top10: {} top5: {}%'.format(rate1*100,rate2*100,rate3*100))
                file.writelines('查询准确率为top20:{} top10: {} top5: {}%\n'.format(rate1*100,rate2*100,rate3*100))
                writer.add_scalar('correct_train/rate20',rate1,epoch)
                writer.add_scalar('correct_train/rate10',rate2,epoch)
                writer.add_scalar('correct_train/rate5',rate3,epoch)
        scheduler.step(l)
    writer.close()
    file.close()
 
# 测试准确度函数
def test(rnn_q,api_output,apis2,query,k,hidden_size,device,temperature=0.1):
    with torch.no_grad():
        # 初始化网络的隐藏层
        w = torch.empty(4,query.shape[0],hidden_size).to(device)
        init_state=torch.nn.init.orthogonal_(w, gain=1)
        query_state=init_state
        q_output=rnn_q(query,query_state)
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
        l=[]
        for ind in index:
            a=[]
            for idx in ind:
                a.append(apis2[idx].tolist())
            l.append(a)
        return l

def collection(rnn_api,data_iter,batch_size, hidden_size,dataset,device):
    with torch.no_grad():
        # api_output=torch.tensor([0.0]).expand(batch_size,out_size).to(device)
        #使用apis收集对应的api_idx序列
        api_output = []
        apis=[]
        for iter, (query,api,method) in enumerate(data_iter):
            query=query.to(device)
            api=api.to(device)
            method=method.to(device)
        #初始化rnn_api的初始隐层状态
    #     !!!!注意 apis中添加的应该是一个原生的字符串序列
            apis.append(api.cpu().detach().numpy())

            # 初始化网络的隐藏层
            w = torch.empty(4,query.shape[0],hidden_size).to(device)
            init_state=torch.nn.init.orthogonal_(w, gain=1)
            api_state=init_state
            output=rnn_api(api,method,api_state)
            api_output.append(output.cpu().detach().numpy())
            if (iter + 1) % 1000 == 0:
                print("iter[{}/{}]".format((iter+1)*batch_size, len(dataset)))
            # api_output=torch.cat((api_output,output))
        #对生成的向量进行裁剪，去除最开始初始化为零的api_output部分
        #   注 意！！！
    #   需要把输出的api_output初始expand函数产生的全为零的部分清除,之后才可以保存到csv文件中(已裁剪)
        api_output=np.concatenate(api_output, axis=0)
        apis=np.concatenate(apis,axis=0)
        # print(api_output.shape)
        # print(np.sum(api_output * api_output, axis=1).shape)

# 下方的代码是对api_output进行一个数值上的norm（归一化）
        api_output = api_output / np.sqrt(np.sum(api_output * api_output, axis=1)).reshape([-1, 1])

        # api_output=api_output/torch.norm(api_output,dim=1).unsqueeze(1)
        return api_output,apis
          

if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api_dict = {}
    #设置相关超参数
    hidden_size,out_size = 1024, 512
    lr, batch_size, num_epochs =0.02,64, 60
    neg=5
    method_max_length=20
    #注意此时 res_tag的形状即为（batch_size,batch_size）


    #假设此时的batch_size为2，应该是一个对角矩阵（对角线均为1.0）
    query_max_length=20
    #需要提前计算api数据集中api序列的最大长度
    api_max_length=10
    path="./2020.txt"
    in_vocab,dataset=read_data(path,query_max_length,api_max_length,method_max_length,api_dict)


    print(in_vocab)
    print(len(in_vocab))
    rnn_q = GRU_QUERY(batch_size,len(in_vocab),hidden_size,out_size).to(device)
    rnn_api = GRU_API(batch_size,len(in_vocab),hidden_size,out_size).to(device)
    model=Similarity(rnn_api,rnn_q)
    writer=SummaryWriter('tensorboard')
    train(model, dataset, lr, batch_size, num_epochs,hidden_size,device,neg)

    
    print('训练阶段完毕。。。。。。。')
    print("开始预测部分")
    data_iter=Data.DataLoader(dataset,batch_size,shuffle=True)

    api_output,apis=collection(model.rnn_api,data_iter,batch_size, hidden_size,dataset,device)
    np.save('./api_dict.npy',api_dict)
    np.save('./api_output.npy',api_output)
    np.save('./apis.npy',apis)
    # 对collection函数进行调用，得到api_output(特征向量)，api（api的字符串序列）
    print('收集结束。。。。。')
