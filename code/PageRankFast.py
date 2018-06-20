'''
PageRank实现，采用字典存储有向图，改进了节点更新权重的计算方式。
大数据集耗时1h左右，小数据集耗时70s不到
'''

#记录时间
import time
import argparse

#数据集来自 http://socialcomputing.asu.edu/datasets/Twitter
#           https://snap.stanford.edu/data/egonets-Twitter.html
#大数据集包含1000万个顶点和8000万条边，小数据集包含8万个顶点，170万条边
bigDataFilename = './Twitter-dataset/data/edges.csv'
smallDataFilename = './web-Stanford.txt'


def initPR(nodes): 
    '''
    初始化PageRank值为 1/N
    @param_in nodes: dict, 有向图的节点PR值

    @param_out nodes: dict, 初始化后的节点PR值
    ''' 
    N = len(nodes)
    nodes = dict.fromkeys(nodes, 1.0 / N)
    return nodes


def updatePR(edgeIn, edgeOut, v, rate):
    '''
    PR值第一部分
    对v列向量上的每个结点计算其本轮的pagerank值.
    函数需要遍历所有的结点, 对每个结点遍历所有的入度, 因此会遍历所有的边
    @param_in edgeIn：dict，有向图的所有入边
    @param_in edgeOut：dict，有向图的所有出边
    @param_in v：dict，所有节点PR值
    @param_in rate：float，随机打开新网址的概率为 1-rate

    @param_out vNew：dict，迭代后的节点PR值
    '''
    vNew = {}
    for row in v.keys():
        t = 0.0
        if edgeIn.__contains__(row):  # 并非每个结点都有入度
            #对每个节点的入边起点循环
            for fromNode in edgeIn[row]:
                t += 1.0 / len(edgeOut[fromNode]) * v[fromNode]  # t += your share of V[key] * importance of V[key]
            vNew[row] = t * rate
        else:   #无入度节点
            vNew[row] = 0.0
    return vNew


def walkOut(rate, v):
    '''
    PR值更新第二部分
    模拟一定概率随机打开新网页
    @param_in rate：float，随机打开新网址的概率为 1-rate
    @param_in v：dict，节点PR值
    '''
    for key in v.keys():
        v[key] += rate


def calcLoss(nextV, v, epsilon):
    '''
    计算L1总损失，损失小于epsilon则迭代停止
    @param_in nextV：dict，更新后的节点PR值
    @param_in v：dict，节点PR值
    @param_in epsilon：float，允许精度

    '''
    loss = 0.0
    for key in v.keys():
        loss += abs(v[key] - nextV[key])
    if loss > epsilon:
        return False
    return True

def pageRank(rate, edgeIn, edgeOut, v, max_iter=1e3, epsilon=1e-10): 
    '''
    pageRank算法
    @param_in edgeIn：dict，有向图的所有入边
    @param_in edgeOut：dict，有向图的所有出边
    @param_in v：dict，所有节点PR值
    @param_in rate：float，随机打开新网址的概率为 1-rate
    @param_in max_iter：int，最大迭代次数
    @param_in epsilon：float，损失允许精度

    @param_out v：dict，多轮迭代直至loss小于给定精度后的节点PR值
    '''
    #初始化
    initNodeValue = v[list(v.keys())[0]]
    prob = (1 - rate) * initNodeValue
    count = 0 #迭代次数

    nextV = updatePR(edgeIn, edgeOut, v, rate)
    walkOut(prob, nextV)

    #迭代至收敛
    while not calcLoss(nextV, v, epsilon) and count < max_iter:
        v = nextV
        nextV = updatePR(edgeIn, edgeOut, v, rate)
        walkOut(prob, nextV)  # update nextV
        count += 1
        #print('round count: %d' % count)
    print("exit, total rounds count: ", count)
    return v


def buildFromFile(size = 'small'):
    '''
    从文件中获取节点和边的信息，构建入边和出边字典
    @param_in f：str，有向图文件名
    @param_in size：str，数据集类型，small为小数据集，否则为大数据集

    @param_out edgeOut：出边字典
    @param_out edgeIn：入边字典
    @param_out nodes：节点字典
    '''
    edgeOut = {}
    edgeIn = {}
    nodes = {}
    f = smallDataFilename if size == 'small' else bigDataFilename
    with open(f) as file: 
        count = 0 
        for line in file.readlines():
            if size == "small":#小数据集先去掉开头几行说明
                if count < 4: count+=1;continue
            start, end = (int(x)-1 for x in line.split()) if size == 'small' else (int(x)-1 for x in line.split(','))
            if not edgeOut.__contains__(start):
                edgeOut[start] = []
                nodes[start] = 0 #节点起始PR值
            if not edgeIn.__contains__(end):
                edgeIn[end] = []
                nodes[end] = 0 #节点起始PR值

            edgeOut[start].append(end)
            edgeIn[end].append(start)

            #print(head, tail)
    return edgeOut, edgeIn, nodes

def do(arg):
    '''
    main函数，执行所有操作
    @param_in arg：命令行参数
    '''
    global edgeIn, edgeOut
    print("PageRank begins!")
    #构建有向图
    startReadTime = time.time()
    edgeIn, edgeOut, nodes = buildFromFile(arg.size)
    print("Reading elapsed time: ", time.time() - startReadTime)
    #初始化
    nodes = initPR(nodes) 
    #pagerank迭代更新
    rate = 1 - arg.rate
    startTime = time.time()
    PageRankResult = pageRank(rate, edgeIn, edgeOut, nodes, arg.num_iter, arg.eps)
    print("PageRank iter elapsed time: ", time.time() - startTime)
    #将结果写入resultFilename中
    f = open(arg.result, 'w')
    f.write(str(PageRankResult))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #命令行获取处理何种数据集、随机游走概率、最大迭代次数以及结果存放位置，提供了合理的default值
    parser.add_argument('-size', type=str, default='small' ,help='dataset size, small or big')
    parser.add_argument('-rate', type=float, default=0.2 ,help='walk out rate, the probability of a man opening a new web')
    parser.add_argument('-num_iter', type=int, default=1e3 ,help='max iterations of pageRank')
    parser.add_argument('-result', type=str, default='./result.txt' ,help='file name to store result')
    parser.add_argument('-eps', type=float, default=1e-10 ,help='stop condition, if loss < eps then stop iteration')
    arg = parser.parse_args()
    do(arg)
