'''
朴素 PageRank，几乎不能运行
'''
import argparse #解析命令行

import networkx as nx
import time

#数据集来自 http://socialcomputing.asu.edu/datasets/Twitter
#           https://snap.stanford.edu/data/egonets-Twitter.html
#大数据集包含1000万个顶点和8000万条边，小数据集包含8万个顶点，170万条边
bigDataFilename = './Twitter-dataset/data/edges.csv'
smallDataFilename = './web-Stanford.txt'


def buildFromFile(size = 'small'):
    '''
    从文件中获取节点和边的信息，构建入边和出边字典
    @param_in f：str，有向图文件名
    @param_in size：str，数据集类型，small为小数据集，否则为大数据集

    @param_out edgeOut：出边字典
    @param_out edgeIn：入边字典
    @param_out nodes：节点字典
    '''
    dg = nx.DiGraph()
    f = smallDataFilename if size == 'small' else bigDataFilename
    with open(f) as file: 
        count = 0 
        for line in file.readlines():
            if size == 'small':#小数据集先去掉开头几行说明
                if count < 4: count+=1;continue
            start, end = (int(x) for x in line.split()) if size == 'small' else (int(x) for x in line.split(','))
            #pygraph 添加节点和边不能重复，需要先进行判断
            if not dg.has_node(start):
                dg.add_node(start)
            if not dg.has_node(end):
                dg.add_node(end)
            if not dg.has_edge(start, end):
                dg.add_edge(start, end)
    return dg


def PageRank(dg, rate=0.85, max_iterations=1000, epsilon=1e-10):
    '''
    朴素PageRank
    @param_in dg；digraph, 有向图G
    @param_in rate: float, 阻尼系数α
    @param_in max_iteration: int，最大迭代次数
    @param_in epsilon：Loss精度

    @param_out page_ranks：PR值向量
    '''
    nodes = dg.nodes()
    N = len(nodes)

    if N == 0:
        return {}
    rank = dict.fromkeys(nodes, 1.0 / N)  # 给每个节点赋予初始的PR值: 1/N
    damping_value = (1.0 - rate)  # 更新公式中的(1−α)

    flag = False
    for i in range(max_iterations):
        loss = 0 #损失
        for node in nodes:
            t = 0
            if len(list(dg.successors(node))) == 0:
                t = damping_value
            else:
                for incident_page in dg.predecessors(node):  # 遍历所有入边
                    t += rate * (rank[incident_page] / len(list(dg.successors(incident_page)))) #更新的后半部分
                t += damping_value
            loss += abs(rank[node] - t)  # 绝对值
            rank[node] = t


        if loss < epsilon:
            flag = True
            break
    if flag:
        print("finished in %s iterations!" % i)
    else: #超过最大迭代次数
        print("max iterations exceeded!")
    return rank

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #命令行获取处理何种数据集、随机游走概率、最大迭代次数以及结果存放位置，提供了合理的default值
    parser.add_argument('-size', type=str, default='small' ,help='dataset size, small or big')
    parser.add_argument('-rate', type=float, default=0.15 ,help='walk out rate, the probability of a man opening a new web')
    parser.add_argument('-num_iter', type=int, default=1e3 ,help='max iterations of pageRank')
    parser.add_argument('-eps', type=float, default=1e-10 ,help='stop condition, if loss < eps then stop iteration')
    parser.add_argument('-result', type=str, default='./result.txt' ,help='file name to store result')
    arg = parser.parse_args()

    print("PageRank begins!")
    startReadTime = time.time()
    dg = buildFromFile(arg.size)
    print("Reading elapsed time: ", time.time() - startReadTime)

    startTime = time.time()
    page_ranks = PageRank(dg, 1 - arg.rate, arg.num_iter, arg.eps)
    print("PageRank iter elapsed time: ", time.time() - startTime)