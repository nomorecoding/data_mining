'''
PageRank的numpy+稀疏矩阵实现，csr稀疏矩阵存储有向图，numpy矩阵计算迭代。
'''


import argparse #解析命令行
import numpy as np
from scipy import sparse
import time #记录时间
#数据集来自 http://socialcomputing.asu.edu/datasets/Twitter
#           https://snap.stanford.edu/data/egonets-Twitter.html
#大数据集包含1000万个顶点和8000万条边，小数据集包含8万顶点，170万条边

bigDataFilename = './Twitter-dataset/data/edges.csv'
smallDataFilename = './web-Stanford.txt'


def dataset2csr(size='small'):
    '''
    将数据集转化为csr稀疏矩阵存储，NODES和EDGES为数据集提供的节点数和边数，需要提前获取减少一次循环
    '''
    row = []
    col = []   
    f = smallDataFilename if size=='small' else bigDataFilename
    with open(f) as file:  
        count = 0
        for line in file.readlines():
            if size == "small":#小数据集先去掉开头几行说明
                if count < 4: count+=1;continue
            origin, destiny = (int(x)-1 for x in line.split()) if size == 'small' else (int(x)-1 for x in line.split(','))
            row.append(destiny)
            col.append(origin)
    NODES = 281903 if size=="small" else 11316811
    EDGES = 2312497 if size=="small" else 85331845
    return(sparse.csr_matrix(([True]*EDGES,(row,col)),shape=(NODES,NODES)))           

def PageRank(G, rate=0.85, epsilon=10**-10, max_iter = 1e3):
    '''
    用numpy 和 sparse矩阵高效计算PageRank

    @param_in G : boolean 邻接矩阵. np.bool8, 若G(j,i)为真,则i->j.
    @param_in rate: float, 转移概率.
    @param_in epsilon: float, loss精度

    @param_out  (rank, count): tuple(np.array，int)，(PageRank值向量，迭代次数).

    '''    
    n,_ = G.shape
    #循环外提前计算好常数
    deg_out_rate = G.sum(axis=0).T/rate #出度/rate
    #初始化
    ranks = np.ones((n,1))/n 
    count = 0
    flag = True
    while flag and count < max_iter:        
        count +=1
        with np.errstate(divide='ignore'): #忽略divide by 0
            newRank = G.dot((ranks/deg_out_rate)) 
        newRank += (1-newRank.sum()) / n
        #计算L1损失
        if np.linalg.norm(ranks-newRank,ord=1)<=epsilon:
            flag = False  
        ranks = newRank
    return(ranks, count)



if __name__ == "__main__":

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
    csr = dataset2csr(arg.size)
    print("Reading elapsed time: ", time.time() - startReadTime)

    startTime = time.time()
    pr,iters = PageRank(csr, 1 - arg.rate, arg.eps, arg.num_iter)
    print("PageRank iter elapsed time: ", time.time() - startTime)

    print ('Iterations: {0}'.format(iters))
    f = open(arg.result, 'w')
    f.write(str(pr))
    f.close()