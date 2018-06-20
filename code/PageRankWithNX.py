"""
apply networkx to calculate PageRank
"""
import argparse #解析命令行

import time

import networkx as nx

#数据集来自 http://socialcomputing.asu.edu/datasets/Twitter
#           https://snap.stanford.edu/data/egonets-Twitter.html
#大数据集包含1000万个顶点和8000万条边，小数据集包含8万个顶点，170万条边
bigDataFilename = './Twitter-dataset/data/edges.csv'
smallDataFilename = './web-Stanford.txt'

def buildFromFile(G, size="small"):
    global file
    f = smallDataFilename if size == 'small' else bigDataFilename
    with open(f) as file:  
        count = 0
        for line in file.readlines():
            if size == "small":#小数据集先去掉开头几行说明
                if count < 4: count+=1;continue
            start, end = (int(x)-1 for x in line.split()) if size == 'small' else (int(x)-1 for x in line.split(','))
            G.add_edge(start, end)
    return G


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #命令行获取处理何种数据集、随机游走概率、最大迭代次数以及结果存放位置，提供了合理的default值
    parser.add_argument('-size', type=str, default='small' ,help='dataset size, small or big')
    parser.add_argument('-rate', type=float, default=0.15 ,help='walk out rate, the probability of a man opening a new web')
    parser.add_argument('-num_iter', type=int, default=1e3 ,help='max iterations of pageRank')
    parser.add_argument('-eps', type=float, default=1e-10 ,help='stop condition, if loss < eps then stop iteration')
    parser.add_argument('-result', type=str, default='./result.txt' ,help='file name to store result')
    parser.add_argument('-type', type=str, default='scipy' ,help='pagerank algorithm type. naive for pagerank(), numpy for pagerank_numpy(), scipy for pagerank_scipy(), google for google_matrix()')
    arg = parser.parse_args()

    print("PageRank begins!")
    beginReadTime = time.time();
    G = nx.DiGraph()
    G = buildFromFile(G, arg.size)
    print("building Graph elapsed time: ", time.time() - beginReadTime)

    if arg.type == 'naive':
        beginTime = time.time()
        PageRankResult = nx.pagerank(G, alpha=1-arg.rate, max_iter=int(arg.num_iter), tol=arg.eps)
        print("PageRank elapsed time: ", time.time() - beginTime)
    elif arg.type == 'numpy':
        beginTime = time.time()
        PageRankResult = nx.pagerank_numpy(G, alpha=1-arg.rate, max_iter=int(arg.num_iter), tol=arg.eps)
        print("pagerank_numpy elapsed time: ", time.time() - beginTime)
    elif arg.type == 'scipy':
        beginTime = time.time()
        PageRankResult = nx.pagerank_scipy(G, alpha=1-arg.rate)
        print("PageRank_scipy elapsed time: ", time.time() - beginTime)
    elif arg.type == 'google':
        beginTime = time.time()
        PageRankResult = nx.google_matrix(G, alpha=1-arg.rate)
        print("Google Matrix elapsed time: ", time.time() - beginTime)

    f = open(arg.result, 'w')
    f.write(str(PageRankResult))
    f.close()