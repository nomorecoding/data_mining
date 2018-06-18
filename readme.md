## 数据仓库与数据挖掘大作业报告——PageRank算法实现

#### 1500012830 钟易澄

### 1. PageRank算法原理：

####1.1 背景：

​	随着互联网的普及与发展，其所产生的信息量越来越大，如何从如此繁多庞杂的数据中提取出有效的信息一直是数据挖掘的重要课题。一种有效的解决方案是网络搜索引擎，通过关键字筛选来获得有效且重要的信息。PageRank是Google公司创办人Larry Page提出来的对网页相关性和重要程度排名的一个算法。

​	PageRank算法核心思想是：对于一个网页，它的重要程度取决于其他网页对其的评价，而不是其自身所含有的信息。

#### 1.2 算法描述：

​	1). 初始化：构建有向图$G$，每个页面设置相同的PageRank值$\frac{1}N$，这里$N$是所有网页数。

​	2). 更新：每个页面将其当前的PageRank值平均分配到其所包含的出边节点上，则每个页面都获得了相应权值。然后每个页面将所有指向本页面的的入边权值求和，得到新的PageRank值。计算所有页面上一轮与本轮PageRank值之差的绝对值之和，若其小于给定精度则认为以达到收敛，迭代结束。

#### 1.3 改进：

​	由于存在一些网页，其出边为0，对于这样的网页，采用原始的PageRank算法对它计算会使得它的PR值接近1，但这显然是不合理的。所以增加一项阻尼系数(damping factor)$\alpha$，其含义是用户在到达某网页后，以$\alpha$的概率继续点击其内含的链接浏览，以$1-\alpha$的概率打开新的网页进行浏览，从而使得其他节点能获得一定的得分。从而计算一个页面A的PR值的公式为：

$PR(A)=\frac{1-\alpha}{N}+\alpha\Sigma_{v∈B_A}{\frac{PR(v)}{L(v)}} \tag{1}$

其中$B_A$表示链接到A网页的所有网页集合，$L(v)$表示$v$的出度。

写成矩阵形式即：

$R=\alpha MR+\frac{(1-\alpha)}{N}I \tag{2}$

这里$R$即PR值矩阵，$$M_{ij}=\begin{cases}  \frac{1}{L(p_j)}, & if \ j\ links\ to\ i\\  0, & otherwise  \end{cases}$$ 

### 2. 算法实现：

​	根据上述PageRank算法原理，首先根据数据集构建有向图，这里数据集我选用了一大一小两个，大数据集来自http://socialcomputing.asu.edu/datasets/Twitter，具体规模如下：

![pr4](C:\Users\201网咖\Desktop\pr4.png)

其中边数应为85331845而非他所说的85331846...这是一个坑。

小数据集来自SNAP-Stanford，http://snap.stanford.edu/data/web-Stanford.html

![pr3](C:\Users\201网咖\Desktop\pr3.png)

​	有向图的存储可以采用Python的字典，因为使用矩阵存储的空间复杂度是$O(V^2)$的，用邻接表进行存储空间复杂度为$O(V+E)$，对于网页的情况，$G$通常是一个稀疏矩阵，边数远远不及$V^2$的量级，并且Python的字典查找效率是$O(1)$的，比起list的查找高效许多。

​	在迭代更新的过程中，我选择实现公式$(1)$中的算法，因为普通的矩阵乘法复杂度是$O(n^3)$，这对于稀疏矩阵来说是及其浪费的，而通过出边字典找到指向本网页的所有节点，再通过出边字典获取本网页得到的权重值，即函数updatePR中的：

$t\  += 1.0 / len(edgeOut[fromNode]) * v[fromNode] \tag3$

​	这样规避了矩阵乘法，时间复杂度由原先的$O(V^2)$变成了$O(E)$。

​	此外，参考了网上许多博客之后，也可以使用幂法进行更新，但是这样就必须使用矩阵计算，所以我选择用稀疏矩阵存储数据，numpy进行矩阵计算。scipy的稀疏矩阵有许多格式，这里我选用csr，因为其对矩阵运算支持比较多，效率更高，同时比较稳定，而其他比如coo，csc对许多矩阵运算都不支持或者效率极低，dok则和字典相差无几，建立时间比较久。

​	最后考虑到Python本身的效率问题，我还使用了pypy进行编译加速。（pypy的下载链接：http://pypy.org/download.html）

### 3. 实验结果：

​	参考2中的实现细节，我实现了4个版本的PageRank：

​	1) 使用pygraph中的有向图digraph存储图，使用更新公式$(1)$迭代更新的NaivePageRank.py.

​	2) 使用Networkx库自带的PageRank函数处理的PageRankWithNX.py.

​	3) 使用字典存储图和上述更新公式$(1)$迭代更新的PageRankFast.py。

​	4) 使用scipy稀疏矩阵存储图，numpy进行矩阵计算，使用幂法迭代更新的PageRankAdvanced.py。

实验环境为win10系统，CPU为四核Intel i5 7500U，16G内存。软件环境为python==3.6.3，pypy，numpy，scipy，python-graph-core，networkx皆为最新版本。

![pr9](C:\Users\201网咖\Desktop\pr9.png)

​	对于大数据集 1) 和 2) 都会出现Memory Error，因为没有进行内存优化，无法存储下如此多的节点和边。3) 在大数据集上的运行时间为3563s，即1h左右，而4) 在大数据集上运行时间仅为100s不到，不过当我进程开得比较多的时候也会偶尔出现Memory Error（实际上关掉腾讯QQ之后就可以运行...），可见16G内存对于如此大的数据集处理是十分吃力的，即便用上了优化的空间存储方案：

![pr6](C:\Users\201网咖\Desktop\pr6.png)

​	对于小数据集，朴素PageRank算法十分耗时，超过了1h，我没有跑出具体结果。

​	调用Networkx库消耗的时间如下：

![pr2](C:\Users\201网咖\Desktop\pr2.png)

​	使用Python运行PageRankFast.py消耗时间如下：

![pr1](C:\Users\201网咖\Desktop\pr1.png)

​	使用pypy运行PageRankFast.py消耗时间如下：

![pr8](C:\Users\201网咖\Desktop\pr8.png)

​	使用python运行PageRankAdvanced.py消耗时间如下：

![pr7](C:\Users\201网咖\Desktop\pr7.png)

代码运行方法：4个皆可简单地使用python [filename]运行，我都设置了合理的默认参数，当然后两者也可在命令行中声明PageRank的一些参数：

![pr10](C:\Users\201网咖\Desktop\pr10.png)

此外安装了pypy的话也可以使用pypy PageRankFast.py运行，运行另外的文件需要安装pypy下的numpy和scipy。

### 4.结论：

​	从上述实验结果可以看出，朴素的PageRank，空间复杂度和时间复杂度都极高，无法处理大规模的数据集，Networkx库的pagerank实现比较快，但是也无法处理超大规模数据集，并且时间并不如使用numpy和稀疏矩阵优化后的PageRank，当然它还有pagerank_numpy，但这个函数似乎也没对内存进行优化，在小数据集上也无法运行。由于图G的系数特性，在改进了图的存储方式后，PageRank算法可以对更大的数据集使用，使用pypy编译运行的结果基本和Networkx库实现的PageRank性能相当，并且空间复杂度更优，而使用numpy和scipy进一步优化后的算法效率则更高，相比PageRankFast.py又提升了30多倍的速度，一方面可能是numpy对矩阵计算的优化，另一方面应该是对于稀疏矩阵，幂法能更快的迭代至收敛。