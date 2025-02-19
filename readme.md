# tokenizeWordfromDF

作用：配置菜单式对含有文本信息字段的pandas.DataFrame的指定文本字段进行分词，并将分词结果、词性标注、关键词词频等结果作为新的字段拼接到输入的DataFrame中返回，配置菜单的基本格式见app.py。

优点：存档点可以将任务根据占用的内存大小切分为多个小任务并在每组小任务完成后立即保存到硬盘实现断点续运行。

特点：使用多进程最大限度利用现代计算机的多核性能（相较于简单的使用for-loop遍历观测并顺序操作），使用pickle格式存储中间过程以减少过程内存占用并实现存档点功能。

本程序使用thulac（https://github.com/thunlp/THULAC）作为分词工具，为了使用本程序，用户需要对本地安装的thulac（一般在python安装路径下的Lib/site-packages/thulac目录下，如果不希望修改全局生效，也可以将thulac文件夹复制到工作目录下修改）做出如下修改：
1. manage/Preprocesser.py中注释掉如下部分（一般是29-41行）

        def __init__(self, rm_space=False):
            self.isOther = self.is_X(self.otherSet)
            self.isSinglePun = self.is_X(self.singlePunSet)
            self.isHttp = self.is_X(self.httpSet)
            self.isModalParticleSet = self.is_X(self.modalParticleSet)

        def is_X(self, charType):
            def func(c):
                if(c in charType):
                    return True
                else:
                    return False
            return func

并将如下部分解除注释（一般是45-67行）

    def isOther(self, c):
        if(c in self.otherSet):
            return True
        else:
            return False
    
    def isSinglePun(self, c):
        if(c in self.singlePunSet):
            return True
        else:
            return False
    
    def isHttp(self, c):
        if(c in self.httpSet):
            return True
        else:
            return False

    def isModalParticleSet(self, c):
        if(c in self.modalParticleSet):
            return True
        else:
            return False

这样做的目的是为了确保thulac的实例对象能够被正确序列化，否则无法使用在多进程中。
2. (optional 可选) manage/Filter.py中在`self.posSet = ["n","np","ns","ni","nz","v","a","id","t","uw"]`列表中（一般在第8行）加入一个"j"改为`self.posSet = ["n","np","ns","ni","nz","v","a","id","t","uw","j"]`。"j"在thulac中是"简称"的词性标注。顾名思义Filter类定义的是当需要筛选（实例化时filt=True）时的操作流程，它是通过词性来筛选的。对于thulac的作者而言简称并没有太大的意义，故而应该被删去，所以并没有进入self.posSet中。但在很多情况下，特别是在全称并未在文中出现的情况下，专业术语的简称对于判断文章主题是至关重要的。对于简称最好的处理方法时将其映射到全称，这样的话当简称和全称都出现在文章中时可以正确计算代表特定实体的词语出现的次数，但这需要一个字典（映射词表）来完成。由于我处理的主要是财经新闻相关文本，一般专有名词的全称很少出现（一般新闻稿中不会把证监会给完整称呼为证券监督管理委员会），因此采用了直接保留简称的做法。这个问题只是影响最终分词返回结果的一个小瑕疵，并不影响整个程序是否能够运行，因此是否进行修改是一个可选项。

thulac的其他问题：
1. 对于数字的识别存在问题，特别是当存在小数点或百分号时（可能程序将小数点视作英语句号了），我在程序中使用正则表达式`text=re.sub(r"(\d+)\.?(\d+)?%?","24",text)`将文本中出现的所有数字改为24再处理（反正无论是分词还是词性标注数字的值具体是多少是无所谓的，不妨用一个具体且简单的数字替代）。
2. 对于名词修饰名词的识别存在问题，识别依赖于助词"的"。例如其可以正常识别"面包的价格"，但无法正确识别"面包价格"，但后者可能是一种更常见的说法，特别是在前面还有其他限定词已经使用了助词"的"的情况下，例如"魏玛共和国的面包价格在短短2个月的时间内上升了79.08%，突破了50万马克的大关"。这个问题我还没有想好怎么解决，若有用户有修改的注意欢迎在我这里或thulac的github页面那里提issue或直接pull request（尽管thulac似乎已经停止了维护）。

其他：有任何的bug或改进意见都欢迎在issue中提出（当然也欢迎通过nexus_yang@126.com邮件私发给我），会持续改进；requirements.txt仅供参考，应该numpy和pandas任何版本都行吧。