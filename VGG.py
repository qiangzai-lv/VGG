import torch.nn as nn
import torch


# 定义VggNet网络模型
class VGG(nn.Module):
    # init()：进行初始化，申明模型中各层的定义
    # features：make_features(cfg: list)生成提取特征的网络结构
    # num_classes：需要分类的类别个数
    # init_weights：是否对网络进行权重初始化
    def __init__(self, features, num_classes=1000, init_weights=False):
        # super：引入父类的初始化方法给子类进行初始化
        super(VGG, self).__init__()
        # 生成提取特征的网络结构
        self.features = features
        # 生成分类的网络结构
        # Sequential：自定义顺序连接成模型，生成网络结构
        self.classifier = nn.Sequential(
            # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            # ReLU(inplace=True)：将tensor直接修改，不找变量做中间的传递，节省运算内存，不用多存储额外的变量
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        # 如果为真，则对网络参数进行初始化
        if init_weights:
            self._initialize_weights()

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 将数据输入至提取特征的网络结构，N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        # 图像经过提取特征网络结构之后，得到一个7*7*512的特征矩阵，进行展平
        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        x = torch.flatten(x, start_dim=1)
        # 将数据输入分类网络结构，N x 512*7*7
        x = self.classifier(x)
        return x

    # 网络结构参数初始化
    def _initialize_weights(self):
        # 遍历网络中的每一层
        # 继承nn.Module类中的一个方法:self.modules(), 他会返回该网络中的所有modules
        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # uniform_(tensor, a=0, b=1)：服从~U(a,b)均匀分布，进行初始化
                nn.init.xavier_uniform_(m.weight)
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # constant_(tensor, val)：初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 正态分布初始化
                nn.init.xavier_uniform_(m.weight)
                # 将所有偏执置为0
                nn.init.constant_(m.bias, 0)


# 生成提取特征的网络结构
# 参数是网络配置变量，传入对应配置的列表（list类型）
def make_features(cfg: list):
    # 定义空列表，存放创建的每一层结构
    layers = []
    # 输入图片是RGB彩色图片
    in_channels = 3
    # for循环遍历配置列表，得到由卷积层和池化层组成的一个列表
    for v in cfg:
        # 如果列表的值是M字符，说明该层是最大池化层
        if v == "M":
            # 创建一个最大池化层，在VGG中所有的最大池化层的kernel_size=2，stride=2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 否则是卷积层
        else:
            # in_channels：输入的特征矩阵的深度，v：输出的特征矩阵的深度，深度也就是卷积核的个数
            # 在Vgg中，所有的卷积层的padding=1，stride=1
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            # 将卷积层和ReLU放入列表
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    # 将列表通过非关键字参数的形式返回，*layers可以接收任意数量的参数
    return nn.Sequential(*layers)


# 定义cfgs字典文件，每一个key代表一个模型的配置文件，如：VGG11代表A配置，也就是一个11层的网络
# 数字代表卷积层中卷积核的个数，'M'代表池化层的结构
# 通过函数make_features(cfg: list)生成提取特征网络结构
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 实例化给定的配置模型,这里使用VGG16
# **kwargs表示可变长度的字典变量，在调用VGG函数时传入的字典变量
def vgg(model_name="vgg16", **kwargs):
    # 如果model_name不在cfgs，序会抛出AssertionError错误，报错为参数内容“ ”
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    # 得到VGG16对应的列表
    cfg = cfgs[model_name]
    # 实例化VGG网络
    # 这个字典变量包含了分类的个数以及是否初始化权重的布尔变量
    model = VGG(make_features(cfg), **kwargs)
    return model