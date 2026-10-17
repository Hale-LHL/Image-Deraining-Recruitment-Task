# Part1-任务三

## 问题回答

### 1. 环境搭建

1. 虚拟环境名为qvyv,截图如下:  
   ![配置.png · 353](https://img.cdn1.vip/i/6accd9ea9589c_1791810026.webp)

### 2. 实现模型

1. 模型代码如下：

```commandline
import torch.nn as nn
class  BaselineNet(nn.Module):
    def __init__(self,in_channels =3,num_channels =64,use_1x1conv = True,stride = 1):
        super(BaselineNet, self).__init__()
        self.all_conv2 =nn.ModuleList()
        self.all_bn2 = nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=num_channels,kernel_size=3,padding= 1,stride = stride)
        for i in range(7):
            self.all_conv2.append(nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,padding= 1,stride = stride))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1, padding=0,stride=stride)
        #1×1 卷积永远 padding=0
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(in_channels)
        for i in range(7):
            self.all_bn2.append( nn.BatchNorm2d(num_channels))
        self.relu= nn.ReLU(inplace = True)
        self.conv_out = nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1)

    def forward(self,X):
        Y = self.relu(self.conv1(self.bn1(X)))
        for i in range(7):
            Y =self.relu(self.all_conv2[i](self.all_bn2[i](Y)))
        if self.conv3:
            shortcut = self.conv3(X)# 不能加bn1，这个一定不能缩放，要保持干净
        Y =Y + shortcut
        Y = self.relu(Y)
        rain = self.conv_out(Y)
        return X - rain
#最后要原图减去雨层，这一步别忘了，注意shortcut这个重命名，不要用X，会把原图覆盖掉（最后还要用原图）
'''
注意不能用这样的组合，这样条用7次是同样的权重：
self.conv2 = nn.Conv2d(in_channels=in_channels,out_channels=num_channels,kernel_size=3,padding= 1,stride = stride)
self.bn2 = nn.BatchNorm2d(num_channels)
……
for i in range(7):
    Y =self.relu(self.all_conv2(self.all_bn2(Y)))
'''
```

2. 针对随着网络加深会带来梯度消失、训练困难等问题，解决方法如下：

- 调整初始化权重的大小，不要太大或太小，权重w太小的话,wx会减小x之间的差距，导致深层接到一堆几乎一样的数据，根本学不动
- 加入batch normalization(BN)，将数据约束在0到1以内，也就是在sigmoid等函数梯度最大的区域，以防梯度消失
- 利用残差连接（反向传播用的是∂L/∂x = ∂L/∂y·(1 + ∂F/∂x)），可以缓解梯度消失，保持可训练性
- 将sigmoid等饱和型激活函数换为非饱和性激活函数，减弱梯度消失

  查找资料后补充以下几点：

- 利用梯度裁剪：在梯度超过阈值后，等比缩放，防止出现梯度太小或者太大
- 在浅层直接接分类头算损失，不依赖深层的反向传播，没有连乘的累计，几乎不会出现梯度太小或者太大
- 利用多尺度采样（也就是Part2中的DerainNet的处理方法）
- 利用transformer全局依赖
- 将卷积核换成1xk型的长条形，专门识别雨丝（但如果雨丝是斜的，这种处理的效果应该不好）

### 3.代码实现

1. 结果如图：

![结果.png · 537](https://img.cdn1.vip/i/6acd7a2c02eed_1791851052.webp)

2. 新代码学习笔记

| 代码                                                                               | 作用                                   | 语法                                                                  | 提示                                                                                                       |
| ---------------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| parser = argparse.ArgumentParser(description="描述")                               | 造一个参数解析器（造一个对应类的实例） | descripyion是添加描述（在help文档中可见）                             | 若必填没写会报错：main.py: error: the following arguments are required: --a                                |
| parser.add_argument("--a", type=int, required=True, help="描述")                   | 录入一个必填参数                       | required=True是此参数必填，help是添加描述（在help文档中可见）         | 若参数类型不对会报错：main.py: error: argument --epochs: invalid int value: 'abc'                          |
| parser.add_argument("--a", type=str, default="b", choices=["b", "c"], help="描述") | 录入一个选择性的参数                   | default是默认值，choice列表是可选参数值,add_argument加入一个参数      | 若取值取了choice之外的会报错：main.py: error: argument --model: invalid choice: 'x' (choose from 'b', 'c') |
| parser.add_argument("--a", action="store_true", help="描述")                       | 录入一个开关参数                       | action="store_true"，命令行写了a = 打开（True），不写 = 关闭（False） | 无                                                                                                         |
| parser.add_argument("--a", type=int, nargs="+", default=[8], help="描述")          | 录入一个多值参数                       | nargs="+"表示可以取多值                                               | 无                                                                                                         |
| return parser.parse_args()                                                         | 定义函数返回值                         | parse_args() 返回一个 Namespace 对象                                  | 用 args.a 访问参数值                                                                                       |
| def main(): args = get_args()                                                      | 为调用输入的参数做准备                 | 无                                                                    | 这一行之后，args.x 就是在命令行敲的值                                                                      |
| if **name** == "**main**": main()                                                  | 运行main()时调用                       | 无                                                                    | 防止import时异常调用                                                                                       |
| python main.py --data_dir "数据集地址" --model baseline --mode train --epochs 100  | 命令行的启动指令                       | python main.py --a b --a' b'每个元素用空格隔开                        | 要在main.py所在地址执行此指令                                                                              |
| cd /d D:/qvyv                                                                      | 改换命令行地址                         | /d是换盘必备（从C:/换到D:/）                                          |                                                                                                            |
| python main.py --help                                                              | 查看参数的相关描述                     | 无                                                                    | 命令行输入                                                                                                 |

3. 改PSNR

```commandline


def cal_psnr(img_target,img_pred,data_range):
    error = img_pred - img_target
    mse = np.mean(error ** 2)
    if mse ==0:
        return float('inf')
    else:
        return 10 * np.log10(data_range ** 2 / mse)



def calculate_metrics(img1, img2, lpips_fn):

    img1_np = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img2_np = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img1_np = np.clip(img1_np, 0, 1)
    img2_np = np.clip(img2_np, 0, 1)

    batch_psnr, batch_ssim = 0, 0
    for i in range(img1_np.shape[0]):
        batch_psnr += cal_psnr(img1_np[i], img2_np[i], data_range=1.0)
        batch_ssim += ssim(img1_np[i], img2_np[i], data_range=1.0, multichannel=True, channel_axis=2, win_size=7)

    avg_psnr = batch_psnr / img1_np.shape[0]
    avg_ssim = batch_ssim / img1_np.shape[0]

    img1_lpips = img1.clone() * 2 - 1
    img2_lpips = img2.clone() * 2 - 1

    with torch.no_grad():
        lpips_score = lpips_fn(img1_lpips, img2_lpips).mean().item()

    return avg_psnr, avg_ssim, lpips_score

```
