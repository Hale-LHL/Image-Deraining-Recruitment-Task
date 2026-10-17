# Part2-第一部分-任务一

## 基本问题

1. 代码如下：

```commandline
parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet","DerainNet","Mymodel"], help="Model to use")
```

2. 导入代码如下：

```commandline
from models.DerainNet import DerainNet
from models.Mymodel import Mymodel
```

3. 代码如下：

```commandline
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.model == "baseline":
        model = BaselineNet().to(device)
    elif args.model == "unet":
        model = UNet().to(device)
    elif args.model == "DerainNet":
        model = DerainNet().to(device)
    elif args.model == "Mymodel":
        model = Mymodel().to(device)
```
