# Streaming FL

## 1. 环境配置

- 创建新环境

  ```bash
    conda create -n sfl python=3.10
    conda activate sfl
  ```

- 安装pytorch

  ```bash
    # install pytorch according to you own situation
    pip3 install torch torchvision torchaudio
  ```

- 安装其它依赖

  ```bash
  pip install flwr[simulation] wandb omegaconf timm hydra-core
  ```

## 2. 如何运行程序？

### 2.1 模拟(simulation)

详细内容请查看`main.py`，运行以下命令行

```bash
python main.py -c <YOUR_CONFIG_PATH>
```

### 2.2 真正分布式训练

需要创建一个shell脚本，来运行服务器和客户端，是以for循环的样式启动所有客户端的

```sh
#!/bin/bash
set -e
#cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Variables
device="cuda:0"
cfg_path="<YOUR_CONFIG_PATH>"

echo "Starting server"
python server.py -c $cfg_path&
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 19`; do
    echo "Starting client $i"
    python client.py -c $cfg_path --idx $i &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
```

### 2.3 测试客户端的所有功能

可以指定`--dry`为`True`来测试客户端的所有功能：

- get/set parameters
- train/evaluate

```bash
python client.py -c <YOUR_CONFIG_PATH> --dry True
```

## 3. 如何增加新的算法

**注意**，在运行`模拟(simulation)`的时候，所有客户端是无状态的，也就是说，每一轮都会重新创建`client_num`个客户端，所以如果想保存历史数据的话，可以尝试运行分布式训练，或者像`clients\fedbn_client.py`里面保存历史数据

关于FL client和 strategy中执行的顺序，可以查看：<https://flower.ai/docs/framework/how-to-implement-strategies.html>

1. 定义新的客户端
   - 在`clients`文件夹里头新加一个py，继承BaseClient，并修改原有函数
   - 在`clients\client_factory.py`中将自己的客户端加进去CLIENT_LIST中（类似于timm和huggingface的注册）
   - 客户端只有4个函数，get/set parameter, fit/eval，如果你的客户端行为与BaseClient相同，那么就不用额外写新函数（因为是继承关系），下面的策略，训练函数同理

2. 定义新的策略（其实是服务器）
   - 所有步骤同1

3. 定义新的训练函数
   - 在`engine`文件夹里头新加一个py，继承BaseStrategy，并修改train和eval两个函数
   - 在`utils.py`里头更新TRAINER_LIST（类似于timm和huggingface的注册）

4. 注册新的模型
   - 在`models`文件夹里头新建一个py，继承nn.Module
   - 在`utils.py`里头更新MODEL_LIST（类似于timm和huggingface的注册）

5. 损失函数
   - 损失函数目前只支持 nn.XX，也就是nn.CrossEntropy,nn.MSELoss,nn.MAELoss等
   - 如果想自己注册的话，也可以仿照`utils.py`中的build_trainer函数，重写我的`build_criterion`

6. 修改配置文件
   - 增加一个新的配置文件，指定任务类型，算法类型，和算法特有的参数
   - 运行如下命令即可

      ```bash
        python main.py -c <YOUR_CONFIG_PATH>
        ```

## 4. 代码架构

```bash
├─clients # 存放客户端
├─configs # 存放设置
├─data # 数据集定义
├─datasets # 数据集文件存储
├─engine # 训练代码
├─logs # 运行日志
├─models # 模型定义
├─other_baseline # 其它baseline
│  └─FedWeIT
└─strategies # 服务器的策略
```
