# #!/bin/bash

#!/bin/bash

# 启动循环，从0到9，步长为1
for i in {0..19}; do
    # 在后台执行Python脚本
    nohup python client.py -c configs/fedprox.yaml -p 8083 --idx $i >> ./testLog/clientProx0908.out 2>&1 &
done

# 等待所有后台进程完成
wait


# # 启动循环，从0到9，步长为1
# for i in {0..14}; do
#     # 在后台执行Python脚本，并重定向输出
#     nohup python client.py -c configs/fedproxCuda0.yaml -p 8083 --idx $i >> ./testLog/clientProx0904.out 2>&1 &
# done

# for i in {15..29}; do
#     # 在后台执行Python脚本，并重定向输出
#     nohup python client.py -c configs/fedproxCuda1.yaml -p 8083 --idx $i >> ./testLog/clientProx0904.out 2>&1 &
# done

# # 等待所有后台进程完成
# wait
