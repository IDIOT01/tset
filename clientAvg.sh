# #!/bin/bash



for i in {0..19}; do
    # 在后台执行Python脚本
    nohup python client.py -c configs/fedavgDense.yaml -p 8082 --idx $i >> ./testLog/clientAvg0909.out 2>&1 &
done
wait
# # 启动循环，从0到9，步长为1
# for i in {0..14}; do
#     # 在后台执行Python脚本，并重定向输出
#     nohup python client.py -c configs/fedavgDenseCuda0.yaml -p 8082 --idx $i >> ./testLog/clientAvg0903.out 2>&1 &
# done

# for i in {15..29}; do
#     # 在后台执行Python脚本，并重定向输出
#     nohup python client.py -c configs/fedavgDenseCuda1.yaml -p 8082 --idx $i >> ./testLog/clientAvg0903.out 2>&1 &
# done

# # 等待所有后台进程完成
# wait
