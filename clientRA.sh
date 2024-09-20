#!/bin/bash

# 启动循环，从0到9，步长为1
for i in {0..19}; do
    # 在后台执行Python脚本
    nohup python client.py -c configs/fedRA.yaml -p 8085 --idx $i >> ./testLog/clientRA0909.out 2>&1 &
done

# 等待所有后台进程完成
wait