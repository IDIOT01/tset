#!/bin/bash

# 启动循环，从0到9，步长为1
for i in {0..1}; do
    # 在后台执行Python脚本
    nohup python client.py -c configs/template.yaml -p 8089 --idx $i >> ./testLog/test.out 2>&1 &
done

# 等待所有后台进程完成
wait