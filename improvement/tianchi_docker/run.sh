#!/bin/sh
CURDIR="`dirname $0`" #获取此脚本所在目录
echo $CURDIR
cd $CURDIR #切换到该脚本所在目录
#打印GPU信息
#nvidia-smi
#执行 xuanshou.py
python3 ./code/main.py