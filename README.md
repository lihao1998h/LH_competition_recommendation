# 推荐系统竞赛通用解决方案
根据大多数的Topline，数据竞赛的流程一般是数据探索性分析EDA->数据预处理->特征工程->model->submission


赛题类型：CTR预估

算法模型：lightGBM


# 特征工程
特征维数：64

# 运行环境
- 操作系统 Ubuntu 14.04.4 LTS (GNU/Linux 4.2.0-27-generic x86_64)
- 内存 128GB
- CPU 32  Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
- 显卡 TITAN X (Pascal) 12GB
- 语言 Python3.6
- Python依赖包
  1. Keras==2.0.6  
  2. lightgbm==0.1  
  3. matplotlib==2.0.0  
  4. numpy==1.11.3  
  5. pandas==0.19.2  
  6. scikit-learn==0.18.1  
  7. scipy==0.18.1  
  8. tensorflow-gpu==1.2.1  
  9. tqdm==4.11.2  
  10. xgboost==0.6a2  
- 其他库
  LIBFFM v121










# ML-Data-Mining
框架化数据竞赛

根据大多数的Topline，数据竞赛的流程一般是数据探索性分析EDA->数据预处理->特征工程->model->submission

仅针对标准的机器学习回归/分类问题，特征工程+树模型+模型融合确实是最好的模型。

目前支持竞赛类型：
toy：单变量分类、单变量回归

medium：时间序列预测



# 文件介绍

main.py是主程序，设置完参数后直接运行即可

setting.py是程序的初始化设置

EDA_Base.py是EDA

Preprocessing.py是数据预处理

feature_engineering.py是特征工程



...