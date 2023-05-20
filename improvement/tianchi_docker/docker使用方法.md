# docker使用方法

阿里云镜像仓库:https://cr.console.aliyun.com/repository/cn-hangzhou/llhh/tianchi-submit/details

docker上传的镜像中包含，环境、预训练模型和推理脚本



一些实用方法

```
export DOCKER_REGISTRY=registry.cn-hangzhou.aliyuncs.com/llhh


# 进入到sh目录下执行下列命令自动上传镜像
sh docker_build.sh [version]

# 运行镜像查看是否有bug 
docker run $DOCKER_REGISTRY/tianchi-submit:[version] sh run.sh
# 设置gpu
docker run --gpus 0 $DOCKER_REGISTRY/tianchi-submit:[version] sh run.sh
# 进入环境
docker run $DOCKER_REGISTRY/tianchi-submit:[version] /bin/sh
```

