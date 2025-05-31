## 训练环境

GPU NVIDIA GEFORCE RTX 3060 LAPTOP


## 环境配置
```bash

# 安装依赖

pip install torch transformers pandas scikit-learn  hf_xet
pip install transformers[torch]
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#cuda版本：12.9，但torch最高支持12.8
#transformers版本：transformers in d:\python\lib\site-packages (4.52.3)
#数据文件放在./data下
```
## 训练
```bash
python train.py
#首次运行时需下载预训练模型，可能需要vpn
```


## 测试
```bash
python test.py
```

