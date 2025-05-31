# #接口调用
# from classify import RumourDetectClass

# # 初始化检测器（首次加载约2-3秒）
# detector = RumourDetectClass()

# # 示例检测
# samples = [
#     "Swiss museum accepts looted Nazi art http://t.co/gxqAmjNSoV",
#     "BREAKING: Officer involved in #Ferguson shooting identified",
#     "Weather forecast shows sunny skies for tomorrow"
# ]

# for text in samples:
#     result = detector.classify(text)
#     print(f"文本: {text[:40]}... | 预测: {'谣言' if result == 1 else '非谣言'}")

import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    # 测试张量计算
    x = torch.rand(5, 3).cuda()
    print(f"张量设备: {x.device}")
else:
    print("警告: PyTorch 未检测到 GPU 支持！")