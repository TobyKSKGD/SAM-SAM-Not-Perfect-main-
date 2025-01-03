# SAM性能评估（SAM-Not-Perfect-main）
[Segment Anything Is Not Always Perfect](https://arxiv.org/abs/2304.05750)展示了Segment Anything Model（SAM）的性能。

为了进一步观察SAM的性能，我们在[原代码](https://github.com/LiuTingWed/SAM-Not-Perfect)的基础上简单添加了模型分割每张图片的平均时长，以此将一部分的注意力着眼于SAM的分割速度，为后续SAM的改进提供参考。

此外我们还在原代码的基础上添加了常见的mIoU评价指标。

最后为了方便查看模型的分割情况，我们在`amg.py`中加入的进度条以显示当前分割进度。
# 如何开始

本代码仅在*Segment Anything Is Not Always Perfect*的基础上做出简单的调整，使用方法可完全参考[原文代码](https://github.com/LiuTingWed/SAM-Not-Perfect)。
