# 脑卒中严重程度分级

训练BERT模型对脑卒中NIHSS评分进行预测, 0-1分为正常, 2-4分为轻度, 5-15分为中度, 16-20分为重度, 21-42分为重度以上。

从0开始的代码实现为 `脑卒中预测实验.ipynb` 。

训练脚本为 `train.py` , 安装依赖包直接运行即可。
```shell
pip install torch transformers sentencepiece
```

model文件夹下存放了一个训练了100个epoch的模型, 验证脚本为 `eval.py` , 用于验证模型的准确率, 不过验证数据集 `data/test.jsonl` 是自己瞎写的, 仅供参考。

| Epoch | Train_loss | PPL | Eval Acc | Test Acc |
| --- | --- | --- | --- | --- |
| 100 | 0.0019 | 0.1283 | 1.00 | 0.7 |

