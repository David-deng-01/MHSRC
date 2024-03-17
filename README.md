# MHSRC
Multimodal Humor and Sarcasm Recognition in Chinese: Dataset Construction, Association Analysis and Fine-grained Analysis

**0. 数据下载**

三模态特征文件以及json文件下载地址：[BaiduNetDisk]()

每个文件的执行顺序，我们通过文件名中的编号来体现。

---

**1. 划分数据集**

你可以下载我们已经划分好的结果[[BaiduNetDisk]()]，或者通过以下方式自行划分

配置文件：`config/split_data/main.yaml`

运行：`python 1_split_data.py`

---

**2. 生成缓存文件**

你可以下载我们已经处理好的缓存文件[[BaiduNetDisk]()]，或者通过以下方式自行生成

*确保你已经完成了第一步。*考虑到数据是标签不平衡的，因此我们对训练集过采样、验证集测试集欠采样生成缓存文件。

配置文件：`config/generate_cache/main.yaml`

运行：`python 2_generate_cache.py`

---

**3. 生成缓存文件（类别分类）**

你可以下载我们已经处理好的缓存文件[[BaiduNetDisk]()]

*确保你已经完成了第一步。*

配置文件：`config/type_classifier/generate_cache.yaml`

运行：`python 3_generate_classifier_cache.py`

---

**4. 训练小模型**

*确保你已经完成了第二步*

配置文件：`config/small_model/main.yaml`

运行：`4_train_small_model.py`

---

**5. 训练多任务模型**

*确保你已经完成了第二步*

配置文件：`config/small_model/ml/main.yaml`

运行：`5_train_small_model.py`

---

**6. 训练类别分类器**

*确保你已经完成了第三步*

配置文件：`config/type_classifier/main.yaml`

运行：`6_train_type_classifier_model.py`