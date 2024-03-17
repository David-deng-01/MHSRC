# MHSRC

Multimodal Humor and Sarcasm Recognition in Chinese: Dataset Construction, Association Analysis and Fine-grained Analysis

**0. Download the dataset**

Three-mode feature file and json file download address: [[BaiduNetDisk]()]

The execution order of each file is reflected by the number in the file name.

---

**1. Partition data set**

You can download the results we've already divided [[BaiduNetDisk]()], or divide them yourself as follows

Configuration file: `config/split_data/main.yaml`

Run: `python 1_split_data.py`

---

**2. Generate cache file**

You can download the cached file we have already processed [[BaiduNetDisk]()], or you can generate your own in the following way

*Make sure you've completed step 1.* Considering that the data is label unbalanced, we oversample the training set and undersample the verification set and test set to generate a cache file.

Configuration file: `config/generate_cache/main.yaml`

Run: `python 2_generate_cache.py`

---

**3. Generate cache files (category classification)**

You can download the cache file that we have processed [[BaiduNetDisk]()]

*Make sure you've completed step 1.*

Configuration file: `config/type_classifier/generate_cache.yaml`

Run: `python 3_generate_classifier_cache.py`

---

**4. Training Small parameter quantity model**

*Make sure you've completed step 2.*

Configuration file: `config/small_model/main.yaml`

Run: `4_train_small_model.py`

---

**5. Train a multitasking model**

*Make sure you've completed step 2.*

Configuration file: `config/small_model/ml/main.yaml`

Run: `5_train_small_model.py`

---

**6. Train the class classifier**

*Make sure you've completed step 3.*

Configuration file: `config/type_classifier/main.yaml`

Run: `6_train_type_classifier_model.py`