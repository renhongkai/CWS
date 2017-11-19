# CWS
中文分词
#### word_segement_model.py:分词模型，能跑通，但速度很慢（之前的代码没有记录时间，所以没有记录确切的时间，
					             但是github上的代码添加了记录时间）。11月18日晚学长说这个代码在服务器上占用了8
					             个线程，一跑这个代码就会占用很多服务器资源，其他人就没法用服务器了。
#### word_segement_model_cuda.py：使用cuda加速计算，调用utils文件
#### word_segement_model_Dataset.py：使用pytorch的Dataset处理数据。调用utils_dataset文件。没跑通。
#### 使用的训练数据文件为：pku_training_BIO.utf8
#### 使用的测试数据文件为：pku_test_gold_BIO.utf8
