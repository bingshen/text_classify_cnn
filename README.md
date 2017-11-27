# text_classify_cnn

kaggle上的那道电影评论情感分析
题目链接：https://www.kaggle.com/c/word2vec-nlp-tutorial/data

之前写过两篇解法。一篇是使用word2vec做的，还有一篇用的LSTM+attention

这里采用cnn写一次

最后效果为0.9447+ 略比LSTM+Attention高一点点

网络结构采用3种不同规格的1维卷积核卷积词向量。然后把结果拼接在一起，取每个卷积核的最大值，然后放入全连接网络拟合。
最后的模型采用了l2的正则化