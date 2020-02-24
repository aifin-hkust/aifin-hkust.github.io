
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题,或者转换负号为字符串

# 训练的语料
sentences1 = [['设计', '显示','界面','目标','图片','产品','对象','外观设计','属性','主视图 ']]
sentences2 = [['图像','人脸','装置','目标','信息','进行','特征','实施','存储介质','电子设备']]
sentences3 = [['图像','界面','装置','设计','特征','信息','变化','进行','用户','目标']]

# 利用语料训练模型
model1 = Word2Vec(sentences1,window=5, min_count=1)
model2 = Word2Vec(sentences2,window=5, min_count=1)
model3 = Word2Vec(sentences3,window=5, min_count=1)
# 基于2d PCA拟合数据
X_1 = model1[model1.wv.vocab]
X_2 = model2[model2.wv.vocab]
X_3 = model3[model3.wv.vocab]
pca = PCA(n_components=2)
result_1 = pca.fit_transform(X_1)
result_2 = pca.fit_transform(X_2)
result_3 = pca.fit_transform(X_3)
# 可视化展示
plt.scatter(result_1[:, 0], result_1[:, 1])
plt.scatter(result_2[:, 0], result_2[:, 1])
plt.scatter(result_3[:, 0], result_3[:, 1])
words1 = list(model1.wv.vocab)
words2 = list(model2.wv.vocab)
words3 = list(model3.wv.vocab)
for i, word in enumerate(words1):
    plt.annotate(word, xy=(result_1[i, 0], result_1[i, 1]),bbox=dict(fc='#C4E1FF', ec='k',lw=1 ,alpha=0.5))
for i, word in enumerate(words2):
    plt.annotate(word, xy=(result_2[i, 0], result_2[i, 1]),bbox=dict(fc='#003D79', ec='k',lw=1 ,alpha=0.5))
for i, word in enumerate(words3):
    plt.annotate(word, xy=(result_3[i, 0], result_3[i, 1]),bbox=dict(fc='#2894FF', ec='k',lw=1 ,alpha=0.5))
plt.show()
