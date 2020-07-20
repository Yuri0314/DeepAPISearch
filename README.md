# DeepAPISearch
简介：阿巴阿巴阿巴阿巴

# python-model文件夹中主要模型进度介绍

 prepareData.ipynb ：处理类似test.txt的文件，将API和描述中的脏标点符号删除，对API方法名按照大写字母分割，对于无描述的API使用分割后的方法名替代。（已继承进下方的similarity模型中）**~~下一步实现功能：将所有数据大写字母转为小写字母~~**（已实现）
 
 similarity.ipynb : 主要模型代码,目前仅实现训练部分，可以通过test.txt测试。**下一步实现功能：~~1.继承上方处理数据（大小写转换）~~（已实现）2.添加unknown，防止预测的时候对于词表中不存在的单词报错 3.添加模型预测函数**
 
 similarity.py ：与上方jupyter文件内容相同，调试使用
 
 seq2seq.py ：通过翻译模型实现代码API推荐，目前模型完整。**下一步实现添加unknown，防止预测报错**

