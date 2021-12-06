# NLP-beginner

> 此项目包含复旦大学邱锡鹏老师的NLP入门练习的四个任务 [NLP-Beginner：自然语言处理入门练习](https://github.com/FudanNLP/nlp-beginner)  
> 此项目主要通过借鉴了网上相关资源以及自己的理解完成，在此对所有相关的作者表示感谢。
如果造成了侵权，相关作者可联系我进行协商修改.

## 任务介绍
### [任务一：基于深度学习的文本分类]
> Paper: [**Convolutional Neural Networks for Sentence Classification**](https://arxiv.org/abs/1408.5882)  

### [任务二：基于注意力机制的文本匹配]
> Paper: [**Enhanced LSTM for Natural Language Inference**](https://arxiv.org/pdf/1609.06038v3.pdf)
### [任务三：基于LSTM-CRF的序列标注]
> Paper: [**Neural Architectures for Named Entity Recognition**](https://arxiv.org/pdf/1603.01360.pdf)
### [任务四：基于神经网络的语言模型]
> 用LSTM来训练字符级的语言模型，暂时只进行[**古诗生成**](https://github.com/WangXiaoCao/poetry-generation) ，后续更新其他生成并进行困惑度计算。

代码所用数据在各自部分代码中的`datasets`文件夹中。

## 代码结构介绍
所有项目按照统一结构完成，如下：  
arguments.py 包含数据处理、模型、以及训练所用的全部参数。  
data_loader.py 包含数据预处理、数据加载器  
model.py 包含项目所用的神经模型  
framework.py 包含模型训练、验证和测试三个函数。  

## 使用

所需环境：  
> python 3.6  
> torch > 1.2.0  
> numpy  
> sklearn

训练：  
> python train.py

测试：
> python test.py
