@[toc]
# 0. BERT介绍
google 在**2018年放出的大杀器**， 作为当前NLP的最新技术，此模型在**NLP的多个上游下游问题上都取得很好的成绩，刷新记录**， 具体原理可以自行google， 这样一个新的技术， **值得我们学习一下， 下面我将给出一个使用BERT进行情感分类预测及代码实例， 供大家分享学习。**

# 1. BERT配置
要求配置如下, tensorflow版本不能太低。
```bush
tensorflow >= 1.11.0   # CPU Version of TensorFlow.
tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
```

## 1.1. clone BERT 代码
首先clone 官方代码，地址如下
```
https://github.com/google-research/bert.git
```
之后我们**主要专注于run_classifier.py**， 因为这个py文件是BERT的分类任务接口， **接下来我们主要就是对这个文件进行修改， 然后运行即可**。 同理 其他任务则运行其他文件，例如问答训练是run_squad.py等

## 1.2. 数据处理
我们一共需要两种数据
- 数据是BERT开放的预训练模型
- 我们的数据集
### 1.2.1预训练集合
这是google花费大量**资源训练出来的预训练模型**， 我的数据集是英文句子， 所以我们使用 **BERT-base，uncased** 
也就是基础版本+有大小写版本， 当然如果你用中文，google也提供了相应的数据集

下载地址如下： [BERT-Base, Uncased下载地址](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
其他的可以在仓库的readme里找到相应的下载地址
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517011805279.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxODc0NDU1OTUz,size_16,color_FFFFFF,t_70) 
# 1.2.2 数据集处理
这次使用的是一个10000条数据的语句情感多分类问题，**数据集我已经连通代码一并上传了**，所以不运行以下的处理也是可以的，原数据包括了2个文件， 其中 
- data.csv是有正确标签的数据集， 我们把这个数据集分成2个部分：训练集， 测试集，开发集， 比例为8 : 1 ：1， 我们使用**pandas**来进行数据划分
- prediction_example.csv 是只带句子的用于预测的数据集

代码如下， 数据都是存放在glue文件夹下
**data_cut_off.py**
```py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def train_valid_test_split(x_data, y_data,
        validation_size=0.1, test_size=0.1, shuffle=True):
    x_, x_test, y_, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)
    valid_size = validation_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=valid_size, shuffle=shuffle)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == '__main__':
    path = "glue/"
    pd_all = pd.read_csv(os.path.join(path, "data.csv") )
    pd_all = shuffle(pd_all)
    x_data, y_data = pd_all.text, pd_all.classtype

    x_train, x_valid, x_test, y_train, y_valid, y_test = \
            train_valid_test_split(x_data, y_data, 0.1, 0.1)

    train = pd.DataFrame({'label':y_train, 'x_train': x_train})
    train.to_csv("glue/train.csv", index=False, sep=',')
    valid = pd.DataFrame({'label':y_valid, 'x_valid': x_valid})
    valid.to_csv("glue/dev.csv", index=False, sep=',')
    test = pd.DataFrame({'label':y_test, 'x_test': x_test})
    test.to_csv("glue/test.csv", index=False, sep=',')
```



# 2. 修改代码

因为这次是分类问题， 所以我们需要修改run_classify.py


## 2.1  加入新的处理类

因为我们是做一个分类的任务， 里面自带4个任务的处理类， 其中ColaProcessor是单句分类，和我们的任务较为相近， **所以我们模仿这个类，写一个自己的处理类。**

```py

class EmloProcessor(DataProcessor):
  """Processor for the Emotion data set ."""
	
  def _read_csv(cls, input_file, quotechar=None):
    """读取数据的函数， 这里是读取csv， 因为默认类DataProcessor函数是以 /t作为分隔符，但是
	我们读的csv是以 ‘，’为分隔符， 所以需要重载一下这个函数
	"""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=",", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def get_train_examples(self, data_dir):
    """定义开发集的数据是什么，data_dir会作为参数传进去， 这里就是加上你的文件名即可 """
    return self._create_examples(
        self._read_csv(os.path.join(data_dir, "train.csv"), ), "train")

  def get_dev_examples(self, data_dir):
    """定义开发集的数据是什么，data_dir会作为参数传进去， 这里就是加上你的文件名即可"""
    return self._create_examples(
        self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

  def get_test_examples(self, data_dir):
    """定义测试集的数据是什么"""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

  def get_labels(self):
    """ 这里是显示你一共有几个分类标签， 在此任务中我有3个标签，如实写上  标签值和 csv里面存的值相同 """
    return ["neutral", "positive", "negative"]

  def _create_examples(self, lines, set_type):
    """这个函数是用来把数据处理， 把每一个例子分成3个部分，填入到InputExample的3个参数
    text_a 是 第一个句子的文本
    text_b 是 第二个句子的文本 但是由于此任务是单句分类， 所以 这里传入为None
    guid 是一个二元组  第一个表示此数据是什么数据集类型（train dev test） 第二个表示数据标号
    label 表示句子类别
	"""
    examples = []
    for (i, line) in enumerate(lines):
      # 去掉表头
      if i == 0: continue
      # 获取guid 
      guid = "%s-%s" % (set_type, i)
      #print(line, i)
      # 获取text  因为我们的集合都是只有两列  第一列类别 第二列 文本句子  所以不用分不同数据集填写
      text_a = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      # 加入样本
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

```

有兴趣的盆友可以看看其他的3个类。


## 2.2 处理类注册

同样我们需要在主函数里把我们的类**当做参数选项**，给他加个选项， 也就是**当参数填emlo时，使用的数据处理类是我们自己写的处理类**
```py
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "emlo": EmloProcessor
  }

```



# 3. 运行代码

运行代码需要提供参数， 这里我建议**直接在pycharm编译器里加参数**，或者直接命令行运行参数， 而不用按照官方教材  run  xxx.sh

这里我给出我的编译参数， 如果你运行不了， 建议 改小max_seq_length， train_batch_size,
```shell
python
run_classifier.py
--task_name=emlo
--do_train=true
--do_eval=true
--data_dir=./glue
--vocab_file=./uncased/uncased_L-12_H-768_A-12/vocab.txt
--bert_config_file=./uncased/uncased_L-12_H-768_A-12/bert_config.json
--init_checkpoint=./uncased/uncased_L-12_H-768_A-12/bert_model.ckpt
--max_seq_length=128
--train_batch_size=32
--learning_rate=2e-5
--num_train_epochs=3.0
--output_dir=./tmp/emotion/
```

- task_name 表示我调用的是什么处理类 ，这里我们是用我们新的类所以选 emlo
- 文件dir 可以自己定义， 如果无定义到会出错， 我这里是有3个文件夹 uncased里面放预训练模型， glue放数据，tmp/emotion里面放结果

**训练结果如下：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517014108743.png)


# 4. 分类预测






# 5. 运行问题

## 5.1 出现内存不够

官方解释影响内存大小的参数
The factors that affect memory usage are:

- **`max_seq_length`**: The released models were trained with sequence lengths
  up to 512, but you can fine-tune with a shorter max sequence length to save
  substantial memory. This is controlled by the `max_seq_length` flag in our
  example code.
- **`train_batch_size`**: The memory usage is also directly proportional to
  the batch size.
- **Model type, `BERT-Base` vs. `BERT-Large`**: The `BERT-Large` model
  requires significantly more memory than `BERT-Base`.
- **Optimizer**: The default optimizer for BERT is Adam, which requires a lot
  of extra memory to store the `m` and `v` vectors. Switching to a more memory
  efficient optimizer can reduce memory usage, but can also affect the
  results. We have not experimented with other optimizers for fine-tuning.

Using the default training scripts (`run_classifier.py` and `run_squad.py`), we
benchmarked the maximum batch size on single Titan X GPU (12GB RAM) with
TensorFlow 1.11.0:

| System       | Seq Length | Max Batch Size |
| ------------ | ---------- | -------------- |
| `BERT-Base`  | 64         | 64             |
| ...          | 128        | 32             |
| ...          | 256        | 16             |
| ...          | 320        | 14             |
| ...          | 384        | 12             |
| ...          | 512        | 6              |
| `BERT-Large` | 64         | 12             |
| ...          | 128        | 6              |
| ...          | 256        | 2              |
| ...          | 320        | 1              |
| ...          | 384        | 0              |
| ...          | 512        | 0              |

- 对于参数max_seq_length， train_batch_size 越小， 内存使用越小
- 对于使用的预训练集 BERT-Base  使用内存比 BERT-Large小
- 使用不同的优化器也会造成一定的影响

作者给出了一些他使用的12g显存的搭配， 可以参考使用


# 6. 源码 GITHUB 地址
特意直接开放源码， 供大家参考

https://github.com/wangjiwu/BERT-emotion-classification


