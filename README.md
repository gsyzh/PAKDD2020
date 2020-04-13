**天池--PAKDDPAKDD 2020 阿里巴巴智能运维算法大赛** 

给定一段连续采集(天粒度)的硬盘状态监控数据（Self-Monitoring, Analysis, and Reporting Technology; often written as SMART)以及故障标签数据，参赛者需要自己提出方案，按天粒度判断每块硬盘是否会在未来30日内发生故障。例如，可以将预测故障问题转化为传统的二分类问题，通过分类模型来判断哪些硬盘会坏；或者可以转化为排序问题，通过Learning to rank的方式判断硬盘的损坏严重程度等。

初赛会提供训练数据集，供参赛选手训练模型并验证模型效果使用。同时，也将提供测试集，选手需要对测试集中的硬盘按天粒度进行预测，判断该硬盘是否会在未来30天内发生故障，并将模型判断出的结果上传至竞赛平台，平台会根据提交的预测结果，来评估模型预测的效果。

在复赛中，面对进一步的问题和任务，选手需要提交一个docker镜像，镜像中需要包含用来进行故障预测所需的所有内容，也即完整预测处理解决方案脚本。其中，镜像中的预测脚本需要能够根据输入的测试集文件（文件夹）位置，来对测试集中的硬盘故障预测，并把预测结果以指定的CSV文件格式输出到指定位置。

赛题链接及数据集地址：https://tianchi.aliyun.com/competition/entrance/231775/information

**PAKDD**

​	|--**project**  初赛代码

​	|--**projectDocker** 复赛代码

​	|--**README.md**   算法思路

### **初赛思路**：

 **project**  

​	|--**data**     原始数据文件

​	|--**user_data** 	中间处理完的数据

​	|--**feature**    特征工程

​	|--**prediction_result**   预测结果文件

​	|--**code**  模型训练

#### 0、解决方案及算法介绍文件

![image-20200413092201551](https://gitee.com/gsyzh8023/gsyzhimage2/raw/master/天池大规模硬盘预测/image-20200413092201551.png)

### 复赛思路：

**projectDocker**

​	**|--clf.pkl**  模型文件

​	**|--code.py**  预测代码

​	**|--run.sh**	执行文件

​ **|--DockerFile**	Docker	

​	**|--dataprocessing.py**	特征工程及数据预处理

​ **|--model.py**	模型训练

#### 0、解决方案及算法介绍

![image-20200413092120934](https://gitee.com/gsyzh8023/gsyzhimage2/raw/master/天池大规模硬盘预测/image-20200413092120934.png)
