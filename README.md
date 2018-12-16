## 1. 训练过程
### 1.1 下载数据
下载Wider_face，解压后放入data目录；
 ###  1.2 训练PNet

 - 运行gen_pnet_widerface.py生成pnet训练数据
 - 运行get_label_list.py生成文件列表
 - 运行convert_data_hdf5.py生成hdf5文件
 - 运行train_pnet.sh开始训练PNet，训练完成后将相应模型放入model文件夹

 ###  1.3 训练ONet
训练ONet的过程和PNet基本一样：
 - 运行gen_onet_widerface.py生成pnet训练数据
 - 运行get_label_list.py生成文件列表
 - 运行convert_data_hdf5.py生成hdf5文件
 - 运行train_onet.sh开始训练ONet，训练完成后将相应模型放入model文件夹
    
## 2. 测试过程      
运行demo.p y

## 3. 最后
因为时间关系，这个代码仅仅训练了PNet和ONet，RNet和关键点的训练，以后有时间我也训练下。**欢迎对人工智能有兴趣的朋友，扫一扫下面的二维码，添加我的微信(微信号：tuge7893)，跟我交流**
<img src="https://img-blog.csdnimg.cn/20181216165408677.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzI0MTQzOTMx,size_16,color_FFFFFF,t_70 "   width = 300 height = 300 div align=left/>
