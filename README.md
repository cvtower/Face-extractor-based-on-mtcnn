# Face-extractor-based-on-mtcnn
A simple face extractor based on mtcnn
MTCNN is one of the best face detection algorithms.Here is inference(detect&&alignment) only for MTCNN face detector on Tensorflow, which is based on https://github.com/cyberfire/tensorflow-mtcnn.
# why commit this repo
In fact i am so familiar with mtcnn since it is published, and have alternative project experience(porting it to android platform using ncnn/opencl) with it.
However, yesterday(2018/5/10)when help workmate to solve a face extracting related issue i found that no simple and clear samples available on github contains the "warp and crop" operation, which is needed for recognition and liveness detection and so on research. So 2 types of crop implement is provided here, guess it will be simple to understand since it is commited by this new python user(used to play with cuda/opencl/c++) here.
Feel free to use it, hope it will be helpful.
# Contact Author: tower.zhangcvtowerzhang@gmail.com 

