image list格式
第一行是moving图像
第二行是fixed图像
。。。
以此类推

目前dataloader输入图像较多，输入和预处理根据自己需要改一下，
loss可能也不太适合，

调用方法：
train：
	python train_mynet2_v2.py -cfg config/config2_v2.yaml 
validation：
	CUDA_VISIBLE_DEVICES=0 python val_mynetwork2.py -m mynetwork2_v2_air/network_1270.pth

PVP 1-3 step 4
UCP 0.5 0.5 6 0.5-2 step 4
EAP gaussian 2