image list��ʽ
��һ����movingͼ��
�ڶ�����fixedͼ��
������
�Դ�����

Ŀǰdataloader����ͼ��϶࣬�����Ԥ��������Լ���Ҫ��һ�£�
loss����Ҳ��̫�ʺϣ�

���÷�����
train��
	python train_mynet2_v2.py -cfg config/config2_v2.yaml 
validation��
	CUDA_VISIBLE_DEVICES=0 python val_mynetwork2.py -m mynetwork2_v2_air/network_1270.pth

PVP 1-3 step 4
UCP 0.5 0.5 6 0.5-2 step 4
EAP gaussian 2