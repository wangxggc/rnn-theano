#Sequence to sequence - theano

##使用方法说明
	
	训练：
 		python train.py -t train_data_file1 train_data_file2 dic_file model_save_file pre_trained_file(optional)"
 			train_data_file1, 前一句话，一行一个文本
 			train_data_file2, 后一句话，即回答

 	生成：
 		python train.py -g model_file dic_file post_data_file response_save_file"

 	句子向量化，Embdding Sentence：
 		python train.py -e model_file dic_file sentences_file embdding_save_file"