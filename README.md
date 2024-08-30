# About 
The source code of "Correlating Electrocardiograms with Echocardiographic Parameters in Hemodynamically-Significant Aortic Regurgitation Using Deep Learning"

![Summary_r1](https://github.com/user-attachments/assets/d69cd1ae-5163-40e5-be92-a6b295a0e6ee)



# Files
The dataset is composed of a mixtutre of Japanese and Taiwanese data
* Model training and validation: main.py  
* Dataset preperation: dataset.py  
* Testing code: test.py  
* Shapley value calculation and visualization: SHAP.py  
  
We also provide the version of code to train the model only on Taiwanese dataset and finetune on Japanese dataset
* Please refer to the transfer_learning.sh


# Models
* Base: models weights trained solely on Taiwanese dataset
* MixedData: models trained on Taiwanese+Japanese dataset
* Finetuned: models finetuned on the Japanese dataset
  
