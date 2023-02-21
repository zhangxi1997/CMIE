## Environment

Anaconda 4.8.4, python 3.6.8, pytorch 1.6 and cuda 10.2. For other libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n videoqa python==3.6.8
>conda activate videoqa
>pip install -r requirements.txt #may take some time to install
```
## Data Preparation
Please download the pre-computed features and QA annotations from [here](https://drive.google.com/drive/folders/1gKRR2es8-gRTyP25CvrrVtV6aN5UxttF?usp=sharing). There are 4 zip files: 
- ```['vid_feat.zip']```: Appearance and motion feature for video representation. (With code provided by [HCRN](https://github.com/thaolmk54/hcrn-videoqa)).
- ```['qas_bert.zip']```: Finetuned BERT feature for QA-pair representation. (Based on [pytorch-pretrained-BERT](https://github.com/LuoweiZhou/pytorch-pretrained-BERT/)).
- ```['nextqa.zip']```: Annotations of QAs and GloVe Embeddings. 
- ```['models.zip']```: HGA model. 

After downloading the data, please create a folder ```['data/feats']``` at the same directory as ```['NExT-QA']```, then unzip the video and QA features into it. You will have directories like ```['data/feats/vid_feat/', 'data/feats/qas_bert/' and 'NExT-QA/']``` in your workspace. Please unzip the files in ```['nextqa.zip']``` into ```['NExT-QA/dataset/nextqa']``` and ```['models.zip']``` into ```['NExT-QA/models/']```. 

*(You are also encouraged to design your own pre-computed video features. In that case, please download the raw videos from [VidOR](https://xdshang.github.io/docs/vidor.html). As NExT-QA's videos are sourced from VidOR, you can easily link the QA annotations with the corresponding videos according to the key 'video' in the ```['nextqa/.csv']``` files, during which you may need the map file ```['nextqa/map_vid_vidorID.json']```)*.


## Usage
Once the data is ready, you can easily run the code. First, to test the environment and code, we provide the prediction and model of the SOTA approach (i.e., HGA) on NExT-QA. 
You can obtain the prediction by running: 
```
>./main.sh 0 val #Test the model with GPU id 0
```
The command above will load the model under ['videoqa_saves/'] and generate the prediction file.
If you want to train the model, please run
```
>./main.sh 0 train # Train the model with GPU id 0
```

## Acknowledgement
We refer to the repo [NExT-QA](https://github.com/doc-doc/NExT-QA/) for baseline re-implementations.
