# Self Supervised Deep Fake Detection

This is an adaptation of this <a href="https://github.com/AndrewAtanov/simclr-pytorch"> SimCLR Pytorch repo </a>
to the problem of DeepFake Detection on the FaceForensics++ dataset. <br />
We train a SimCLR encoder on the unmanipulated/Real class in the FF++ dataset.
We then assess the resulting latent space on two downstream tasks - supervised linear evaluation 
and a semi-supervised KNN anomaly detection algorithm inspired by
 <a href="https://arxiv.org/abs/2002.10445"> Bergmen et al 2020 </a>.
Our code contributions to the source repo are in the dfd_utils folder, as well as modifications in the models/ssl.py, models/resnet.py, models/encoder.py and train.py to support new dataset loading, refractor the resnet class to remove redundant fc layer, load a torchvision pre-trained resnet and add additional training parameters related to the Deepfake detection task and dataset.
## Enviroment Setup


Create a python enviroment with the provided config file:

```(bash)
conda env create -f environment.yml
conda activate simclr_pytorch

```

## Training and Evaluation
We provide a Jupyter Notebook named Wrapper.ipynb with all training and evaluation procedures.
For more details about running distributed training from command line, see the 
<a href="https://github.com/AndrewAtanov/simclr-pytorch"> original repo. </a>

## Acknowledgements
This work was done as part of the Deep Learning course given by professor Raja Giryes
at the School of Engineering in Tel Aviv University.
