# DeepGene: An Efficient Foundation Model for Genomics based on Pan-genome Graph Transformer
We introduce DeepGene, a model leveraging Pan-genome and Minigraph representations to encompass the broad diversity of genetic language. DeepGene employs the rotary position embedding to improve the length extrapolation in various genetic analysis tasks. On the 28 tasks in Genome Understanding Evaluation, DeepGene reaches the top position in 9 tasks, second in 5, and achieves the overall best score. DeepGene outperforms other cutting-edge models for its compact model size and its superior efficiency in processing sequences of varying lengths.

Preprint: https://www.biorxiv.org/content/10.1101/2024.04.24.590879v1

## 1. Environment setup
Please see ```PanGeneGraphTrans/requirements.txt```.
## 2. Pan-genome Dataset
### 2.1 Download data
Download [Minigraph file (.rgfa)](https://drive.google.com/file/d/1x7vSy7BTISx3K0su6FhdHLTmv1LPgH1B/view?usp=drive_link) and place it in the ```dataPretreatment``` folder.
### 2.2 Data processing
Please see ```dataPretreatment``` and ```PanGeneGraphTrans/dataset.py```.
## 3. Model Pre-training
Please see ```PanGeneGraphTrans/pretrain.py```.
## 4. Model Fine-tuning
### 4.1 Download pre-trained model
Download [pretrained model](https://drive.google.com/drive/folders/1gb2IqO3NdSMbydKMLGZBFAbsbKhgm0i8?usp=drive_link).

### 4.2 Fine-tune with pre-trained model
Please see ```PanGeneGraphTrans/finetune.py```.