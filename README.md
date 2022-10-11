# PhyDIR
Pytorch implementation of Physically-Guided Disentangled Implicit Rendering for 3D Face Modeling (2022 CVPR)

## 💻 Setup
```
env: nsml v2
gpu: V100-1
```
1. Environment setup
    - set up your environment following the [official guide](https://github.com/elliottwu/unsup3d)
    - OR use `nvcr.io.nvidia.cuda:phydir_v1` image (If you request, I will deliver the image.)
2. Dataset setup (use symbolic link)
```
mkdir data
ln -s /mnt/video-nfs5/users/solang/src/phydir/data/ data/
```
3. Training
    - change `phydir path`, `conda path` and `config name`
```
cd [phydir] && \
source /home/nsml/anaconda3/etc/profile.d/conda.sh && \
conda activate unsup3d && \
python run.py --config configs/[cfg_name].yml --gpu 0 --num_workers 4
```

## 📁 file structure
```
--------------------
├── data
│   ├── celeba
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   │   ├── datalist ([id].json)
│   │   │   ├── *.jpg
|   ├── casia
|   ├── celebamask_hq
├── configs
│   ├── *.yml
├── phydir
│   ├── datasets
|   ├── models
|   ├── *.py
├── results
├── run.py 
--------------------
```
---
**References**
- [Physically-guided Disentangled Implicit Rendering for 3D Face Modeling](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Physically-Guided_Disentangled_Implicit_Rendering_for_3D_Face_Modeling_CVPR_2022_paper.pdf)
- [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://arxiv.org/pdf/1911.11130.pdf) [code](https://github.com/elliottwu/unsup3d)
- [Learning to Aggregate and Personalize 3D Face from In-the-Wild Photo Collection](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_To_Aggregate_and_Personalize_3D_Face_From_In-the-Wild_Photo_CVPR_2021_paper.pdf)