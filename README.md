# PhyDIR
Pytorch implementation of Physically-Guided Disentangled Implicit Rendering for 3D Face Modeling (2022 CVPR)

### file structure
```
--------------------
├── data
│   ├── celeba_cropped
│   │   ├── train
│   │   │   ├── datalist ([id].json)
│   │   │   ├── *.jpg
│   │   ├── val
│   │   ├── test
|   ├── CASIA-WebFace (todo)
├── configs
│   ├── *.yml
├── phydir
│   ├── datasets
|   ├── models
|   ├── *.py
├── run.py 
--------------------
```

### Prior Works
- [Physically-guided Disentangled Implicit Rendering for 3D Face Modeling](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Physically-Guided_Disentangled_Implicit_Rendering_for_3D_Face_Modeling_CVPR_2022_paper.pdf)
- [Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild](https://arxiv.org/pdf/1911.11130.pdf) [code](https://github.com/elliottwu/unsup3d)
- [Learning to Aggregate and Personalize 3D Face from In-the-Wild Photo Collection](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Learning_To_Aggregate_and_Personalize_3D_Face_From_In-the-Wild_Photo_CVPR_2021_paper.pdf)