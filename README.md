# Dive deep into regions: Exploiting regional information transformer for single image deraining

[Baiang Li](https://ztmotalee.github.io), [Zhao Zhang](https://sites.google.com/site/cszzhang), [Huan Zheng](), [Xiaogang Xu](https://xiaogang00.github.io), [Yanyan Wei](http://faculty.hfut.edu.cn/weiyanyan/en/index.htm), [Jingyi Zhang](), [Jicong Fan]() and [Meng Wang]()

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()
[![paper](https://img.shields.io/badge/paper-camera%20ready-orange)]()
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)]()

<hr />

> **Abstract:** *Transformer-based Single Image Deraining (SID) methods have achieved remarkable success, primarily attributed to their robust capability in capturing long-range interactions. However, we've noticed that current methods handle rain-affected and unaffected regions concurrently, overlooking the disparities between these areas, resulting in confusion between rain streaks and background parts, and inabilities to obtain effective interactions, ultimately resulting in suboptimal deraining outcomes. To address the above issue, we introduce the Region Transformer (Regformer), a novel SID method that underlines the importance of independently processing rain-affected and unaffected regions while considering their combined impact for high-quality image reconstruction. The crux of our method is the innovative Region Transformer Block (RTB), which integrates a Region Masked Attention (RMA) mechanism and a Mixed Gate Forward Block (MGFB). Our RTB is used for attention selection of rain-affected and unaffected regions and local modeling of mixed scales. The RMA generates attention maps tailored to these two regions and their interactions, enabling our model to capture comprehensive features essential for rain removal. To better recover high-frequency textures and capture more local details, we develop the MGFB as a compensation module to complete local mixed scale modeling. Extensive experiments demonstrate that our model reaches state-of-the-art performance, significantly improving the image deraining quality.*
<hr />

## Network Architecture

<img src = "figs/Regformer.pdf">

## Datasets
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
  </tr>
</tbody>
</table>
Here, these datasets we provided are fully paired images, especially SPA-Data. 

## Training
```
cd Regformer
bash train.sh
```
Run the script then you can find the generated experimental logs in the folder `experiments`.

## Testing
1. Please download the corresponding testing datasets and put them in the folder `test/input`. Download the corresponding pre-trained models and put them in the folder `pretrained_models`.
2. Note that we do not use MEFC for training Rain200L and SPA-Data, because their rain streaks are less complex and easier to learn. Please modify the file `DRSformer_arch.py`.
3. Follow the instructions below to begin testing our model.
```
python test.py --task Deraining --input_dir './test/input/' --result_dir './test/output/'
```
Run the script then you can find the output visual results in the folder `test/output/Deraining`.

## Pre-trained Models
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
  </tr>
</tbody>
<tbody>
  <tr>
    <td>Google Drive</td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
    <td> <a href="">Download</a> </td>
  </tr>
</tbody>
</table>

## Performance Evaluation
<img src = "figs/table.png">

## Visual Deraining Results
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Rain200L</th>
    <th>Rain200H</th>
    <th>DID-Data</th>
    <th>DDN-Data</th>
    <th>SPA-Data</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DualGCN</td>
    <td> <a href="https://pan.baidu.com/s/1o9eLMv7Zfk_GC9F4eWC2kw?pwd=v8qy">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1QiKh5fTV-QSdnwMsZdDe9Q?pwd=jnc9">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1Wh7eJdOwXPABz5aOBPDHaA?pwd=3gdx">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1ML1A1boxwX38TGccTzr6KA?pwd=1mdx">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/16RHVyrBoPnOhW1QuglRmlw?pwd=lkeb">Download</a> </td>
  </tr>
  <tr>
    <td>SPDNet</td>
    <td> <a href="https://pan.baidu.com/s/1u9F4IxA8GCxKGk6__W81Og?pwd=y39h">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1wSTwW6ewBUgNLj7l7i6HzQ?pwd=mry2">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1z3b60LHOyi8MLcn8fdNc8A?pwd=klci">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/130e74ISgZtlaw8w6ZzJgvQ?pwd=19bm">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1J0ybwnuT__ZGQZNbMTfw8Q?pwd=dd98">Download</a> </td>
  </tr>
  <tr>
    <td>Uformer</td>
    <td> - </td>
    <td> - </td>
    <td> <a href="https://pan.baidu.com/s/1fWLjSCSaewz1QXdddkpkIw?pwd=4uur">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1cWY7piDJRF05qKYPNXt_cA?pwd=39bj">Download</a> </td>
    <td> - </td>
  </tr>
  <tr>
    <td>Restormer</td>
    <td> <a href="https://pan.baidu.com/s/1jv6PUMO7h_Tc4ovrCLQsSw?pwd=6a2z">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/16R0YamX-mfn6j9sYP7QpvA?pwd=9m1r">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1b8lrKE82wgM8RiYaMI6ZQA?pwd=1hql">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1GGqsfUOdoxod9vAUxB54PA?pwd=crj4">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1IG4T1Bz--FrDAuV6o-fykA?pwd=b40z">Download</a> </td>
  </tr>
  <tr>
    <td>IDT</td>
    <td> <a href="https://pan.baidu.com/s/1jhHCHT64aDknc4g0ELZJGA?pwd=v4yd">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/10TZzZH0HisPV0Mw-E4SlTQ?pwd=77i4">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1svMZAUvs6P6RRNGyCTaeAA?pwd=8uxx">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1FSf3-9HEIQ-lLGRWesyszQ?pwd=0ey6">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/16hfo5VeUhzu6NYdcgf7-bg?pwd=b862">Download</a> </td>
  </tr>
  <tr>
    <td>Ours</td>
    <td> <a href="https://pan.baidu.com/s/1-ElpyJigVnpt5xDFE6Pqqw?pwd=hyuv">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/13aJKxH7V_6CIAynbkHXIyQ?pwd=px2j">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1Xl3q05rZYmNEtQp5eLTTKw?pwd=t879">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1D36Z0cEVPPbm5NljV-8yoA?pwd=9vtz">Download</a> </td>
    <td> <a href="https://pan.baidu.com/s/1Rc36xXlfaIyx3s2gqUg_Bg?pwd=bl4n">Download</a> </td>
  </tr>
</tbody>
</table>

For DualGCN, SPDNet, Restormer and IDT, we retrain their models provided by the authors if no pretrained models are provided, otherwise we evaluate them with their online codes. For Uformer, we refer to some reported results in [IDT](https://github.com/jiexiaou/IDT). Noted that since the PSNR/SIMM codes used to test DID-Data and DDN-Data in their paper are different from ours, we retrain the Uformer on the DID-Data and DDN-Data. For other previous methods, we refer to reported results in [here](https://ieeexplore.ieee.org/document/10035447/) with same PSNR/SIMM codes.


## Citation
If you are interested in this work, please consider citing:


## Acknowledgment
This code is based on the [Restormer](https://github.com/swz30/Restormer). Thanks for their awesome work.

## Contact
Should you have any question or suggestion, please contact chenxiang@njust.edu.cn.
