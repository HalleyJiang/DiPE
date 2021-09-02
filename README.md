# DiPE

The Pytorch code for our following papers

> **DiPE: Deeper into Photometric Errors for Unsupervised Learning of Depth and Ego-motion from Monocular Videos**, [IROS 2020 (pdf)](http://ras.papercept.net/images/temp/IROS/files/0845.pdf)
>
> **Unsupervised Monocular Depth Perception: Focusing on Moving Objects**, accepted by IEEE Sensors Journal
>
> [Hualie Jiang](https://hualie.github.io/), Laiyan Ding, Zhenglong Sun and Rui Huang
>
> 

# Preparation

#### Installation

Install pytorch first by running

```bash
conda install pytorch=1.0.0 torchvision=0.2.1  cuda100 -c pytorch
```



Then install other requirements

```bash
pip install -r requirements.txt
```

#### Datasets 

Please download and preprocess the KITTI dataset as [Monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) does. 

For the [Cityscapes](https://www.cityscapes-dataset.com/downloads/) dataset, one should download *gtFine_trainvaltest.zip*, *leftimg8bit_sequence_trainvaltest.zip*, and *disparity_trainvaltest.zip* and extract them to one directory. 



# Training 

### Monocular Depth

#### KITTI Eigen Zhou Split

```
python train.py --data_path $DATA_PATH(raw_data) --model_name dipe_eigen --split kitti_eigen_zhou --dataset kitti 
```

#### KITTI Official Benchmark Split

```
python train.py --data_path $DATA_PATH(raw_data) --model_name dipe_bench --split kitti_benchmark --dataset kitti 
```

#### Unsupervised Monocular Depth Perception: Focusing on Moving ObjectsCityscapes

```
python train.py --data_path $DATA_PATH(cityscapes) --model_name cityscapes --split cityscapes --dataset cityscapes --png --frame_ids 0 -2 2 
```

### Monocular Odometry

#### Pair-frames 

```
python train.py --data_path $DATA_PATH(odometry) --model_name dipe_odom2 --split kitti_odom --dataset kitti_odom \ 
--frame_ids 0 -1 1 --pose_model_input pairs
```

#### 3-frames 

```
python train.py --data_path $DATA_PATH(odometry) --model_name dipe_odom3 --split kitti_odom --dataset kitti_odom \
--frame_ids 0 -1 1 --pose_model_input all
```

#### 5-frames 

```
python train.py --data_path $DATA_PATH(odometry) --model_name dipe_odom5 --split kitti_odom --dataset kitti_odom \
--frame_ids 0 -2 -1 1 2 --pose_model_input all --disable_occlusion_mask_from_photometric_error
```



# Evaluation  

The pretrained models of our paper is available on [GoogleDrive](https://drive.google.com/file/d/15HLlcknpEZ87WH1rdPHfe9c5KnyFAI4S/view?usp=sharing) and [BaiduDisk](https://pan.baidu.com/s/17j4J4A8S4zgy836O5v3LDQ) (code:rq3a). 

### Monocular Depth

#### KITTI Eigen Split

```
python evaluate_kitti.py --data_path $DATA_PATH(raw_data) --load_weights_folder $MODEL_PATH(dipe_eigen) \ 
--eval_mono --eval_split kitti_eigen
```

#### KITTI Official Benchmark Split

```
python evaluate_kitti.py --data_path $DATA_PATH(depth) --load_weights_folder $MODEL_PATH(dipe_bench) --dataset kitti_depth \ 
--eval_mono --eval_split kitti_benchmark
```

#### Results

|   Split   | Abs Rel | Sq Rel | RMSE  | RMSE_log |  a1   |  a2   |  a3   |
| :-------: | :-----: | :----: | :---: | :------: | :---: | :---: | :---: |
|   Eigen   |  0.112  | 0.875  | 4.795 |  0.190   | 0.880 | 0.960 | 0.981 |
| Benchmark |  0.086  | 0.556  | 3.923 |  0.133   | 0.928 | 0.983 | 0.994 |



#### KITTI Eigen Split with Background and Moving Objects Separately

```
python train.py  evaluate_kitti_eigen_moving_objects.py --data_path $DATA_PATH(raw_data) --load_weights_folder $MODEL_PATH(dipe_eigen) \ 
--eval_mono --eval_split kitti_eigen
```

#### Results

|     Region      | Abs Rel | Sq Rel | RMSE  | RMSE_log |  a1   |  a2   |  a3   |
| :-------------: | :-----: | :----: | :---: | :------: | :---: | :---: | :---: |
|   Background    |  0.107  | 0.784  | 4.614 |  0.180   | 0.886 | 0.964 | 0.984 |
| Dynamic Objects |  0.215  | 3.083  | 7.172 |  0.319   | 0.737 | 0.883 | 0.931 |



#### CityScapes with Background and Objects Separately

```
python train.py  evaluate_cityscapes.py --data_path $DATA_PATH(cityscapes) --load_weights_folder $MODEL_PATH(dipe_cityscapes) --eval_mono
```

#### Results

|   Region   | Abs Rel | Sq Rel | RMSE  | RMSE_log |  a1   |  a2   |  a3   |
| :--------: | :-----: | :----: | :---: | :------: | :---: | :---: | :---: |
| Background |  0.155  | 2.381  | 8.127 |  0.220   | 0.808 | 0.947 | 0.980 |
|  Objects   |  0.365  | 13.401 | 9.742 |  0.336   | 0.697 | 0.861 | 0.924 |



### Monocular Odometry

#### Divide the poses for different tests

```
python ./splits/odom/kitti_divide_poses.py --data_path $DATA_PATH(odometry) 
```

#### 2-pair-frames 

```
python evaluate_kitti.py --data_path $DATA_PATH(odometry) --load_weights_folder $MODEL_PATH(dipe_odom2) \ 
--dataset kitti_odom --pose_model_input pairs --eval_split kitti_odom_09 
```

#### 3-all-frames 

```
python evaluate_kitti.py --data_path $DATA_PATH(odometry) --load_weights_folder $MODEL_PATH(dipe_odom3) \ 
--dataset kitti_odom --frame_ids 0 -1 1 --pose_model_input all --eval_split kitti_odom_09 
```

#### 5-all-frames 

```
python evaluate_kitti.py --data_path $DATA_PATH(odometry) --load_weights_folder $MODEL_PATH(dipe_odom5) \
--dataset kitti_odom --frame_ids 0 -2 -1 1 2 --pose_model_input all --eval_split kitti_odom_09 
```

To test sequence 10, we need to set ```--eval_split odom_10```.

#### Results

<table>  
    <tr>
        <th rowspan="2">#frames</th>
        <th colspan="2">Sequence 09</th>
        <th colspan="2">Sequence 10</th>
    </tr>
    <tr>
        <td>ATE</td> 
        <td>RE</td>
        <td>ATE</td> 
        <td>RE</td>
    </tr> 
    <tr>
        <td>2</td>
        <td>0.0125 &plusmn 0.0055</td> 
        <td>0.0023 &plusmn 0.0011</td>
        <td>0.0122 &plusmn 0.0081</td> 
        <td>0.0027 &plusmn 0.0019</td>
    </tr> 
    <tr>
        <td>3</td>
        <td>0.0122 &plusmn 0.0057</td> 
        <td>0.0025 &plusmn 0.0012</td>
        <td>0.0120 &plusmn 0.0082</td> 
        <td>0.0029 &plusmn 0.0020</td>
    </tr> 
    <tr>
        <td>5</td>
        <td>0.0120 &plusmn 0.0057</td> 
        <td>0.0026 &plusmn 0.0014</td>
        <td>0.0118 &plusmn 0.0082</td> 
        <td>0.0030 &plusmn 0.0022</td>
    </tr> 
</table>





## Acknowledgements

The project is built upon [Monodepth2](https://github.com/nianticlabs/monodepth2). We thank Monodepth2's authors for their excellent work and repository. 

## Citation

Please cite our papers if you find our work useful in your research.

```
@inproceedings{jiang2020dipe,
  title={DiPE: Deeper into Photometric Errors for Unsupervised Learning of Depth and Ego-motion from Monocular Videos},
  author={Jiang, Hualie and Ding, Laiyan and Sun, Zhenglong and Huang, Rui},
  booktitle={In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
@article{jiang2021unsupervised,
  title={Unsupervised Monocular Depth Perception: Focusing on Moving Objects},
  author={Jiang, Hualie and Ding, Laiyan and Sun, Zhenglong and Huang, Rui},
  journal={arXiv preprint},
  year={2021}
}
```
