# DiPE

The Pytorch code for our following paper

> **DiPE: Deeper into Photometric Errors for Unsupervised Learning of Depth and Ego-motion from Monocular Videos**
>
> [Hualie Jiang](https://hualie.github.io/), Laiyan Ding, Zhenglong Sun and Rui Huang
>
> [IROS 2020 (pdf)](http://ras.papercept.net/images/temp/IROS/files/0845.pdf)



# Preparation

#### Installation

Install pytorch first by running

```bash
conda install pytorch=0.4.1 torchvision=0.2.1  cuda100 -c pytorch
```

Then install other requirements

```bash
pip install -r requirements.txt
```

#### Dataset 

Please download and preprocess the KITTI dataset as [Monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) does. 



# Training 

### Monocular Depth

#### Eigen Zhou Split

```
python train.py --data_path $DATA_PATH(raw_data) --model_name dipe_eigen --split eigen_zhou --dataset kitti 
```

#### Official Benchmark Split

```
python train.py --data_path $DATA_PATH(raw_data) --model_name dipe_bench --split benchmark --dataset kitti 
```

### Monocular Odometry

#### Pair-frames 

```
python train.py --data_path $DATA_PATH(odometry) --model_name dipe_odom2 --split odom --dataset kitti_odom \ 
--frame_ids 0 -1 1 --pose_model_input pairs
```

#### 3-frames 

```
python train.py --data_path $DATA_PATH(odometry) --model_name dipe_odom3 --split odom --dataset kitti_odom \
--frame_ids 0 -1 1 --pose_model_input all
```

#### 5-frames 

```
python train.py --data_path $DATA_PATH(odometry) --model_name dipe_odom5 --split odom --dataset kitti_odom \
--frame_ids 0 -2 -1 1 2 --pose_model_input pair --disable_occlusion_mask_from_photometric_error
```



# Evaluation  

The pretrained models of our paper is available on [BaiduDisk](https://pan.baidu.com/s/1gaIU0s8CibAb4pv_pJaYjQ) (code:4n8z). 

### Monocular Depth

#### Eigen Split

```
python evaluate_kitti.py --data_path $DATA_PATH(raw_data) --load_weights_folder $MODEL_PATH(dipe_eigen) \ 
--eval_mono --eval_split eigen
```

#### Official Benchmark Split

```
python evaluate_kitti.py --data_path $DATA_PATH(depth) --load_weights_folder $MODEL_PATH(dipe_bench) --dataset kitti_depth \ 
--eval_mono --eval_split benchmark
```

#### Results

|   Split   | Abs Rel | Sq Rel | RMSE  | RMSE_log |  a1   |  a2   |  a3   |
| :-------: | :-----: | :----: | :---: | :------: | :---: | :---: | :---: |
|   Eigen   |  0.112  | 0.875  | 4.795 |  0.190   | 0.880 | 0.960 | 0.981 |
| Benchmark |  0.086  | 0.556  | 3.923 |  0.133   | 0.928 | 0.983 | 0.994 |



### Monocular Odometry

#### Divide the poses for different tests

```
python ./splits/odom/kitti_divide_poses.py --data_path $DATA_PATH(odometry) 
```

#### 2-pair-frames 

```
python evaluate_kitti.py --data_path $DATA_PATH(odometry) --load_weights_folder $MODEL_PATH(dipe_odom2) \ 
--dataset kitti_odom --pose_model_input pairs --eval_split odom_09 
```

#### 3-all-frames 

```
python evaluate_kitti.py --data_path $DATA_PATH(odometry) --load_weights_folder $MODEL_PATH(dipe_odom3) \ 
--dataset kitti_odom --frame_ids 0 -1 1 --pose_model_input all --eval_split odom_09 
```

#### 5-all-frames 

```
python evaluate_kitti.py --data_path $DATA_PATH(odometry) --load_weights_folder $MODEL_PATH(dipe_odom5) \
--dataset kitti_odom --frame_ids 0 -2 -1 1 2 --pose_model_input all --eval_split odom_09 
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

Please cite our paper if you find our work useful in your research.

```
@inproceedings{jiang2020dipe,
  title={DiPE: Deeper into Photometric Errors for Unsupervised Learning of Depth and Ego-motion from Monocular Videos},
  author={Jiang, Hualie and Ding, Laiyan and Sun, Zhenglong and Huang, Rui},
  booktitle={In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```
