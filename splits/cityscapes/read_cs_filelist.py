# written by Hualie Jiang, PhD at CUHKSZ, jianghualie0@gmail.com

import argparse
import os
import random


def read_cs_filelist():
    parser = argparse.ArgumentParser(description='read cityscapes file list')
    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the CityScapes data',
                        default='/media/vslam/Elements/Cityscapes/')
    parser.add_argument('--train_right_images',
                        help='train with right images too',
                        action="store_true")
    parser.add_argument('--num_edge_frames',
                        type=int,
                        help='the number of edge frames to remove',
                        default=3)
    opt = parser.parse_args()
    
    # read train set
    train_path = os.path.join(opt.data_path, "leftImg8bit_sequence/train")
    train_cities = sorted(os.listdir(train_path))
    
    train_list = []
    
    for city in train_cities:
        image_names = sorted(os.listdir(os.path.join(train_path, city)))
        num_images = len(image_names)
        for i in range(num_images//30): 
                for j in range(opt.num_edge_frames, 30-opt.num_edge_frames):
                    image_name = image_names[30*i+j]
                    splits = image_name.split('_')
                    train_list.append("train/"+city+" "+splits[1]+splits[2]+" "+"l")
                    if opt.train_right_images:
                        train_list.append("train/"+city+" "+splits[1]+splits[2]+" "+"r")
     
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(train_list)
    with open('train_files.txt', 'w') as f:
        for sample in train_list:
            f.writelines(sample+'\n')
    
    # read test set. we treat the original validation set as test set,
    # as it provides the semantic segmenation labels for depth evaluation on objects 
    test_path = os.path.join(opt.data_path, "leftImg8bit_sequence/val")
    test_cities = sorted(os.listdir(test_path))

    test_list = [];
    
    for city in test_cities:
        image_names = sorted(os.listdir(os.path.join(test_path, city)))
        num_images = len(image_names)
        for i in range(num_images//30):                    
            image_name = image_names[30*i+19]    # test samples with gt labels 
            splits = image_name.split('_')
            test_list.append("val/"+city+" "+splits[1]+splits[2]+" "+"l")
    
    with open('test_files.txt', 'w') as f:
        for sample in test_list:
            f.writelines(sample+'\n')


    # read val set. we treat the original test set as validation set,
    # as its semantic segmenation labels are not available
    val_path = os.path.join(opt.data_path, "leftImg8bit_sequence/test")
    val_cities = sorted(os.listdir(val_path))
    
    val_list = []
    
    for city in val_cities:
        image_names = sorted(os.listdir(os.path.join(val_path, city)))
        num_images = len(image_names)
        for i in range(num_images//30):                    
            image_name = image_names[30*i+19]    # use a subset for validation 
            splits = image_name.split('_')
            val_list.append("test/"+city+" "+splits[1]+splits[2]+" "+"l")
    
    with open('val_files.txt', 'w') as f:
        for sample in val_list:
            f.writelines(sample+'\n')


if __name__ == "__main__":
    read_cs_filelist()
