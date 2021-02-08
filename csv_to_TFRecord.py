from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm
import os
import numpy as np
from atpbar import atpbar
from mantichora import mantichora
import json

from TFRecord_config import configs

from dataclasses import dataclass

@dataclass
class To_TFRecord_Config:
    train_or_valid: str
    tile_size : str
    csv_fp : str
    class_indice_fp : str
        
c = To_TFRecord_Config(**configs)

output_dir = f'/home/agsl0905/PDL1_HER2_data/tpu_data/final_lung/tile_{c.tile_size}/{c.train_or_valid}'
print(f"Output dir {output_dir}")

df_data = pd.read_csv(c.csv_fp,index_col=0)
df_data.reset_index(inplace=True,drop=True)
df_data = df_data.sample(frac=1,random_state = 123).reset_index(drop=True) # shuffle df
# Load class indice

f = open(c.class_indice_fp)
class_indice = json.load(f)
one_hot_map = {v:k for k,v in class_indice.items()} # reverse class_indice for one hot encoding

# slicing df into partitions base on cpu count
#permuted_indices = np.random.permutation(len(df_data))
dfs = np.array_split(df_data, np.ceil(len(df_data)/5000)) # each partition has ~2500 images, ~1000MB per tfrecord
dfs = [df.reset_index(drop=True) for df in dfs]
N_PARTITIONS = len(dfs) # each partition has ~2500 images, ~1000MB per tfrecord
print("Number of partitions:", N_PARTITIONS)

def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

def csv_to_TFRecord(df,partition):
    os.makedirs(output_dir,exist_ok=True)
    output_fp = os.path.join(output_dir,f'PDL1_{c.train_or_valid}_data_examples_{df.shape[0]}_{partition}_.tfrec')
    corrupted = []
    writer = tf.io.TFRecordWriter(output_fp)

    for j in atpbar(range(len(df)), name=f'partition {partition} of {N_PARTITIONS}'):
        fp = df.loc[j,'fp']
        bits = tf.io.read_file(fp)
        
        try:
            image = tf.image.decode_png(bits,channels=3)
            image = tf.image.resize(image,[296,296])
            height = image.shape[0]
            width = image.shape[1]
            encoded_image = tf.image.encode_png(image)     
            label = str.encode(df.loc[j,'label'])
            class_num = int(one_hot_map.get(label.decode("utf-8")))
            one_hot_class = np.eye(len(class_indice))[class_num].astype(np.int8)

            example = tf.train.Example(features=tf.train.Features(feature={'image' :  _bytestring_feature([encoded_image.numpy()]), # .numpy() is needed to turn a tensor to byte
                                                                            "class":        _int_feature([class_num]),              # one class in the list
                                                                            "label":         _bytestring_feature([label]),          # fixed length (1) list of strings, the text label
                                                                            "size":          _int_feature([height, width]),         # fixed length (2) list of ints
                                                                            "one_hot_class": _float_feature(one_hot_class.tolist())})) # variable length  list of floats, n=len(CLASSES)



            writer.write(example.SerializeToString())
        except:
            print(f'{fp} is corrupted')
            corrupted.append(fp)
    writer.close()


with mantichora(nworkers=62) as mcore:
    for i in range(len(dfs)):
        mcore.run(csv_to_TFRecord,dfs[i],i)
    results = mcore.returns()
