
import numpy as np
import json
import os
import os.path as osp
import sys

CAFFE_ROOT = '/home/hjkim/caffe_comp/caffe/'
if osp.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, osp.join(CAFFE_ROOT, 'python'))

import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2

#   Referred codes 
#   ==================
#       http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
#       http://deepdish.io/2015/04/28/creating-lmdb-in-python/


# extracted dataset
lmdb_file2 ="/home/hjkim/hdd/local/sdb-hjkim/lmdb/ilsvrc12_train_256_conv_extract_5"
lmdb_env2 = lmdb.open(lmdb_file2, map_size=int(1e12))
lmdb_txn2 = lmdb_env2.begin(write=True)


# original dataset
lmdb_file ="/home/hjkim/hdd/local/sdb-hjkim/lmdb/ilsvrc12_train_lmdb_256_conv"
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

# max : 1281167
# ---------------
# 10% : 128116 (class 1000) , for each class = 128
# 5%  : 64058  (class 1000) , for each class = 64

data_list = np.zeros([1000])

k = 0
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)

    #if (i%5) :
    if data_list[label] > 63 : # 5% : 63 / 10% : 127
        continue
    else : 

        data_list[label] = data_list[label] + 1
        datum2 = caffe.io.array_to_datum(data, label)
        keystr = '{:0>8d}'.format(k)
        lmdb_txn2.put( keystr, datum2.SerializeToString() )
        lmdb_txn2.commit()
        lmdb_txn2 = lmdb_env2.begin(write=True)
        k += 1

        if k%100 == 0 :
            print(k)

    if np.sum(data_list) == 64000 :
        break

print('last k = ', k)


