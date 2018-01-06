import numpy as np
import json

# adapter converter for sql for a MATRIX

def adapt_array(M):
    len_col=len(M[:,0])
    len_row = len(M[0, :])
    num = M.reshape(1, len_col * len_row)               # konvertiert zu 1D array
    num=np.append(num,len_col)
    num = np.append(num, len_row)
    stri = json.dumps(num.tolist())                     # json macht aus np array ein string (zuerst wir np array zu liste konvertiert)
    return stri


def converter_array(ts):
    lis = json.loads(ts)                                # ladet string und macht es zu liste
    lis=np.asarray(lis)
    #np_arr=lis[0:-2].reshape(int(lis[-2]), int(lis[-1]))
    np_arr=np.reshape(lis[0:-2], (int(lis[-2]), int(lis[-1])))
    return np_arr



# adapter converter for sql for a DICT

def adapt_dict(stru):
    stri = json.dumps(stru)                     # json macht aus np array ein string (zuerst wir np array zu liste konvertiert)
    return stri


def converter_dict(stri):
    lis = json.loads(stri)                                # ladet string und macht es zu liste
    return lis


