import os
import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
from utils import load_audio, STR_CH_FIRST
import warnings
warnings.filterwarnings("ignore")

def save_resampled_npy(idx):
    mp3_path = os.path.join(MSDPATH, idx)
    save_path = os.path.join(NPYPATH, idx[:-3] + "npy")
    src, sr = load_audio(
            path=mp3_path,
            ch_format=STR_CH_FIRST,
            sample_rate=22050,
            downmix_to_mono=True,
    )  # src: (time)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if os.path.isfile(save_path) == 1:
        print(save_path + '_file_already_extracted!')
    else:
        np.save(save_path, src.astype(np.float32))
    
    
if __name__ == '__main__':
    df_zeroshot = pd.read_csv("tag_binary.csv", index_col=0)
    df_tagging = pd.read_csv("tag_binary.csv", index_col=0)
    id_to_path = pickle.load(open("../media/bach2/seungheon/MSD_split/7D_id_to_path.pkl",'rb'))
    MSD_id_to_7D_id = pickle.load(open("../media/bach2/seungheon/MSD_split/MSD_id_to_7D_id.pkl",'rb'))

    msdids = list(set(list(df_zeroshot.index) + list(df_tagging.index)))
    print("Start Extract!", len(msdids))

    msd_paths = [id_to_path[MSD_id_to_7D_id[msdid]] for msdid in msdids]
    pool = mp.Pool(20)
    pool.map(save_resampled_npy, msd_paths)