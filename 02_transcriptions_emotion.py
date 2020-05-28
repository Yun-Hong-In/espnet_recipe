import glob
import pandas as pd 
import numpy as np 

ang_trans = open("/data/corpus/IEMOCAP/esp/tot/transcriptions/ang/ang_tran.txt", 'wt')
hap_trans = open("/data/corpus/IEMOCAP/esp/tot/transcriptions/hap/hap_tran.txt", 'wt')
sad_trans = open("/data/corpus/IEMOCAP/esp/tot/transcriptions/sad/sad_tran.txt", 'wt')
neu_trans = open("/data/corpus/IEMOCAP/esp/tot/transcriptions/neu/neu_tran.txt", 'wt')
emo_list  = pd.read_csv("../emotion_classes.csv")

trans_list = glob.glob("/data/corpus/IEMOCAP/esp/tot/transcriptions/all/*.txt")
for file in trans_list:
    with open(file) as ifp:
        line = ifp.readline()
        while line != "":
            spk = line.split()[0]
            label = np.array(emo_list.EMOTION[emo_list.TURN_NAME == spk])
            if label == ['neu']:
                neu_trans.write(line)
                line = ifp.readline()
            elif label == ['sad']:
                sad_trans.write(line)
                line = ifp.readline()
            elif label == ['hap']:
                hap_trans.write(line)
                line = ifp.readline()
            elif label == ['ang']:
                ang_trans.write(line)
                line = ifp.readline()
            line = ifp.readline()


neu_trans.close()
sad_trans.close()
hap_trans.close()
ang_trans.close()
