import os
import subprocess
import numpy as np
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer

try:
    os.makedirs("/opt/espnet/egs/librispeech/asr1/data/neu")
    os.makedirs("/opt/espnet/egs/librispeech/asr1/data/hap")
    os.makedirs("/opt/espnet/egs/librispeech/asr1/data/ang")
    os.makedirs("/opt/espnet/egs/librispeech/asr1/data/sad")
except:
    pass
    

##### create all wav path.scp #####
subprocess.call("find /data/corpus/IEMOCAP/esp/tot/ -name '*.wav' -print > /data/remote/g2p/emotions/all_wavs.scp", shell=True)

emo_list = pd.read_csv("./emotion_labels.csv")

neu = open("./neu_path.scp", 'wt')
hap = open("./hap_path.scp", 'wt')
sad = open("./sad_path.scp", 'wt')
ang = open("./ang_path.scp", 'wt')
wav_list = open("./all_wavs.scp", 'rt')


line = wav_list.readline()
while line != "":
    file_name = line.split("/")[-1].split(".")[0]
    label = np.array(emo_list.EMOTION[emo_list.name == file_name])
    #print(label)
    if label == ['neu']:
        neu.write(line)
    elif label == ['hap']:
        hap.write(line)
    elif label == ['sad']:
        sad.write(line)
    elif label == ['ang']:
        ang.write(line)
    line = wav_list.readline()

neu.close()
hap.close()
sad.close()
ang.close()

##### create *_ID.scp #####
com_a1 = "cat ang_path.scp | awk -F / '{print($7)}' | awk -F . '{print($1)}' > ang_ID.scp"
com_h1 = "cat hap_path.scp | awk -F / '{print($7)}' | awk -F . '{print($1)}' > hap_ID.scp"
com_s1 = "cat sad_path.scp | awk -F / '{print($7)}' | awk -F . '{print($1)}' > sad_ID.scp"
com_n1 = "cat neu_path.scp | awk -F / '{print($7)}' | awk -F . '{print($1)}' > neu_ID.scp"
subprocess.call(com_a1, shell=True)
subprocess.call(com_h1, shell=True)
subprocess.call(com_s1, shell=True)
subprocess.call(com_n1, shell=True)

##### spk2gender #####
# angry
ang_spk = open("./ang_ID.scp", 'rt')
ang2gen = open("/opt/espnet/egs/librispeech/asr1/data/ang/spk2gender", 'wt')

line = ang_spk.readline()
while line != "":
    if line[5] == "F":
        ang2gen.write(line.split("_")[0] + " f\n")
        line = ang_spk.readline()
    else:
        ang2gen.write(line.split("_")[0] + " m\n")
        line = ang_spk.readline()
ang2gen.close()
ang_spk.close()

# happy
hap_spk = open("./hap_ID.scp", 'rt')
hap2gen = open("/opt/espnet/egs/librispeech/asr1/data/hap/spk2gender", 'wt')

line = hap_spk.readline()
while line != "":
    if line[5] == "F":
        hap2gen.write(line.split("_")[0] + " f\n")
        line = hap_spk.readline()
    else:
        hap2gen.write(line.split("_")[0] + " m\n")
        line =hap_spk.readline()
hap2gen.close()
hap_spk.close()

# sad
sad_spk = open("./sad_ID.scp", 'rt')
sad2gen = open("/opt/espnet/egs/librispeech/asr1/data/sad/spk2gender", 'wt')

line = sad_spk.readline()
while line != "":
    if line[5] == "F":
        sad2gen.write(line.split("_")[0] + " f\n")
        line = sad_spk.readline()
    else:
        sad2gen.write(line.split("_")[0] + " m\n")
        line = sad_spk.readline()
sad2gen.close()
sad_spk.close()

# neutral
neu_spk = open("./neu_ID.scp", 'rt')
neu2gen = open("/opt/espnet/egs/librispeech/asr1/data/neu/spk2gender", 'wt')

line = neu_spk.readline()
while line != "":
    if line[5] == "F":
        neu2gen.write(line.split("_")[0] + " f\n")
        line = neu_spk.readline()
    else:
        neu2gen.write(line.split("_")[0] + " m\n")
        line = neu_spk.readline()
neu2gen.close()
neu_spk.close()
#########################################################################################

ang_ID = open("./ang_ID.scp", "rt")
hap_ID = open("./hap_ID.scp", "rt")
sad_ID = open("./sad_ID.scp", "rt")
neu_ID = open("./neu_ID.scp", "rt")

##### utt2spk #####
# angry
with open("/opt/espnet/egs/librispeech/asr1/data/ang/utt2spk", 'w') as ifp:
    line = ang_ID.readlines()
    for l in line:
        ifp.write(l[:-1] + " " + l[:6] + "\n")

# happy
with open("/opt/espnet/egs/librispeech/asr1/data/hap/utt2spk", 'w') as ifp:
    line = hap_ID.readlines()
    for l in line:
        ifp.write(l[:-1] + " " + l[:6] + "\n")

# sad
with open("/opt/espnet/egs/librispeech/asr1/data/sad/utt2spk", 'w') as ifp:
    line = sad_ID.readlines()
    for l in line:
        ifp.write(l[:-1] + " " + l[:6] + "\n")

# neutral
with open("/opt/espnet/egs/librispeech/asr1/data/neu/utt2spk", 'w') as ifp:
    line = neu_ID.readlines()
    for l in line:
        ifp.write(l[:-1] + " " + l[:6] + "\n")

#########################################################################################

##### wav.scp #####
com_a2 = 'paste -d " " ang_ID.scp ang_path.scp > ./ang_wav.scp'
com_h2 = 'paste -d " " hap_ID.scp hap_path.scp > ./hap_wav.scp'
com_s2 = 'paste -d " " sad_ID.scp sad_path.scp > ./sad_wav.scp'
com_n2 = 'paste -d " " neu_ID.scp neu_path.scp > ./neu_wav.scp'
subprocess.call(com_a2, shell=True)
subprocess.call(com_h2, shell=True)
subprocess.call(com_s2, shell=True)
subprocess.call(com_n2, shell=True)

# angry
with open("./ang_wav.scp", 'rt') as ifp:
    line = ifp.readline()
    ang_wav = open("/opt/espnet/egs/librispeech/asr1/data/ang/wav.scp", 'wt')
    while line != "":
        a, b = line.split()
        ang_wav.write(a +  " " + b + " \n")
        line = ifp.readline()
    ang_wav.close()

# happy
with open("./hap_wav.scp", 'rt') as ifp:
    line = ifp.readline()
    hap_wav = open("/opt/espnet/egs/librispeech/asr1/data/hap/wav.scp", 'wt')
    while line != "":
        a, b = line.split()
        hap_wav.write(a +  " " + b + " \n")
        line = ifp.readline()
    hap_wav.close()

# sad
with open("./sad_wav.scp", 'rt') as ifp:
    line = ifp.readline()
    sad_wav = open("/opt/espnet/egs/librispeech/asr1/data/sad/wav.scp", 'wt')
    while line != "":
        a, b = line.split()
        sad_wav.write(a +  " " + b + " \n")
        line = ifp.readline()
    sad_wav.close()

# neutral
with open("./neu_wav.scp", 'rt') as ifp:
    line = ifp.readline()
    neu_wav = open("/opt/espnet/egs/librispeech/asr1/data/neu/wav.scp", 'wt')
    while line != "":
        a, b = line.split()
        neu_wav.write(a +  " " + b + " \n")
        line = ifp.readline()
    neu_wav.close()

#########################################################################################

##### text #####
ang_trans = open("/data/corpus/IEMOCAP/esp/tot/transcriptions/ang/ang_tran.txt", 'r')
hap_trans = open("/data/corpus/IEMOCAP/esp/tot/transcriptions/hap/hap_tran.txt", 'r')
sad_trans = open("/data/corpus/IEMOCAP/esp/tot/transcriptions/sad/sad_tran.txt", 'r')
neu_trans = open("/data/corpus/IEMOCAP/esp/tot/transcriptions/neu/neu_tran.txt", 'r')

brac = r'\[.*\]'
tokenizer = RegexpTokenizer(r'\w+')

# angry
ang_txt = open("/opt/espnet/egs/librispeech/asr1/data/ang/text", 'w')
line = ang_trans.readline()
while line != "":
    if line[0] != 'S':
        line = ang_trans.readlines()
        continue
    txt = line.split(":")[-1]
    txt = re.sub(brac, "", txt)
    tok = tokenizer.tokenize(txt)
    clean = " ".join(tok)

    spk = line.split(" ")[0]
    tmp = spk.split("_")
    num = tmp[0] + "_" + "".join(tmp[1:])

    if clean == "":
        line = ang_trans.readline()
        continue
    ang_txt.write(num + " " + clean + "\n")
    line = ang_trans.readline()
ang_txt.close()

# happy
hap_txt = open("/opt/espnet/egs/librispeech/asr1/data/hap/text", 'w')
line = hap_trans.readline()
while line != "":
    if line[0] != 'S':
        line = hap_trans.readlines()
        continue
    txt = line.split(":")[-1]
    txt = re.sub(brac, "", txt)
    tok = tokenizer.tokenize(txt)
    clean = " ".join(tok)

    spk = line.split(" ")[0]
    tmp = spk.split("_")
    num = tmp[0] + "_" + "".join(tmp[1:])

    if clean == "":
        line = hap_trans.readline()
        continue
    hap_txt.write(num + " " + clean + "\n")
    line = hap_trans.readline()    
hap_txt.close()

# sad
sad_txt = open("/opt/espnet/egs/librispeech/asr1/data/sad/text", 'w')
line = sad_trans.readline()
while line != "":
    if line[0] != 'S':
        line = sad_trans.readlines()
        continue
    txt = line.split(":")[-1]
    txt = re.sub(brac, "", txt)
    tok = tokenizer.tokenize(txt)
    clean = " ".join(tok)

    spk = line.split(" ")[0]
    tmp = spk.split("_")
    num = tmp[0] + "_" + "".join(tmp[1:])

    if clean == "":
        line = sad_trans.readline()
        continue
    sad_txt.write(num + " " + clean + "\n")
    line = sad_trans.readline()    
sad_txt.close()

# neutral
neu_txt = open("/opt/espnet/egs/librispeech/asr1/data/neu/text", 'w')
line = neu_trans.readline()
while line != "":
    if line[0] != 'S':
        line = neu_trans.readlines()
        continue
    txt = line.split(":")[-1]
    txt = re.sub(brac, "", txt)
    tok = tokenizer.tokenize(txt)
    clean = " ".join(tok)

    spk = line.split(" ")[0]
    tmp = spk.split("_")
    num = tmp[0] + "_" + "".join(tmp[1:])

    if clean == "":
        line = neu_trans.readline()
        continue
    neu_txt.write(num + " " + clean + "\n")
    line = neu_trans.readline()    
neu_txt.close()