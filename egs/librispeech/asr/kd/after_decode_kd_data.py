import os.path
import random


def load_tsv(root,file_name):
    if not file_name.endswith(".tsv"):
        file_name=file_name+".tsv"
    lines= open(os.path.join(root,file_name),'r',encoding='utf-8').readlines()[1:]
    order2target = {}
    order2nframes = {}
    order2path = {}
    order2speaker = {}
    for i in range(len(lines)):
        order2path[i]=lines[i].strip().split("\t")[1]
        order2nframes[i] = lines[i].strip().split("\t")[2]
        order2target[i] = lines[i].strip().split("\t")[3]
        order2speaker[i] = lines[i].strip().split("\t")[4]
    return order2path,order2nframes,order2target,order2speaker


def load_translation_txt(path):
    lines=open(path,'r',encoding='utf-8').readlines()
    order2target={}
    order2translation = {}
    for line in lines:
        order=eval(line.strip().split("\t")[0])
        target = line.strip().split("\t")[2]
        translation=line.strip().split("\t")[3]
        order2target[order]=target
        order2translation[order] = translation
    return order2target,order2translation


if __name__ == '__main__':
    ROOT="/home/zhangyh/gch/data/librispeech_asr"
    head_line="id\taudio\tn_frames\ttgt_text\tspeaker"
    target_tsvs= [f"all-0103-{i}" for i in range(1, 6 + 1)]

    translation_path="/home/zhangyh/gch/data/checkpoints/baseline_1228"
    output_filename="all-0103.tsv"

    output_lines=[]
    id=0
    differ_num=0
    for target_tsv in target_tsvs:
        translation_filename="translation-"+target_tsv+".txt"
        order2path, order2nframes, order2target,order2speaker=load_tsv(ROOT,target_tsv)
        order2target1,order2translation=load_translation_txt(os.path.join(translation_path,translation_filename))
        for order in range(len(order2path)):
            if order2target[order]!=order2target1[order]:
                print("error:\t{}\t{} != {}".format(str(order),order2target[order],order2target1[order]))
            if order2target[order]!=order2translation[order]:
                differ_num+=1
            output_lines.append("999-999-{}\t{}\t{}\t{}".format(order+1,order2path[order],order2nframes[order],order2translation[order],order2speaker[order]))

    with open(os.path.join(ROOT, output_filename), 'w', encoding='utf-8') as f:
        f.write(head_line+"\n")
        for line in output_lines:
            f.write(line+"\n")
    print("写入完成:{}".format(os.path.join(ROOT, output_filename)))
    print("翻译出来与target不同的句子数量:{}".format(differ_num))



