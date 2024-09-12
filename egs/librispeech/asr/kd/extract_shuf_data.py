import os.path
import random


def load_tsv(root,file_name):
    if not file_name.endswith(".tsv"):
        file_name=file_name+".tsv"
    return open(os.path.join(root,file_name),'r',encoding='utf-8').readlines()[1:]

def split_lists(origin_list,split_num):
    length = len(origin_list)
    if length % split_num != 0:
        raise ValueError("请确保需要的数据量(need_line_num)为切分文件数量(split_num)的整数倍!")

    chunk_size = length // split_num
    result_lists = [origin_list[i * chunk_size:(i + 1) * chunk_size] for i in range(split_num)]

    return result_lists



if __name__ == '__main__':
    ROOT="/home/zhangyh/gch/data/librispeech_asr"
    head_line="id\taudio\tn_frames\ttgt_text\tspeaker"
    # train_sets=["train-clean-100","train-clean-360","train-other-500"]
    train_sets=["train-other-500"]
    output_set_tag = "other-500-0104"
    split_num=6
    need_line_num=148686

    origin_all_lines=[]
    for train_set in train_sets:
        origin_all_lines.extend(load_tsv(ROOT,train_set))
    print("输入总行数:{}".format(len(origin_all_lines)))


    random_need_lines=random.sample(origin_all_lines,need_line_num)

    split_extract_lists=split_lists(random_need_lines,split_num)

    for i in range(split_num):
        output_name=output_set_tag+"-"+str(i+1)+".tsv"
        with open(os.path.join(ROOT,output_name),'w',encoding='utf-8') as f:
            f.write(head_line+"\n")
            for line in split_extract_lists[i]:
                f.write(line.strip()+"\n")
        print("写入完成:{}".format(output_name))



