if __name__ == '__main__':
    file_name="/home/gaochenghao/data/librispeech_asr/train-clean-100.tsv"
    lines = open(file_name).readlines()
    head = lines[0]
    lines=lines[1:]

    split_num=8
    num_each_file=int(len(lines)/8)
    for i in range(split_num):
        with open("/home/gaochenghao/data/librispeech_asr/train-clean-100-p{}.tsv".format(i+1),'w') as f:
            f.write(head)
            for j in range(i*num_each_file,(i+1)*num_each_file):
                f.write(lines[j])

    with open("/home/gaochenghao/data/librispeech_asr/train-clean-100-p{}.tsv".format(8), 'a') as f:
        for j in range(split_num*num_each_file,len(lines)):
            f.write(lines[j])