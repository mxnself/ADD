if __name__ == '__main__':

    # 统计最大帧clean-100: max_frames=2451
    file_name_list=["/home/gaochenghao/data/librispeech_asr/train-clean-100.tsv"]
    output_file_list=["/home/gaochenghao/data/librispeech_asr/train-clean-100-p1-tmp.tsv"]

    max_frams=0
    for i in range(len(file_name_list)):
        count = 0
        lines=open(file_name_list[i]).readlines()

        for line in lines[1:]:
            count += 1
            parts = line.strip().split('\t')
            frames = int(parts[2].strip())
            max_frams=max(max_frams,frames)
    print("max_frams={}".format(max_frams))
