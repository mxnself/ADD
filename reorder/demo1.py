


if __name__ == '__main__':

    # step1
    file_name_list=["/home/gaochenghao/data/mustc_ende/tst-COMMON.tsv"]
    output_file_list=["/home/gaochenghao/data/mustc_ende/tst-COMMON-tmp.tsv"]

    for i in range(len(file_name_list)):
        count = 0
        lines=open(file_name_list[i]).readlines()

        with open(output_file_list[i], 'w') as f:
            f.write('{}\n'.format(lines[0].strip()))
            for line in lines[1:]:
                count+=1
                parts=line.strip().split('\t')
                parts[2]=str(count)
                new_line='\t'.join(parts).strip()
                f.write('{}\n'.format(new_line))
        print("finish process {} line".format(count))


    # step2
    # frames_file_name="/home/gaochenghao/data/librispeech_asr/subsample_lens_clean100.txt"
    # file_name = "/home/gaochenghao/data/librispeech_asr/train-clean-100.tsv"
    # output_file="/home/gaochenghao/data/librispeech_asr/train-clean-100-reorder.tsv"
    #
    # index=0
    # lines=[]
    # frames=open(frames_file_name).readlines()
    #
    # lines = open(file_name).readlines()
    #
    # with open(output_file, 'w') as f:
    #     f.write('{}\n'.format(lines[0].strip()))
    #     for line in lines[1:]:
    #         parts = line.strip().split('\t')
    #         parts[2] = str(frames[index].strip())
    #         new_line = '\t'.join(parts).strip()
    #         f.write('{}\n'.format(new_line))
    #         index += 1