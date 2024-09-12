if __name__ == '__main__':


    file_name_list=["/home/gaochenghao/data/librispeech_asr/train-clean-100-reorder.tsv"]
    output_file_list=["/home/gaochenghao/data/librispeech_asr/train-clean-100-scale.tsv"]


    for i in range(len(file_name_list)):
        count = 0
        lines=open(file_name_list[i]).readlines()
        head_line=lines[0].strip()

        with open(output_file_list[i],'w') as f:
            f.write('{}\n'.format(head_line))
            count+=1
            for line in lines[1:]:
                parts = line.strip().split('\t')
                scale_frames = int(parts[2].strip())*4
                parts[2] = str(scale_frames)
                new_line = '\t'.join(parts).strip()
                f.write('{}\n'.format(new_line))
                count+=1
        print("finish process:{} line".format(count))