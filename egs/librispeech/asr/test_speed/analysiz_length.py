import sentencepiece as sp
if __name__ == '__main__':
    spm=sp.SentencePieceProcessor(model_file='/home/zhangyh/gch/data/librispeech_asr/spm_unigram10000.model')

    lines=open("/home/zhangyh/gch/data/librispeech_asr/dev-clean.tsv").readlines()[1:]
    audio_lens=[]
    txt_lens=[]
    for line in lines:
        n_frame=eval(line.split("\t")[2])
        tgt_text=line.split("\t")[3]
        audio_lens.append(n_frame//4)
        txt_lens.append(len(spm.encode(tgt_text.strip()))+2) # sos,eos


    print(sum(audio_lens)/len(audio_lens))
    print(sum(txt_lens)/len(txt_lens))

    print(sum(audio_lens)/sum(txt_lens))