import torch


def load_dict(path):
    lines=open(path).readlines()
    words2idx={}
    idx2words = {}
    idx=4
    for line in lines:
        words2idx[line.split(" ")[0].strip()]=idx
        idx2words[idx]=line.split(" ")[0].strip()
        idx+=1
    return words2idx,idx2words

def load_weight(path):
    return torch.load(path)


def reorder_embedding_weight(phone1220_idx2word,baseline_word2idx,weight):
    vocab_size=10000
    hidden_size=256
    assert weight.size(0)==vocab_size
    assert weight.size(1) == hidden_size
    out=weight.new_zeros((vocab_size,hidden_size))

    out[:4,:]=weight[:4,:]
    for i in range(4,vocab_size):
        #print("{}\t{}\t{}".format(i,phone1220_idx2word[i],baseline_word2idx[phone1220_idx2word[i]]))
        assert phone1220_idx2word[i]==baseline_idx2word[baseline_word2idx[phone1220_idx2word[i]]]
        out[i,:]=weight[baseline_word2idx[phone1220_idx2word[i]],:]

    return out

def print_dict(root):
    if isinstance(root,dict):
        for key in root.keys():
            if not isinstance(root[key],torch.Tensor):
                print("{}:".format(key),end='')
                print_dict(root[key])
    else:
        print(root)

if __name__ == '__main__':
    dict_baseline_path="/home/zhangyh/gch/data/librispeech_asr/spm_unigram10000.txt"
    dict_phone1220_path="/home/zhangyh/gch/data/librispeech_asr_random/spm_unigram10000_random.txt"

    baseline_word2idx,baseline_idx2word=load_dict(dict_baseline_path)
    phone1220_word2idx,phone1220_idx2word=load_dict(dict_phone1220_path)

    model=load_weight("/home/zhangyh/gch/data/checkpoints_new/baseline_newspec/avg_10_checkpoint_best.pt")
    #print_dict(model)
    embedding_weight=model["model"]["decoder.embed_tokens.weight"]
    #output_projection=model["model"]["decoder.output_projection.weight"]
    #print(output_projection.size())
    new_embedding_weight=reorder_embedding_weight(phone1220_idx2word,baseline_word2idx,embedding_weight)
    model["model"]["decoder.embed_tokens.weight"]=new_embedding_weight
    model["model"]["decoder.output_projection.weight"]=new_embedding_weight


    torch.save(model,"/home/zhangyh/gch/data/checkpoints_new/baseline2random_0205.pt")
    print("finish!")

