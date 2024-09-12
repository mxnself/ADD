import json

import torch

from fairseq import checkpoint_utils


def load_dict(path):
    lines = open(path).readlines()
    words2idx = {}
    idx2words = {}
    idx = 4
    for line in lines:
        words2idx[line.split(" ")[0].strip()] = idx
        idx2words[idx] = line.split(" ")[0].strip()
        idx += 1
    return words2idx, idx2words


def load_weight(path):
    return torch.load(path)


def print_dict(root):
    if isinstance(root, dict):
        for key in root.keys():
            if not isinstance(root[key], torch.Tensor):
                print("{}:".format(key), end='')
                print_dict(root[key])
    elif isinstance(root, torch.Tensor):
        return
    elif isinstance(root, tuple):
        a, b = root
        print(a)
    else:
        # a=1
        print(root)


if __name__ == '__main__':
    input_path = "../models/avg_10_checkpoint_best.pt"
    output_path = "../models/base_uninit.pt"

    model_weight=load_weight(input_path)

    embedding_weight=model_weight["model"]["decoder.embed_tokens.weight"]
    print(embedding_weight.size())
    cutoffs=[0,858,1464,2356,3078,3947,4895,5892,7151,8263,9219,10000]
    head_tensor=embedding_weight.new_zeros((11,256))
    for i in range(11):
        start_idx=cutoffs[i]
        end_idx=cutoffs[i+1]
        mean_embedding_cluster_weight=embedding_weight[start_idx:end_idx,:].mean(dim=0)
        head_tensor[i,:]=mean_embedding_cluster_weight
    model_weight["model"]["decoder.adaptive_softmax.tail.weight"]=model_weight["model"]["decoder.output_projection.weight"]
    model_weight["model"].pop("decoder.output_projection.weight",None)
    torch.save(model_weight,output_path)
    print("save finish")

    state = checkpoint_utils.load_checkpoint_to_cpu(
        output_path, {"adaptive_softmax_type": "twostep",
                      "adaptive_softmax_cutoff": "(858, 1464, 2356, 3078, 3947, 4895, 5892, 7151, 8263, 9219)",
                      "adaptive_softmax_dropout": "0.0",
                      "adaptive_softmax_share_embedding": True,
                      "adaptive_softmax_factor": "2",
                      "tie_adaptive_weights": False,
                      "tie_adaptive_proj": False}
    )
    torch.save(state, output_path)
    print("save finish")

    print_dict(state)
