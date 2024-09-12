import numpy as np
from sklearn.cluster import KMeans
def load_data(path):
    lines=open(path).readlines()
    max_len=0
    min_len=999
    all_words=[]
    all_nums=[]
    for line in lines:
        word=line.strip().split("\t")[0]
        nums=[int(num) for num in line.strip().split("\t")[1].split(" ")]
        max_len=max(max_len,len(nums))
        min_len = min(min_len, len(nums))
        all_nums.append(nums)
        all_words.append(word)

    numpy_data=[]
    for nums in all_nums:
        #padded_nums = np.pad(nums, (0, max_len - len(nums)), 'constant', constant_values=0)

        numpy_data.append(nums[:15])


    print("min_len={},max_len={}".format(min_len,max_len))

    matrix = np.array(numpy_data)
    return all_words,matrix
if __name__ == '__main__':

    words,data=load_data("../data/hubert/dict.decode")
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(data)
    id2word={}
    for i in range(len(words)):
        word=''.join(words[i].split())
        if clusters[i] not in id2word:
            id2word[clusters[i]]=[word]
        else:
            id2word[clusters[i]].append(word)


    cutoffs=[]
    with open("../data/hubert/dict.decode.reorder",'w',encoding='utf-8')as f:
        f.write("' 1\n")
        count=4+1
        for i in range(10):
            print("==============={}==============".format(i))
            list_word=id2word[i]
            print(len(list_word))
            print(list_word[:10])
            count+=len(list_word)
            cutoffs.append(count)
            for word in list_word:
                f.write("{} 1\n".format(word))
    print("\n")
    print("cutoffs:",cutoffs)
