gpus=("2" "3" "4" "5" "6" "7")
kd_data=("other-500-0104-1" "other-500-0104-2" "other-500-0104-3" "other-500-0104-4" "other-500-0104-5" "other-500-0104-6")
gpu_num=${#gpus[@]}


data_tag=librispeech_asr
tag=baseline_1228

# 使用循环通过下标索引访问数组元素
for ((i=0; i<$gpu_num; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpus[i]}
    cmd="python -u /mnt/zhangyh/gch/Fairseq-S2T_old/fairseq_cli/generate.py
            /mnt/zhangyh/gch/data/$data_tag
            --config-yaml config.yaml
            --gen-subset ${kd_data[i]}
            --task speech_to_text
            --path /mnt/zhangyh/gch/data/checkpoints/${tag}/avg_10_checkpoint_best.pt
            --results-path /mnt/zhangyh/gch/data/checkpoints/${tag}
            --beam 2
            --lenpen 1.0
            --strict False
            --scoring wer
            --batch-size 1 >${kd_data[i]}.log 2>&1 &"
      echo $cmd
      eval $cmd
      sleep 2
done