#! /bin/bash

# training the model

gpu_num=8
update_freq=1
max_tokens=100000

extra_tag=
extra_parameter=
#extra_tag="${extra_tag}"
#extra_parameter="${extra_parameter} "

exp_tag=
#config_list=(base)
#config_list=(base conformer)
#config_list=(ConformerCTCSmall)

config_list=(purectc_pds_base_16)
#config_list=(pds_base)
#config_list=(pds_big)
#config_list=(pds_deep)

# exp full name
exp_name=

train_config=$(echo ${config_list[*]} | sed 's/ /,/g')

cmd="./run.sh
    --stage 1
    --stop_stage 1
    --gpu_num ${gpu_num}
    --update_freq ${update_freq}
    --train_config ${train_config}
    --max_tokens ${max_tokens}
    "

if [[ -n ${exp_name} ]]; then
    cmd="$cmd --exp_name ${exp_name}"
fi
if [[ -n ${exp_tag} ]]; then
    cmd="$cmd --exp_tag ${exp_tag}"
fi
if [[ -n ${extra_tag} ]]; then
    cmd="$cmd --extra_tag ${extra_tag}"
fi
if [[ -n ${extra_parameter} ]]; then
    cmd="$cmd --extra_parameter \"${extra_parameter}\""
fi

echo ${cmd}
eval ${cmd}
