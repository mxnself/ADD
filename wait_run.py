# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import os
import re
import time

"""
version:1.3
descp:此版本用于排队等候指定数量的显卡，运行程序
time:2022/8/20--13:22
author:gch
"""


class GPU_Check:
    def __init__(self,
                 log_path="./sms.log",  # log文件
                 check_interval=60,  # 每次检测的间隔,60s
                 check_time=10,  # 连续几次检测到有空闲gpu后，执行程序
                 need_run_program=False,
                 need_gpu_count=1,
                 need_gpu_index=[],
                 scrip_dir="",
                 sh_command="",
                 ):

        self.check_time = check_time
        self.log_path = log_path
        self.log_path1 = "./wmt.txt"
        self.check_interval = check_interval
        self.need_run_program = need_run_program
        self.need_gpu_count = need_gpu_count
        self.need_gpu_index = need_gpu_index
        self.scrip_dir=scrip_dir
        self.sh_command=sh_command

    """
    一个检测周期（连续多次检测到有空闲GPU），
    """

    def check_and_run(self):
        count = 0
        while count < self.check_time:
            free_gpus_index = self.check_gpu_free()

            # 检查是否满足需要的显卡
            satisfy_gpu = True

            if len(self.need_gpu_index) > 0:
                # 指定了需要的gpu序号
                for gpu_index in self.need_gpu_index:
                    if gpu_index not in free_gpus_index:
                        satisfy_gpu = False
                        break
            else:
                # 否则根据数量判断
                satisfy_gpu = (len(free_gpus_index) >= self.need_gpu_count)

            if satisfy_gpu:
                count += 1
            else:
                count = 0

            content = "空闲gpu index:{}".format(free_gpus_index)
            self.log(content)

            if count < self.check_time:
                time.sleep(self.check_interval)
            else:
                # 此时立马抢显卡跑程序
                a = 1

        # 连续多次检测到有足够多的空闲gpu，开始跑程序
        # self.run_wmt_test(free_gpus_index)
        if len(self.need_gpu_index) > 0:
            self.run_train(self.need_gpu_index)
        else:
            self.run_train(free_gpus_index)

        self.log("循环检测程序退出~")

    def run_train(self, free_gpu_index: []):
        self.log("尝试运行脚本")
        pre_dir = os.getcwd()
        os.chdir(self.scrip_dir)
        self.log(self.sh_command)
        os.system(self.sh_command)
        os.chdir(pre_dir)
        return

    def run_decode_test(self, free_gpu_index: []):
        self.log("尝试运行脚本")
        pre_dir = os.getcwd()
        os.chdir(self.scrip_dir)
        # exp_names=["fa1700w_share_32k","shuf1000w_share_32000","1000w_32k_noshare","en_hr_bpe32k_share"]

        exp_names = ["en_hr_bpe32k_share"]

        n_grams = [5, 1]
        len_pens = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        for exp_name in exp_names:
            for n_gram in n_grams:
                for len_pen in len_pens:
                    lines = open("run.sh", 'r', encoding='utf-8').readlines()

                    lines[6] = "EXP_NAME={}".format(exp_name)
                    lines[25] = "n_average={}".format(n_gram)
                    lines[27] = "len_pen={}".format(len_pen)
                    self.log(lines[6])
                    self.log(lines[25])
                    self.log(lines[27])

                    with open("run.sh", 'w', encoding='utf-8') as f:
                        for line in lines:
                            f.write("{}\n".format(line.strip()))

                    lines = os.popen("sh run.sh 3").readlines()
                    bleu = lines[-4].split(" ")[2][:-1].strip()
                    sbleu = lines[-3].split(" ")[-14].strip()
                    char2f = lines[-2].split(" ")[-1].strip()
                    ter = lines[-1].split(" ")[-1].strip()
                    print(len(lines))

                    self.log_wmt(exp_name, n_gram, len_pen, bleu, sbleu, char2f, ter)
                    self.log("{}-{}-{}".format(exp_name, n_gram, len_pen))

        os.chdir(pre_dir)
        return

    def log(self,content):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        content = "{}    {}".format(time_str, content)
        print(content)
        if self.log_path is not None:
            with open(self.log_path, 'a', encoding="utf-8") as log_file:
                content = "{}\n".format(content)
                log_file.write(content)
                log_file.flush()

    def log_wmt(self, exp, n_avegra, len_pen, bleu, sbleu, char2f, ter):
        content = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(exp, n_avegra, len_pen, bleu, sbleu, char2f, ter)
        print(content)
        if self.log_path1 is not None:
            with open(self.log_path1, 'a', encoding="utf-8") as log_file:
                content = "{}\n".format(content)
                log_file.write(content)
                log_file.flush()

    """
    linux下利用nvidia-smi命令查看系统空闲的GPU数量
    """

    def check_gpu_free(self):
        cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
        lines = os.popen(cmd).readlines()

        gpu_memory_used = []

        for line in lines:
            temp = re.findall("\d+", line)
            memory = int(temp[0])
            gpu_memory_used.append(memory)

        free_gpus_index = []
        for index in range(len(gpu_memory_used)):
            item = gpu_memory_used[index]
            if item <= 100:
                free_gpus_index.append(index)
        return free_gpus_index


if __name__ == '__main__':
    GPU_Check(
        need_run_program=True,
        need_gpu_index=[0,1,2,3,4,5,6,7],
        check_time=2,
        log_path="./run.log",
        scrip_dir="/home/zhangyh/gch/Fairseq-S2T_old/egs/librispeech/asr",
        sh_command="nohup sh run.sh &",
    ).check_and_run()