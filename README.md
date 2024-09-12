# Adaptive Decoding for Efficient Automatic Speech Recognition

## Overview

This repository is the code of paper "Adaptive Decoding for Efficient Automatic Speech Recognition". a simple and effective adaptive decoding strategy for ASR task.
Here is an example on the MUST-C ST dataset.

- Stage 0 performs the data processing.
- Stage 1 performs the model training, where multiple choices are supported.
- Stage 2 performs the model inference.

- All details are available in **run.sh**.

The main method can be found at: **ADD/fairseq/modules/twostep_softmax.py**

## Installation

1. Clone the repository:

    ```bash
    git clone 
    ```

2. Navigate to the project directory and install the required dependencies:

    ```bash
    cd ADD
    pip install -e .
    ```
    Our version: python 3.10, pytorch 2.0.0.

To train the model on the Librispeech dataset.
Here is an example on the small model.
1. Download your dataset and process it into the format of Librispeech dataset.
2. Run the shell script **run.sh** in the corresponding directory as follows:

```bash
cd egs/librispeech/asr/
# Set root_dir environment variable as the parent directory of ADD directory
export root_dir=/path/to/ADD/..
# Non-autoregressive modeling
./run.sh --stage 0 --stop_stage 2
```

To make the cluster of words, you should run the code in the "ADD/egs/librispeech/asr/sortdict"

Here is an example on the W2P cluster strategy. you should first download the CMU dictionary to 'ADD/egs/librispeech/asr/data/phone' and change paths in the file.

```bash
python sort_dict_by_phone.py
```

## Acknowledgments

- Fairseq community for the base toolkit
- NiuTrans Team for their contributions and research
