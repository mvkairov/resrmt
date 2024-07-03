# Recurrent Memory Transformer implementation for Hugging Face models


## RMT resources

[ [paper](https://arxiv.org/abs/2207.06881) ] [ [code](https://github.com/booydar/recurrent-memory-transformer/) ] **Recurrent Memory Transformer** implementation and various training examples.

[ [paper](https://arxiv.org/abs/2402.10790) ] [ [code](https://github.com/booydar/recurrent-memory-transformer/tree/babilong-release) ] **BABILong** - a long-context benchmark that supports 20 tasks and various sources of background text. 

[ [code](https://github.com/booydar/babilong) ] **Evaluate your long-context LLM on BABILong!**

RMT is a memory-augmented segment-level recurrent Transformer. We implement our memory mechanism as a wrapper for any Hugging Face model by adding special memory tokens to the input sequence. The model is trained to control both memory operations and sequence representations processing.

<img src="img/RMT_scheme.png" alt="drawing" width="400"/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/booydar/t5-experiments/blob/framework_accel/notebooks/rmt_demo_lm.ipynb) Example: LM with RMT


This work was done in collaboration of DeepPavlov.ai, AIRI, and London Institute for Mathematical Sciences.

## Installation
```bash
pip install -e .
```
This command will install `lm_experiments_tools` with only required packages for Trainer and tools.

`lm_experiments_tools` Trainer supports gradient accumulation, logging to tensorboard, saving the best models
based on metrics, custom metrics and data transformations support.

### Install requirements for all experiments
Full requirements for all experiments are specified in requirements.txt. Install requirements after cloning the repo:
```bash
pip install -r requirements.txt
```


## Citation
If you find our work useful, please consider citing the RMT papers:
```
@inproceedings{
        bulatov2022recurrent,
        title={Recurrent Memory Transformer},
        author={Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=Uynr3iPhksa}
}
```
```
@misc{bulatov2023scaling,
      title={Scaling Transformer to 1M tokens and beyond with RMT}, 
      author={Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
      year={2023},
      eprint={2304.11062},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```
@misc{kuratov2024search,
      title={In Search of Needles in a 11M Haystack: Recurrent Memory Finds What LLMs Miss}, 
      author={Yuri Kuratov and Aydar Bulatov and Petr Anokhin and Dmitry Sorokin and Artyom Sorokin and Mikhail Burtsev},
      year={2024},
      eprint={2402.10790},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
