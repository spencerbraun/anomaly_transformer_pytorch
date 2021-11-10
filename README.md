# Anomaly Transformer in PyTorch

This is an implementation of [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642). This paper is currently [under review](https://openreview.net/forum?id=LzQQ89U1qm_) and in need of some clarification around the attention mechanism. This repo will be updated as more information is provided.

## Usage

### Requirements

Install dependences into a virtualenv:

```bash
$ python -m venv env
$ source env/bin/activate
(env) $ pip install -r requirements.txt
```

### Data and Configuration

Custom datasets can be placed in the `data/` dir. Edits should be made to the `conf/data/default.yaml` file to reflect the correct properties of the data. All other configuration hyperparameters can be set in the hydra configs.

### Train

Once properly configured, a model can be trained via `python train.py`.

## Citations

```bibtex
@misc{xu2021anomaly,
      title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
      author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
      year={2021},
      eprint={2110.02642},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
