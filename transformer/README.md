### Implementation of Transformer

#### **1. Project Overview:**

- This project contains a basic implementation of Transformer(Attention is all you need) from scratch
- Only provides training on a small-scale machine translation (Chinese-English) dataset and a few example inferences, without providing evaluation code

#### 2. Installation

```bash
pip install -e .
```

#### 3. quick start

- run the `scripts/train.sh` to  get the checkpoint `./checkpoint/model.bin`

  ```bash
  bash scripts/train.sh
  ```

- run the `scripts/eval.sh` to get results from several examples

  ```python
  bash scripts/eval.sh
  ```

  