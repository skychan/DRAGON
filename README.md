# DARGON

This repository contains the official implementation for the **AAMAS 2026** paper:

> **DARGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization**

DARGON introduces a multi-agent framework where large language models (LLMs) are used to **decompose** complex combinatorial optimization problems into tractable subproblems and **reconstruct** high-quality global solutions through iterative coordination.

Check [paper](https://arxiv.org/abs/2601.06502) for more details.

---

## Environment Settings

We recommend using a Conda environment.

```bash
conda create -n dargon python=3.10 -y
conda activate dargon
pip install uv
uv pip install -r requirements.txt
```

DARGON relies on LLM APIs (e.g., OpenAI and DeepSeek). Create a `.env` file in the project root and configure it as follows:

```txt
OPENAI_API_KEY = 'sk-proj-xxxxxx'
DEEPSEEK_API_KEY='sk-xxxxxx'
```

---


## Dataset Preparation

DARGON expects all problem instances to be organized under a unified `instances/` directory.

1. [TSPLib](https://github.com/mastqe/tsplib)

2. [CVRPLib](https://galgos.inf.puc-rio.br/cvrplib/index.php/en/instances)

3. [CVRP XML100](https://github.com/RoyalSkye/Omni-VRP)

4. [Bin Packing problem](https://github.com/google-deepmind/funsearch/blob/main/bin_packing/bin_packing.ipynb)

5. Multiple Knapsack Problem, generate with `generate_instance.py`, we provide the generated data in `instances/`

6. Assignment problem, generate with `generate_instance.py`, we provide the generated data in `instances/`


Simply unzip `instances.zip`.

---


Prepare your data in the following structure:

```text
instances/
├── assignment/
│   └── large/
│       ├── large_7_0.json
│       ├── large_18_0.json
│       └── ...
├── bin_packing/
│   └── weibull_5k/
│       ├── test_0.json
│       └── ...
├── CVRP/
│   ├── Vrp-Set-X/
│   │   └── X/
│   │       ├── X-n101-k25.vrp
│   │       └── ...
│   └── Vrp-Set-XML/
│       └── ...
├── knapsack/
│   └── multi_knap_large/
│       ├── mkp_393_5_0.json
│       └── ...
└── tsplib/
    ├── a280.npy
    └── ...

```
 

## Run the Code

### TSP:

```bash
python run_tsp.py
```

For TSP with ortools:

```bash
python run_tsp_ortools.py
```

### CVRP:

```bash
python run_cvrp.py
```

For CVRP with ortools:

```bash
python run_cvrp_ortools.py
```

### For MKP:

```bash
python run_mkp.py
```

For MKP with ortools:

```bash
python run_mkp_ortools.py
```

### For BPP:

```bash
python run_bpp.py
```

For MKP with ortools:

```bash
python run_bpp_ortools.py
```

### For AP:

```bash
python run_ap.py
```

For AP with ortools:

```bash
python run_ap_ortools.py
```

---

For other methods, please check the link in Acknowledgements.

---

## Acknowledgements

We would like to express our sincere thanks to the following works:

- OPRO: https://github.com/google-deepmind/opro 
- SGE: https://github.com/Zangir/LLM-for-CP
- LMEA: https://github.com/cschen1205/LMEA
- ReEvo: https://github.com/ai4co/reevo

---

## Citations

If you find this work useful, please consider citing:

```bibtex
@inproceedings{dargon2026,
  title     = {DARGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization},
  author    = {Shengkai Chen and Zhiguang Cao and Jianan Zhou and Yaoxin Wu and Senthilnath Jayavelu and Zhuoyi Lin and Xiaoli Li and Shili Xiang},
  booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year      = {2026}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.
