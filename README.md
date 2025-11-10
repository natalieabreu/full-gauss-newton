# The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton 
This repository accompanies the paper **“[The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](https://arxiv.org/abs/2510.09378)” (Abreu et al., 2025)**. It builds off of the [EasyLM](https://github.com/young-geng/EasyLM) framework to support full and layer-wise **Gauss–Newton (GN)** preconditioning, as well as a prox-linear variant, to study the performance limits of second-order optimization in transformer-based language models. 

--- 

## Repository Structure 

We build directly on top of the EasyLM codebase. Files marked with `(*)` were modified from the EasyLM repo, and those with `(+)` were added for this project. 
```
EasyLM/ ├── data.py (*) # Modified to handle option for pretokenized dataset
        ├── gcs_utils.py (+) # Utilities for Google Cloud Storage
        ├── jax_utils.py (*) # Additional JAX/training utilities 
        ├── layerwise_utils.py (+) # Utilities for layer-wise GN computations
        ├── models/llama/
        │ ├── llama_model.py (*) # Added configs for 45M and 150M models
        │ ├── llama_train.py (*) # Baseline training: AdamW, Muon, SOAP
        │ ├── llama_train_gn.py (+) # Full GN and GN-prox-linear methods
        │ ├── llama_train_gn_layerwise.py (+) # Layer-wise GN and GN-prox-linear
        ├── optimizers.py (*) # Modified to include additional baselines
        ├── pretokenize.py (+) # Data preprocessing and tokenization
        ├── templates/
        │ ├── adam-template.sbatch (+) # SLURM script for baseline optimizers
        │ ├── gn-template.sbatch (+) # SLURM script for GN and GN-prox-linear runs
        ├── sweep_launcher.py (+) # Sweep launcher for hyperparameter tuning

```

 ---

## Training Scripts
| Script | Function |
|---|---|
| `llama_train.py` | Runs baselines (AdamW, Muon, SOAP). |
| `llama_train_gn.py` | Runs full Gauss–Newton (GN) and GN-prox-linear methods. |
| `llama_train_gn_layerwise.py` | Runs layer-wise Gauss–Newton and layer-wise GN-prox-linear methods. |

---

## Running Experiments

Example SLURM templates are provided in `EasyLM/templates/`.

### Baseline (AdamW / Muon / SOAP)
```bash sbatch EasyLM/templates/adam-template.sbatch ```

### Full GN or GN-prox-linear
```bash sbatch EasyLM/templates/gn-template.sbatch ```

Modify paths, model configuration, and hyperparameters as needed for your environment and hardware setup.

--- 

## Dependencies 

All dependencies are captured in `env.yml`.
```bash mamba env create -f env.yml mamba activate easylm ```

---

## Reproducing Paper Results 

Experiments are conducted on **45M- and 150M-parameter LLaMA models** trained on the **C4** dataset. For exact hyperparameters and setup details, see Appendix G of the paper.

---

## Acknowledgements

We gratefully acknowledge the creators of the [EasyLM](https://github.com/young-geng/EasyLM) project for developing the original framework on which this work builds.

---

## Citation 

If you use this codebase or build upon our work, please cite: 

```
@misc{abreu2025potentialsecondorderoptimizationllms,
      title={The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton}, 
      author={Natalie Abreu and Nikhil Vyas and Sham Kakade and Depen Morwani},
      year={2025},
      eprint={2510.09378},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.09378}, 
}
```

---

## License

This project follows the same license as the original EasyLM repository (Apache 2.0). See `LICENSE` for details.
