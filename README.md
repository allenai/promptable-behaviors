# Promptable Behaviors

Official implementation of the paper: "Promptable Behaviors: Personalizing Multi-Objective Rewards from Human Preferences" (CVPR 2024).

## Update

- Initial Code Release (03/27/2024)

## Installation

```bash
conda create --name $MY_ENV_NAME python=3.8
conda activate $MY_ENV_NAME

pip install -r requirements.txt
pip install -e "git+https://github.com/allenai/allenact.git@callbacks#egg=allenact&subdirectory=allenact"
pip install -e "git+https://github.com/allenai/allenact.git@callbacks#egg=allenact_plugins[ai2thor]&subdirectory=allenact_plugins"
pip install git+https://github.com/openai/CLIP.git@3b473b0e682c091a9e53623eebc1ca1657385717
```

### Training in ProcTHOR Environments
```bash
# objectnav
bash ./scripts/procthor-objectnav-train.sh

# fleenav
bash ./scripts/procthor-fleenav-train.sh
```

### Evaluation in ProcTHOR Environments
```bash
# objectnav
bash ./scripts/procthor-objectnav-eval.sh

# fleenav
bash ./scripts/procthor-fleenav-eval.sh
```

## Citation

If you find Promptable Behaviors useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{hwang2024promptable,
  title={Promptable Behaviors: Personalizing Multi-Objective Rewards from Human Preferences},
  author={Hwang, Minyoung and Weihs, Luca and Park, Chanwoo and Lee, Kimin and Kembhavi, Aniruddha and Ehsani, Kiana},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
