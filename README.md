# Promptable Behaviors

Official implementation of the paper: "Promptable Behaviors: Personalizing Multi-Objective Rewards from Human Preferences" (CVPR 2024).

## Update

- Initial Code Release (03/27/2024)

## Installation

```bash
conda create --name allenact python=3.8
conda activate allenact
conda install nvidia/label/cuda-11.7.0::cuda-toolkit
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
pip install -e "git+https://github.com/allenai/allenact.git@callbacks#egg=allenact&subdirectory=allenact"
pip install -e "git+https://github.com/allenai/allenact.git@callbacks#egg=allenact_plugins[ai2thor]&subdirectory=allenact_plugins"
pip install git+https://github.com/openai/CLIP.git@3b473b0e682c091a9e53623eebc1ca1657385717

# Download pretrained ImageNet and CLIP visual encoders
python -c "from torchvision import models; models.resnet50(pretrained=True)"
python -c "import clip; clip.load('RN50')"
```

### Training in ProcTHOR Environments
```bash
export WANDB_API_KEY=<your-api-key>

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
