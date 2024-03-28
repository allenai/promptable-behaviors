while getopts w:c: flag
do
    case "${flag}" in
        w) reward_weights=${OPTARG};;
        c) checkpoint=${OPTARG};;
    esac
done
echo "Reward weights: ${reward_weights} | Checkpoint: ${checkpoint}";

OMP_NUM_THREADS=15 PYTHONPATH=. python procthor_visual_nav/main_objectnav.py \
experiment=procthor_visual_nav/experiments/rgb_clipresnet50gru_ddppo_morl \
agent=locobot wandb.project=procthor-objectnav machine.num_test_processes=15 machine.num_val_processes=10 \
ai2thor.platform=CloudRendering model.add_prev_actions_embedding=true procthor.p_randomize_materials=0.8 \
wandb.name=procthor-objectnav callbacks=procthor_visual_nav/callbacks/wandb_logging.py \
seed=100 wandb.use=false \
morl.adaptive_reward=false \
eval=true evaluation.tasks=["procthor-10k"] evaluation.minival=false \
checkpoint=${checkpoint} morl.reward_weights=${reward_weights}
