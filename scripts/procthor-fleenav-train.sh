while getopts w: flag
do
    case "${flag}" in
        w) reward_weights=${OPTARG};;
    esac
done
echo "Reward weights: ${reward_weights}";
reward_weights_string=$(echo ${reward_weights} | sed 's/,/_/g')

### FLEENAV ###
OMP_NUM_THREADS=45 PYTHONPATH=. python procthor_objectnav/main_fleenav.py \
experiment=procthor_objectnav/experiments/rgb_clipresnet50gru_ddppo_fleenav_morl \
agent=locobot wandb.project=procthor-fleenav machine.num_train_processes=45 machine.num_val_processes=2 \
ai2thor.platform=CloudRendering model.add_prev_actions_embedding=false procthor.p_randomize_materials=0.8 \
wandb.name=fixed-weights-${reward_weights_string} callbacks=procthor_objectnav/callbacks/wandb_logging.py \
seed=100 wandb.use=true \
morl.adaptive_reward=false \
morl.reward_weights=[${reward_weights}]