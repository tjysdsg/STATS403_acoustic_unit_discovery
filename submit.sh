if [ "$#" -le 1 ]; then
  # submit <experiment name> <script> [args...]
  echo "Need arguments"
fi

exp_dir=exp/"$1"
shift
mkdir -p $exp_dir

sbatch -N1 -n1 --ntasks-per-node=1 -p gpu \
  --gres=gpu:8 \
  --mem=10GB --cpus-per-task=20 \
  --time=72:0:0 --export=PATH \
  -e ${exp_dir}/slurm.log -o ${exp_dir}/slurm.log \
  /home/storage15/tangjiyang/DAU-MD/slurm_script_runner.sh "$@"
