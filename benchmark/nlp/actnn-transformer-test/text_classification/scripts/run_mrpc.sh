TASK_NAME="mrpc"
CONFIG="config/default.yaml"

for lr in 3e-5 4e-5 5e-5 6e-5 7e-5; do
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --learning_rate $lr \
  --num_train_epochs 10 \
  --seed 42 \
  --output_dir log/$TASK_NAME-actnn-adp2-10epoch/ \
  --actnn \
  --opt_level adp2
done