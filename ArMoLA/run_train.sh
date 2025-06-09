mola_weights_dir=""
task_name=""
CUDA_VISIBLE_DEVICES=0,1,2,3 python mola_training.py \
         --base_model "model_path" \
         --data_path "dataset_path/${task_name}.hf" \
         --output_dir $mola_weights_dir \
         --batch_size 64 \
         --micro_batch_size 16 \
         --num_epochs 4 \
         --learning_rate 3e-4 \
         --cutoff_len 256 \
         --val_set_size 1 \
         --lora_r "8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8" \
         --lora_alpha 16 \
         --lora_dropout 0.05 \
         --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj" \
         --number_experts "8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8" \
         --top_k "2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
         --train_on_inputs \
         --group_by_length \
         --add_eos_token \
         --obalance True \

python save_rankandexp_num.py --dir $mola_weights_dir --json_name $task_name

