# encoder_layer {1, 2, 3, 4} 3 is the default
python -u main.py --data ETTh1 --features M --input_len 576  --pred_len 720 --encoder_layer 1 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTh2 --features M --input_len 576  --pred_len 720 --encoder_layer 1 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTm1 --features M --input_len 576  --pred_len 720 --encoder_layer 1 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTm2 --features M --input_len 576  --pred_len 720 --encoder_layer 1 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ECL --features M --input_len 576  --pred_len 720 --encoder_layer 1 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Traffic --features M --input_len 576  --pred_len 720 --encoder_layer 1 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 576  --pred_len 720 --encoder_layer 1 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data weather --features M --input_len 576  --pred_len 720 --encoder_layer 1 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTh1 --features M --input_len 576  --pred_len 720 --encoder_layer 2 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTh2 --features M --input_len 576  --pred_len 720 --encoder_layer 2 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTm1 --features M --input_len 576  --pred_len 720 --encoder_layer 2 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTm2 --features M --input_len 576  --pred_len 720 --encoder_layer 2 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ECL --features M --input_len 576  --pred_len 720 --encoder_layer 2 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Traffic --features M --input_len 576  --pred_len 720 --encoder_layer 2 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 576  --pred_len 720 --encoder_layer 2 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data weather --features M --input_len 576  --pred_len 720 --encoder_layer 2 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTh1 --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTh2 --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTm1 --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTm2 --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ECL --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Traffic --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data weather --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTh1 --features M --input_len 576  --pred_len 720 --encoder_layer 4 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTh2 --features M --input_len 576  --pred_len 720 --encoder_layer 4 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTm1 --features M --input_len 576  --pred_len 720 --encoder_layer 4 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ETTm2 --features M --input_len 576  --pred_len 720 --encoder_layer 4 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data ECL --features M --input_len 576  --pred_len 720 --encoder_layer 4 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Traffic --features M --input_len 576  --pred_len 720 --encoder_layer 4 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 576  --pred_len 720 --encoder_layer 4 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data weather --features M --input_len 576  --pred_len 720 --encoder_layer 4 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 20 --itr 5 --train --patience 1
