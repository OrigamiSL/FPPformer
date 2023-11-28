# Solar
python -u main.py --data Solar --features M --input_len 96  --pred_len 96 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --Cross --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 96  --pred_len 192 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --Cross --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 96  --pred_len 336 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --Cross --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 96  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --Cross --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 192  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --Cross --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 384  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --Cross --train_epochs 20 --itr 5 --train --patience 1

python -u main.py --data Solar --features M --input_len 576  --pred_len 720 --encoder_layer 3 --patch_size 6 --d_model 32 --learning_rate 0.0001 --dropout 0.1 --batch_size 8 --Cross --train_epochs 20 --itr 5 --train --patience 1
