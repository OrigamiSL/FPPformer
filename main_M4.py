import argparse
import os
import torch
import numpy as np
import time

from exp.exp_model_M4 import Exp_Model

parser = argparse.ArgumentParser(description='')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--input_len', type=int, default=96, help='input length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
parser.add_argument('--freq', type=str, default='Daily', help='Freq for M4')

parser.add_argument('--d_model', type=int, default=28, help='hidden dims of model')
parser.add_argument('--encoder_layer', type=int, default=3)
parser.add_argument('--patch_size', type=int, default=6, help='the size of each patch')

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--save_loss', action='store_true', help='whether saving results and checkpoints', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--train', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'M4_yearly': {'data': 'Yearly-train.csv', 'root_path': './data/M4/'},
    'M4_quarterly': {'data': 'Quarterly-train.csv', 'root_path': './data/M4/'},
    'M4_monthly': {'data': 'Monthly-train.csv', 'root_path': './data/M4/'},
    'M4_weekly': {'data': 'Weekly-train.csv', 'root_path': './data/M4/'},
    'M4_daily': {'data': 'Daily-train.csv', 'root_path': './data/M4/'},
    'M4_hourly': {'data': 'Hourly-train.csv', 'root_path': './data/M4/'},
}

type_map = {'Yearly': 1, 'Quarterly': 4, 'Monthly': 12, 'Weekly': 1, 'Daily': 1, 'Hourly': 24}
args.frequency = type_map[args.freq]

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.root_path = data_info['root_path']

lr = args.learning_rate
print('Args in experiment:')
print(args)

smape_total = []
owa_total = []

Exp = Exp_Model
for ii in range(args.itr):
    # setting record of experiments
    exp = Exp(args)  # set experiments
    if args.train:
        setting = '{}_ll{}_pl{}_{}'.format(args.data,
                                           args.input_len,
                                           args.pred_len, ii)
        print('>>>>>>>start training| pred_len:{}, settings: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.
              format(args.pred_len, setting))
        try:
            exp = Exp(args)  # set experiments
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from forecasting early')

        print('>>>>>>>testing| pred_len:{}: {}<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments
        smape, owa = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
        smape_total.append(smape)
        owa_total.append(owa)
        torch.cuda.empty_cache()
        args.learning_rate = lr
    else:
        setting = '{}_ll{}_pl{}_{}'.format(args.data,
                                           args.input_len,
                                           args.pred_len, ii)
        print('>>>>>>>testing| pred_len:{} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments

        smape, owa = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
        smape_total.append(smape)
        owa_total.append(owa)
        torch.cuda.empty_cache()
        args.learning_rate = lr

path1 = './result_M4.csv'
if not os.path.exists(path1):
    with open(path1, "a") as f:
        write_csv = ['Time', 'Data', 'input_len', 'pred_len', 'encoder_layer', 'patch_size', 'Mean smape',
                     'Mean owa', 'Std smape', 'Std owa']
        np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
        f.flush()
        f.close()

smape = np.asarray(smape_total)
owa = np.asarray(owa_total)
avg_smape = np.mean(smape)
std_smape = np.std(smape)
avg_owa = np.mean(owa)
std_owa = np.std(owa)

print('|Mean|smape:{}, owa:{}|Std|smape:{}, owa:{}'.format(avg_smape, avg_owa, std_smape, std_owa))
path = './result_M4.log'
with open(path, "a") as f:
    f.write('|{}|pred_len{}: '.format(
        args.data, args.pred_len) + '\n')
    f.write('|Mean|smape:{}, owa:{}|Std|smape:{}, owa:{}'.
            format(avg_smape, avg_owa, std_smape, std_owa) + '\n')
    f.flush()
    f.close()
with open(path1, "a") as f:
    f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    f.write(',{},{},{},{},{},{},{},{},{}'.
            format(args.data, args.input_len, args.pred_len, args.encoder_layer,
                   args.patch_size, avg_smape, avg_owa, std_smape, std_owa) + '\n')
    f.flush()
    f.close()
