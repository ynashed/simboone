import argparse
import h5py as h5
import numpy as np

import torch
from SimIonization import SimIonization

parser = argparse.ArgumentParser(description='Inverse model')
parser.add_argument('--data_file', type=str, default="",
    help='Path to .h5 data file')
parser.add_argument('--sample_id', type=int, default=0,
    help='Select a detection sample from the data file to test')
parser.add_argument('--num_step', type=int, default=2000,
    help='Number of optimization steps')
parser.add_argument('--print_step', type=int, default=100,
    help='Frequence of printing loss')
parser.add_argument('--N', type=int, default=20000,
    help='Number of points in input')
parser.add_argument('--C', type=int, default=3,
    help='Number of channels in input')
parser.add_argument('--test_case', type=int, default=0, choices=[0, 1, 2], 
    help='\
    0: Fixed alpha and beta, \
    1: Uniformed sampled values between [0, 1), \
    2: Normal distribution around a handcrafted value')
parser.add_argument('--use_syn', action='store_true',
    help='use synthetic input')
parser.add_argument('--opt', type=str, default='adam', choices=['sgd', 'adam'],
    help='Optimizer to use in projected gradient descent')
parser.add_argument('--lr', type=float, default=1e-2,
    help='learning rate')
parser.add_argument('--lr_schedule', type=str, default='lineardecay', choices=['lineardecay', 'fixed', 'linear1cycledrop', 'linear1cycle'],
    help='learning rate schedule')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

schedule_dict = {
    'fixed': lambda x: 1,
    'lineardecay': lambda x: 1.0 - x/args.num_step,
    'linear1cycle': lambda x: (9*(1-np.abs(x/args.num_step-1/2)*2)+1)/10,
    'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*args.num_step)-1/2)*2)+1)/10 if x < 0.9*args.num_step else 1/10 + (x-0.9*args.num_step)/(0.1*args.num_step)*(1/1000-1/10),
}

# Create input data x
if args.use_syn:
    x = torch.tensor(np.random.rand(args.N, args.C).astype('f'), requires_grad=False).to(device)
else:
    f=h5.File(args.data_file, 'r')
    vox = f['voxels'][args.sample_id].reshape(-1,len(f['vox_attr']))
    # KH: 100 micron per index, so divide by 100 to turn index=>cm
    # CL: somehow optimization works much better if divided by 10000 (so x coordinates are normalized to be the same value range as energy) 
    data = np.concatenate(
        (np.expand_dims(vox[:, 3], axis=-1),
         np.expand_dims(vox[:, 0] / 10000.0, axis=-1),
         np.expand_dims(vox[:, 4] / 100.0, axis=-1)),
         axis=1)
    # CL: filter out outragous values and inf
    data = data[data[:, 2] < 300. / 100.0]
    data = data[data[:, 2] > 0.8 / 100.0]

    x = torch.tensor(data.astype('f'), requires_grad=False).to(device)

# Create GT output using GT model parameters
model_gt = SimIonization('generate').to(device)
y_gt = model_gt(x)

# Create a model that tries to recover the GT model
model = SimIonization('learn', args.test_case).to(device)

# Create loss, optimizer, learning rate schedule
criterion = torch.nn.MSELoss(reduction='sum')
if args.opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

schedule_func = schedule_dict[args.lr_schedule]
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_func)

# Training loop
for t in range(args.num_step):
    y = model(x)

    loss = criterion(y, y_gt)
    
    if t % args.print_step == args.print_step - 1 or t == 1:
        print(t, ' loss: ', loss.item(), ' lr: ', scheduler.get_lr()[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()



print("real        parameters: alpha {:.4f}, beta {:.4f}, vdrift {:.4f}, v_lifetime {:.4f}".format(model._alpha, model._beta, model._vdrift, model._lifetime))
print("initialized parameters: alpha {:.4f}, beta {:.4f}, vdrift {:.4f}, v_lifetime {:.4f}".format(model._i_alpha, model._i_beta, model._i_vdrift, model._i_lifetime))
print("optimized   parameters: alpha {:.4f}, beta {:.4f}, vdrift {:.4f}, v_lifetime {:.4f}".format(model._v_alpha, model._v_beta, model._v_vdrift, model._v_lifetime))

