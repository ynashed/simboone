import argparse
import h5py as h5
import numpy as np

import torch
from SimIonization import SimIonizationData

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
parser.add_argument('--use_syn', action='store_true',
    help='use synthetic input')
parser.add_argument('--opt', type=str, default='adam', choices=['sgd', 'adam'],
    help='Optimizer to use in projected gradient descent')
parser.add_argument('--lr', type=float, default=1e-2,
    help='learning rate')
parser.add_argument('--lr_schedule', type=str, default='lineardecay', choices=['lineardecay', 'fixed', 'linear1cycledrop', 'linear1cycle'],
    help='learning rate schedule')
parser.add_argument('--scale', type=float, default=0.1, help='learning rate schedule')
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
    data = np.random.rand(args.N, args.C).astype('f')
    x = torch.tensor(data, requires_grad=False).to(device)
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

# Variables for recovering ground truth x
noise = (np.random.rand(data.shape[0], data.shape[1]) - 0.5) *  args.scale
x_estimate = torch.from_numpy(data + noise.astype('float32')).to(device).detach().requires_grad_(True)

# Create GT output using GT model parameters
model = SimIonizationData(x).to(device)

# Create loss, optimizer, learning rate schedule
criterion = torch.nn.MSELoss(reduction='sum')
if args.opt == 'sgd':
    optimizer = torch.optim.SGD([x_estimate], lr=args.lr)
else:
    optimizer = torch.optim.Adam([x_estimate], lr=args.lr)

schedule_func = schedule_dict[args.lr_schedule]
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_func)


x_start = x_estimate.detach().clone()

# Training loop
for t in range(args.num_step):
    y_estimate = model(x_estimate)

    loss = criterion(y_estimate, model.y_gt)

    if t % args.print_step == args.print_step - 1 or t == 1:
        print(t, ' loss: ', loss.item(), ' lr: ', scheduler.get_lr()[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

print('Ground Truth input x:')
print(model.x_gt)
print('---------------------------------------------')
print('End x:')
print(x_estimate)
print('---------------------------------------------')
print('Start x:')
print(x_start)
print('---------------------------------------------')
print('End mean abs diff:')
print(torch.mean(torch.abs(model.x_gt - x_estimate)))
print('---------------------------------------------')
print('Start mean abs diff:')
print(torch.mean(torch.abs(model.x_gt - x_start)))
print('---------------------------------------------')

