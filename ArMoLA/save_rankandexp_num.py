import torch
import json
import os
import argparse
parser = argparse.ArgumentParser(description='None')

parser.add_argument('--dir', help='path of input file directory')
parser.add_argument('--json_name', help='path of exprank_jsons directory')
# parser.add_argument('--budget', help='total number of experts')

args = parser.parse_args()
bin_path = f'{args.dir}/adapter_model.bin'
try:
    adapter_params = torch.load(bin_path, map_location=torch.device('cpu'))
except FileNotFoundError:
    bin_path = f'{args.dir}/best_model/adapter_model.bin'
    adapter_params = torch.load(bin_path, map_location=torch.device('cpu'))

print(bin_path)
rank_dict = {}
expert_dict = {}
for param_name, param_tensor in adapter_params.items():
    if 'expertlora_E_vector' in param_name:
        param_name = '.'.join(param_name.split('.')[2:])
        rank_dict[param_name] = (torch.argmax(param_tensor) + 1).item()
    elif 'router_expert_vector' in param_name:
        param_name = '.'.join(param_name.split('.')[2:-2])
        expert_dict[param_name] = max(2, (torch.argmax(param_tensor) + 1).item())
        # print(param_name, ':', (torch.argmax(param_tensor) + 1).item())

dir_path = f'../exprank_jsons_{args.json_name}'
dir_path = os.path.join(dir_path, args.dir.split('/')[-1])
print(dir_path)
os.makedirs(dir_path, exist_ok=True)
json.dump(rank_dict, open(os.path.join(dir_path, 'rank.json'), 'w', encoding='utf-8'), indent=4)
json.dump(expert_dict, open(os.path.join(dir_path, 'expert.json'), 'w', encoding='utf-8'), indent=4)

