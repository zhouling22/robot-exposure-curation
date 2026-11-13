import hydra
import torch
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import numpy as np
import h5py
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from torch.utils.data import DataLoader

def resolve_device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return name

def get_policy_and_cfg(policy_path, device_name='cuda'):
    device = resolve_device(device_name)
    payload = torch.load(open(policy_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg,)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    del workspace

    policy.to(device)
    policy.eval()

    return policy, cfg


def single_ep_loss(single_ep_dataset, policy, device, num_samples=8, num_workers=0):

    dataloader = DataLoader(single_ep_dataset, batch_size=256, num_workers=num_workers,
                            pin_memory=device.startswith('cuda'))

    running_sum = 0.
    num_examples = 0
    with torch.no_grad():
        for _ in range(num_samples):
            it_sum = 0.
            for batch in dataloader:
                num_in_batch = int(batch['obs'].shape[0])
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss = policy.compute_loss(batch)

                running_sum += num_in_batch*loss.item()
                num_examples += num_in_batch
                it_sum += loss.item()*num_in_batch
            print(it_sum)

    return running_sum/num_examples


def compute_episode_weights(dataset_file, policy_path, output, device_name='cuda',
                            num_samples=8, num_workers=0):

    device = resolve_device(device_name)
    policy, cfg = get_policy_and_cfg(policy_path, device)

    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset, dataset_path=dataset_file)
    assert isinstance(dataset, BaseLowdimDataset)

    num_episodes = dataset.replay_buffer.n_episodes

    mean_losses = []
    for ep_idx in tqdm(range(num_episodes)):
        single_ep_dataset = dataset.get_single_ep_dataset(ep_idx)
        mean_loss = single_ep_loss(single_ep_dataset, policy, device,
                                   num_samples=num_samples, num_workers=num_workers)
        mean_losses.append(mean_loss)

    mean_losses = np.array(mean_losses)

    weights = 1/mean_losses
    weights = (weights - weights.min())/weights.std()

    np.savez(output, weights=weights, mean_losses=mean_losses)



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', required=True)
    parser.add_argument('--policy_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', dest='device_name', default='cuda')
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)

    args = parser.parse_args()

    compute_episode_weights(**vars(args))
