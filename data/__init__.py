import logging
import torch.utils.data

def create_cd_dataloader(dataset, dataset_opt, phase):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_opt['batch_size'],
        shuffle=dataset_opt['use_shuffle'],
        num_workers=dataset_opt['num_workers'],
        pin_memory=True)

def create_cd_dataset(dataset_opt, phase):
    from data.Dataset import CDDataset
    print(dataset_opt["datasetroot"])
    dataset = CDDataset(root_dir=dataset_opt["datasetroot"],
                        resolution=dataset_opt["resolution"],
                        split=phase,
                        data_len=dataset_opt["data_len"]
                        )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s} - {:s}] is created'.format(dataset.__class__.__name__,
                                                                 dataset_opt['name'],
                                                                 phase))
    return dataset
