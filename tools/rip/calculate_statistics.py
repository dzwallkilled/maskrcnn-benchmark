# calculate the statistics of dataset (RIP)
from torch.utils.data.dataloader import DataLoader
from maskrcnn_benchmark.data.datasets.rip import RIPDataset


def calculate_statistics(dataset, verbose=False):
    cnt_dataloder = DataLoader(dataset, batch_size=32, num_workers=16)
    mean, std = 0., 0.
    for idx, (data, _, _) in enumerate(cnt_dataloder):
        if verbose and idx % 50:
            print(f'{idx} images processed.')
        mean += data.mean([2, 3]).sum(0)
        std += data.std([2, 3]).sum(0)

    mean /= len(dataset)  # (127.4730, 127.7317, 109.9307)
    std /= len(dataset)  # (33.1440, 31.7881, 39.5718)

    if verbose:
        print('Total images: ', len(dataset))
        print('Mean: ', mean)
        print('Std: ', std)
    return mean, std


if __name__ == '__main__':
    ann_file = '/data2/data2/zewei/data/RipData/rip_data_train.json'
    root = '/data2/data2/zewei/data/RipData/RipTrainingAllData/'
    dataset = RIPDataset(ann_file, root, remove_images_without_annotations=True, transforms=None)
    mean, std = calculate_statistics(dataset, True)
