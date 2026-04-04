import torch
from encoding.datasets import get_dataset
from torchvision import transforms

def test_dataset():
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    data_kwargs = {'transform': input_transform, 'base_size': 520, 'crop_size': 480}
    trainset = get_dataset('citys', split='train', mode='train', **data_kwargs)
    print('Dataset objects created successfully')
    print('Number of images:', len(trainset))
    img, mask = trainset[0]
    print('Image shape:', img.shape)
    print('Mask shape:', mask.shape)

if __name__ == "__main__":
    test_dataset()
