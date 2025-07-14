import torchvision

dataset_transfroms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.Country211(root="./dataset",
                                            train=False,
                                            transform=dataset_transfroms,
                                            download=True)

print()