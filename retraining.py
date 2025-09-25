import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18,resnet50
import argparse
import os
import random
import numpy as np
from torchvision.utils import save_image
import copy
import datetime
import secrets
import torch.nn.functional as F
import time
import json

def load_dataset(args):
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),     
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), 
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif args.dataset == 'tiny-imagenet-200':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  
            transforms.RandomHorizontalFlip(),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
            transforms.RandomRotation(15),  
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)), 
        ])

        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=args.input_path + args.dataset, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=args.input_path + args.dataset, train=False, download=True, transform=transform_test)
        num_classes = len(train_dataset.classes)
    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=args.input_path + args.dataset, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=args.input_path + args.dataset, train=False, download=True, transform=transform_test)
        num_classes = len(train_dataset.classes)
    elif args.dataset == 'tiny-imagenet-200':
        train_dataset = torchvision.datasets.ImageFolder(root=args.input_path + args.dataset+'/train', transform=transform_train)
        test_dataset = torchvision.datasets.ImageFolder(root=args.input_path + args.dataset+'/val', transform=transform_test)
        classes = os.listdir(args.input_path + args.dataset+'/train')
        num_classes = len(classes)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=2)

    return train_dataset,test_loader,num_classes

def load_model(args, num_classes):
    if args.model == 'resnet18':
        model = resnet18(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif args.model == 'resnet50':
        model = resnet50(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    
    return model

def train(args, train_loader, model, device, optimizer, epoch, bool_change_current):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    temp_list = []
    total_misclassified = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        _, predicted = torch.max(outputs.data, 1)
        incorrect_mask = predicted != targets
        batch_indices = batch_idx * train_loader.batch_size + torch.where(incorrect_mask)[0]
        total_misclassified.extend(batch_indices.cpu().numpy())

    log = f'Epoch: {epoch} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%'
    print(log)
    acc = 100.*correct/total
    return log,acc

def test(test_loader, model, device, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    log = f'Test Epoch: {epoch} | Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%'
    print(log)
    acc = 100.*correct/total
    return log, acc

def prune(args, train_dataset):
    # if args.pr != 0:
    tensor = torch.zeros(len(train_dataset), dtype=torch.bool)
    indices = torch.randperm(len(train_dataset))[:round(len(train_dataset)*(1-args.pr))]
    tensor[indices] = True
    # else:
    #     tensor = None
    #     indices = None
    return tensor,indices

def noise(args, train_dataset, num_classes):
    noise_ratio = args.nr
    num_samples = len(train_dataset)
    num_noisy_samples = int(noise_ratio * num_samples)
    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    for idx in noisy_indices:
        current_label = train_dataset.targets[idx]
        new_label = np.random.choice([i for i in range(num_classes) if i != current_label])
        train_dataset.targets[idx] = new_label

def write_to_file(file_path, file_name, content):
    full_path = os.path.join(file_path, file_name)
    try:
        with open(full_path, 'a', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"Error: {e}")


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            loaded_list = json.load(file)
    except FileNotFoundError:
        print(f" {file_path} not found")
    except json.JSONDecodeError:
        print(f"Failed to parse JSON data in {file_path}. Please ensure the file content is in the correct format.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return loaded_list

def run(args):
    train_dataset,test_loader,num_classes = load_dataset(args)
    model = load_model(args, num_classes)
    
    # optimizer = Lars(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    # torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=5.2,epochs=args.epoch)
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    tensor,indices = prune(args, train_dataset)

    if args.nr != 0:
        noise(args, train_dataset, num_classes)
    

    t = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", t)
    # seed = secrets.randbelow(5000)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # output_file_name = args.dataset + '_' + args.model + '_epoch' + str(args.epoch) + '_lr' + str(args.lr) + '_bs' + str(args.bs) +'_'+ args.pr_basis + '_pr' + str(args.pr) + '_nr'+ str(args.nr) + '_ws' + str(args.ws) + '_' + args.training_type + '_' + args.pr_type +'_seed'+str(seed)
    output_file_name = args.dataset + '_' + args.model + '_epoch' + str(args.epoch) + '_lr' + str(args.lr) + '_bs' + str(args.bs)  + '_pr' + str(args.pr) + '_nr'+ str(args.nr) + '_ws' + str(args.ws) 
    if not os.path.exists(args.output_path+ formatted_time +'/'):
        os.makedirs(args.output_path+ formatted_time +'/')
    if os.path.isfile(args.output_path+ formatted_time +'/' + output_file_name):
        os.remove(args.output_path+ formatted_time +'/' + output_file_name)  

    bool_change_current = False
    current_acc = 0
    max_acc = 0
    
    indices = read_json_file(args.input_correlation_file)
    # print("Start:",indices)
    max_m_indices = np.argsort(indices)[-round(((1-args.pr)*len(train_dataset))):]
    filtered_train_dataset = torch.utils.data.Subset(train_dataset, max_m_indices) 

    train_loader = torch.utils.data.DataLoader(filtered_train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    for epoch in range(args.epoch):
        train_log, acc = train(args,train_loader, model, device, optimizer, epoch, bool_change_current)
        test_log, test_acc = test(test_loader, model, device, epoch)
        write_to_file(args.output_path + formatted_time +'/', output_file_name + formatted_time, str(train_log) +'\n'+ str(test_log) +'\n')
        if test_acc > max_acc:
            max_acc = test_acc
        print("The best acc:", max_acc)
        scheduler.step()
    write_to_file(args.output_path + formatted_time +'/', output_file_name + formatted_time,  str(args.epoch_end) + "'s best acc:" + str(max_acc) +'\n')
    write_to_file(args.output_path , output_file_name, str(args.epoch_end) +'s best acc:'+ str(max_acc) +'\n')
if __name__ == '__main__':
    today = datetime.date.today()
    formatted_date = today.strftime("%Y%m%d")
    parse = argparse.ArgumentParser(description='Dataset Pruning')
    parse.add_argument('--dataset',default='cifar10',type=str, help='dataset: cifar10/cifar100/tiny-imagenet-200')
    parse.add_argument('--model', default='resnet18', type=str, help='model:resnet18/resnet50')
    parse.add_argument('--epoch', default=200, type=int, help='training epoch')
    parse.add_argument('--epoch_end', default=2, type=int, help='training epoch end')
    parse.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parse.add_argument('--bs', default=128, type=int, help='batch size')
    parse.add_argument('--pr', default=0.5, type=float, help='pruning ratio')
    parse.add_argument('--nr', default=0, type=float, help='noise ratio')
    parse.add_argument('--ws', default=10, type=int, help='window_size')   
    parse.add_argument('--device', default=0, type=str, help='GPU device')
    parse.add_argument('--input_path', default='/home/caifei/Project/Datasets/',type=str, help='the path of the dataset')
    parse.add_argument('--input_correlation_file', default='./outputs/suggorate/',type=str, help='the path of the correlation scores')
    # parse.add_argument('--output_path', default='./outputs/retraining/', type=str, help='the path of the result in this expriment')
    parse.add_argument('--output_path', default='./outputs/retraining/' + formatted_date + '/', type=str, help='the path of the result in this expriment')
    args = parse.parse_args()

    run(args)

#
