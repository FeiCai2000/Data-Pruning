import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
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
    elif args.dataset == 'imagenet':
        # 数据预处理
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    elif args.dataset == 'imagenet':
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
    elif args.model == 'imagenet1k_v1':
        pretrained = False
        model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1000)
    elif args.model == 'resnet34':
        pretrained = False
        model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1000) 
    return model

def train(args, train_loader, model, device, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    temp_list = []
    total_misclassified = []
    total_indices_more = []
    total_indices_less = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        temp_list.extend(loss)
        # current_loss = loss
        loss = loss.mean()
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
    mean = sum(temp_list)/len(temp_list)
    indices = [i for i, x in enumerate(temp_list) if x > mean]
    total_indices_more.extend(indices)
    indices_less = [i for i, x in enumerate(temp_list) if x <= mean]
    total_indices_less.extend(indices_less)
    log = f'Epoch: {epoch} | Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}%'
    print(log)
    acc = 100.*correct/total
    return log, total_indices_more,total_indices_less, acc

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

def balance_prune(args, train_dataset):
    class_indices = {i: [] for i in range(len(train_dataset.classes))}
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    selected_indices = []
    for label, indices in class_indices.items():
        n_keep = round(len(indices)*(1-args.pr)) 
        selected_indices.extend(np.random.choice(indices, n_keep, replace=False))
    print("len:",len(selected_indices))
    tensor = torch.zeros(len(train_dataset), dtype=torch.bool)
    tensor[selected_indices] = True
    return tensor,torch.tensor(selected_indices)

def noise(args, train_dataset, num_classes):
    noise_ratio = args.nr
    num_samples = len(train_dataset)
    num_noisy_samples = int(noise_ratio * num_samples)
    noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
    for idx in noisy_indices:
        current_label = train_dataset.targets[idx]
        new_label = np.random.choice([i for i in range(num_classes) if i != current_label])
        train_dataset.targets[idx] = new_label
    return noisy_indices

def write_to_file(file_path, file_name, content):
    full_path = os.path.join(file_path, file_name)
    try:
        with open(full_path, 'a', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        print(f"Error: {e}")

def write_to_json_file(file_path, file_name, content):
    full_path = os.path.join(file_path, file_name)
    try:
        with open(full_path, 'w') as file:
            json.dump(content, file, indent=4)
    except Exception as e:
        print(f"Error: {e}")

def run(args):
    train_dataset,test_loader,num_classes = load_dataset(args)
    model = load_model(args, num_classes)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    tensor,indices = prune(args, train_dataset)
    # tensor,indices = balance_prune(args, train_dataset)
    # if args.nr != 0:
    #     noise_indices = noise(args, train_dataset, num_classes)
    

    t = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", t)
    
    output_file_name = args.dataset + '_' + args.model + '_epoch' + str(args.epoch) + '_lr' + str(args.lr) + '_bs' + str(args.bs) + '_pr' + str(args.pr) + '_nr'+ str(args.nr)
    if not os.path.exists(args.output_path + formatted_time +'/'):
        os.makedirs(args.output_path + formatted_time +'/')
    if not os.path.exists(args.output_path + formatted_time +'/correlation1/'):
        os.makedirs(args.output_path + formatted_time +'/correlation1/')
    if not os.path.exists(args.output_path + formatted_time +'/correlation2/'):
        os.makedirs(args.output_path + formatted_time +'/correlation2/')
    if not os.path.exists(args.output_path + formatted_time +'/correlation3/'):
        os.makedirs(args.output_path + formatted_time +'/correlation3/')
    if not os.path.exists(args.output_path + formatted_time +'/correlation4/'):
        os.makedirs(args.output_path + formatted_time +'/correlation4/')
    if os.path.isfile(args.output_path + formatted_time +'/' + output_file_name):
        os.remove(args.output_path + formatted_time +'/' + output_file_name)  

    # noise_indices_arr = np.zeros(len(train_dataset), dtype=int)
    # np.add.at(noise_indices_arr, noise_indices, 1)
    # be_learn_arr = np.zeros(len(train_dataset), dtype=int)
    correlation_1_arr = np.zeros(len(train_dataset), dtype=int)
    correlation_2_arr = np.zeros(len(train_dataset), dtype=int)
    correlation_3_arr = np.zeros(len(train_dataset), dtype=int)
    correlation_4_arr = np.zeros(len(train_dataset), dtype=int)
    max_acc = 0
    last_epoch_indices = torch.zeros(round((1-args.pr)*len(train_dataset)))
    last_total_indices = []
    last_be_learn_indices = []
    for epoch in range(args.epoch):
        filtered_train_dataset = torch.utils.data.Subset(train_dataset, indices) 
        print("len:",len(filtered_train_dataset))
        train_loader = torch.utils.data.DataLoader(filtered_train_dataset, batch_size=args.bs, shuffle=False, num_workers=4)
        train_log, total_indices_more,total_indices_less, acc = train(args,train_loader, model, device, optimizer, epoch)    
        test_log, test_acc = test(test_loader, model, device, epoch)
        write_to_file(args.output_path + formatted_time +'/', output_file_name, str(train_log) +'\n'+ str(test_log) +'\n')
        if test_acc > max_acc:
            max_acc = test_acc
        print("The best acc:", max_acc)

        if epoch > 0:
            # correlation_1
            #last epoch samples' loss be low(<loss_mean)
            correlation_1_indices = list(set(last_epoch_indices[last_total_indices_more].tolist()) - set(indices[total_indices_more].tolist()))
            np.add.at(correlation_1_arr, correlation_1_indices, 1)
            output_json_file_name = args.dataset + '_' + args.model + '_epoch' + str(args.epoch) + '_lr' + str(args.lr) + '_bs' + str(args.bs) + '_pr' + str(args.pr) + '_nr'+ str(args.nr) + '_' + str(epoch)
            write_to_json_file(args.output_path + formatted_time +'/correlation1/', output_json_file_name + '_correlation_1_list.json', correlation_1_arr.tolist())
            # correlation_2
            #last epoch samples' loss be high(>=loss_mean)
            correlation_2_indices = list(set(last_epoch_indices[last_total_indices_less].tolist()) - set(indices[total_indices_less].tolist()))
            np.add.at(correlation_2_arr, correlation_2_indices, 1)
            output_json_file_name = args.dataset + '_' + args.model + '_epoch' + str(args.epoch) + '_lr' + str(args.lr) + '_bs' + str(args.bs) + '_pr' + str(args.pr) + '_nr'+ str(args.nr) + '_' + str(epoch)
            write_to_json_file(args.output_path + formatted_time +'/correlation2/', output_json_file_name + '_correlation_2_list.json', correlation_2_arr.tolist())
            # correlation_3
            #last epoch samples' loss be still high(>=loss_mean)
            correlation_3_indices = list(set(last_epoch_indices[last_total_indices_more].tolist()) - set(indices[total_indices_less].tolist()))
            np.add.at(correlation_3_arr, correlation_3_indices, 1)
            output_json_file_name = args.dataset + '_' + args.model + '_epoch' + str(args.epoch) + '_lr' + str(args.lr) + '_bs' + str(args.bs) + '_pr' + str(args.pr) + '_nr'+ str(args.nr) + '_' + str(epoch)
            write_to_json_file(args.output_path + formatted_time +'/correlation3/', output_json_file_name + '_correlation_3_list.json', correlation_3_arr.tolist())
            # correlation_4
            #last epoch samples' loss be still low(<loss_mean)
            correlation_4_indices = list(set(last_epoch_indices[last_total_indices_less].tolist()) - set(indices[total_indices_more].tolist()))
            np.add.at(correlation_4_arr, correlation_4_indices, 1)
            output_json_file_name = args.dataset + '_' + args.model + '_epoch' + str(args.epoch) + '_lr' + str(args.lr) + '_bs' + str(args.bs) + '_pr' + str(args.pr) + '_nr'+ str(args.nr) + '_' + str(epoch)
            write_to_json_file(args.output_path + formatted_time +'/correlation4/', output_json_file_name + '_correlation_4_list.json', correlation_4_arr.tolist())
        last_epoch_indices = indices
        last_total_indices_more  = total_indices_more
        last_total_indices_less  = total_indices_less
        tensor,indices = prune(args, train_dataset)
        
        # tensor,indices = balance_prune(args, train_dataset)
        scheduler.step()
    write_to_file(args.output_path + formatted_time +'/', output_file_name, "The best acc:" + str(max_acc) + '\n')
if __name__ == '__main__':
    today = datetime.date.today()
    formatted_date = today.strftime("%Y%m%d")
    parse = argparse.ArgumentParser(description='Dataset Pruning')
    parse.add_argument('--dataset',default='cifar10',type=str, help='dataset: cifar10/cifar100/tiny-imagenet-200/imagenet')
    parse.add_argument('--model', default='resnet18', type=str, help='model:resnet18/resnet50')
    parse.add_argument('--epoch', default=200, type=int, help='training epoch')
    # parse.add_argument('--epoch_max', default=1, type=int, help='training epoch')
    parse.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parse.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parse.add_argument('--bs', default=128, type=int, help='batch size')
    parse.add_argument('--pr', default=0, type=float, help='pruning ratio')
    parse.add_argument('--nr', default=0, type=float, help='noise ratio')
    parse.add_argument('--ws', default=10, type=int, help='window_size')   
    parse.add_argument('--device', default=0, type=str, help='GPU device')
    parse.add_argument('--input_path', default='/home/caifei/Project/Datasets/',type=str, help='the path of the dataset')
    # parse.add_argument('--output_path', default='./outputs/suggorate/imbalance/', type=str, help='the path of the result in this expriment')
    parse.add_argument('--output_path', default='./outputs/' + formatted_date + '/static/exp/suggorate/', type=str, help='the path of the result in this expriment')
    # parse.add_argument('--output_path', default='./outputs/' + formatted_date + '/why/', type=str, help='the path of the result in this expriment')
    args = parse.parse_args()

    run(args)
