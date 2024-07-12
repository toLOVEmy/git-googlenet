import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import GoogLeNet

import time

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # get data root path
    image_path = os.path.join(data_root, "SCUT-FBP5500_1", "SCUT-FBP5500")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])  # windows下线程设为0就行，linux下可以设为8
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.4)

    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)

    epochs = 60
    best_acc = 0.0
    save_path = r'E:\xwc\pycharm_virtual_cnn\SCUT-FBP5500_1\SCUT-FBP5500\FBPgoogleNet.pth'
    train_steps = len(train_loader)

    # Log file path
    log_file = 'training_log.txt'

    # Check if log file exists
    if os.path.exists(log_file):
        with open(log_file, 'a') as f:
            f.write('\nResuming Training...\n')
    else:
        with open(log_file, 'w') as f:
            f.write('Epoch,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy,Time Taken\n')

    train_start_time = time.perf_counter()  # 记录训练开始时间，用于计算训练用时

    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        # train
        net.train()
        running_loss = 0.0
        correct_train = 0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct_train += torch.sum(preds == labels.to(device)).item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_loss = running_loss / train_steps
        train_accuracy = correct_train / train_num

        # 更新学习率
        scheduler.step()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_loss = val_loss / len(validate_loader)
        val_accurate = acc / val_num

        epoch_end_time = time.perf_counter()
        time_taken = epoch_end_time - epoch_start_time

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f  time_taken: %.3f sec' %
              (epoch + 1, train_loss, val_loss, train_accuracy, val_accurate, time_taken))

        # Write to log file
        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1},{train_loss:.3f},{val_loss:.3f},{train_accuracy:.3f},{val_accurate:.3f},{time_taken:.3f}\n")

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    train_end_time = time.perf_counter()
    elapsed_time = train_end_time - train_start_time
    minute_time = elapsed_time // 60
    second_time = elapsed_time % 60
    print(f"代码执行耗时：{minute_time}分{second_time}秒")

if __name__ == '__main__':
    main()
