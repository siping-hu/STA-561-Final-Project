import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
 
import numpy as np
import matplotlib.pyplot as plt
import os
 
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

}

dataset = 'animals'
train_directory = os.path.join('/home/sh542/workspace/561/project/'+dataset, 'train')
valid_directory = os.path.join('/home/sh542/workspace/561/project/'+dataset, 'valid')
test_directory = os.path.join('/home/sh542/workspace/561/project/'+dataset, 'test')

 
batch_size = 32
num_classes = 10
 
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
 
 
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

 
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)


print(train_data_size, valid_data_size, test_data_size)


resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.requires_grad = False

# child_counter = 0
# for child in resnet50.children:
#     if child_counter>48:
#         for param in child.parameters():
#             param.requires_grad = True
#     child_counter += 1




fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1)
)

resnet50 = resnet50.to('cuda:1')

loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())
# optimizer = optim.Adam(filter(lambda p:p.requires_grad,resnet50.parameters()))



def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    history = []
    best_acc = 0.0
    best_epoch = 0
    fig = plt.figure()
    class_names = ['bear', 'chimp', 'giraffe', 'gorilla', 'llama', 'ostrich', 'porcupine', 'skunk',
                   'triceratops', 'zebra']
 
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
 
        model.train()
 
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
 
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
 
            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
 
            outputs = model(inputs)
 
            loss = loss_function(outputs, labels)
 
            loss.backward()
 
            optimizer.step()
 
            train_loss += loss.item() * inputs.size(0)
 
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
 
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
            train_acc += acc.item() * inputs.size(0)
 
        with torch.no_grad():
            model.eval()
 
            for j, (inputs, labels) in enumerate(valid_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                outputs = model(inputs)
 
                loss = loss_function(outputs, labels)
 
                valid_loss += loss.item() * inputs.size(0)
 
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
 
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
                valid_acc += acc.item() * inputs.size(0)

                # images_so_far = 0
                # if (j == 1):
                #     for k in range(inputs.size()[0]):
                #         if (k<6):
                #             images_so_far +=1
                #             print(images_so_far)
                #             ax = plt.subplot(6//2, 2, images_so_far)
                            
                #             ax.axis('off')
                #             ax.set_title('predicted: {}'.format(class_names[predictions[k]]))
                #             def denormalize(x):
                #                 # x[0] = x[0]*0.529+0.485
                #                 # x[1] = x[1]*0.524+0.456
                #                 # x[2] = x[2]*0.525+0.406
                #                 return (x * 0.5 + 0.5).clip(0, 1)
                #                 # return x.clip(0, 1)
                #             x = inputs.cpu().data[k].numpy().transpose(1, 2, 0)
                #             ax.imshow(denormalize(x))
                        
                #     fig.savefig('/home/sh542/workspace/561/project/save/train-all/predict'+str(epoch)+'.png')
                #     fig.show()
                #     fig.clf()

            for j, (inputs, labels) in enumerate(test_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                outputs = model(inputs)
 
                loss = loss_function(outputs, labels)
 
                test_loss += loss.item() * inputs.size(0)
 
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
 
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
                test_acc += acc.item() * inputs.size(0)

                images_so_far = 0
                if (j == 1):
                    for k in range(inputs.size()[0]):
                        if (k<6):
                            images_so_far +=1
                            print(images_so_far)
                            ax = plt.subplot(6//2, 2, images_so_far)
                            
                            ax.axis('off')
                            ax.set_title('predicted: {}'.format(class_names[predictions[k]]))
                            def denormalize(x):
                                # x[0] = x[0]*0.529+0.485
                                # x[1] = x[1]*0.524+0.456
                                # x[2] = x[2]*0.525+0.406
                                return (x * 0.5 + 0.5).clip(0, 1)
                                # return x.clip(0, 1)
                            x = inputs.cpu().data[k].numpy().transpose(1, 2, 0)
                            ax.imshow(denormalize(x))
                        
                    fig.savefig('/home/sh542/workspace/561/project/save/train-all/predict'+str(epoch)+'.png')
                    fig.show()
                    fig.clf()
 
        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size

        avg_test_loss = test_loss/test_data_size
        avg_test_acc = test_acc/test_data_size
 
        history.append([avg_train_loss, avg_valid_loss, avg_test_loss, avg_train_acc, avg_valid_acc, avg_test_acc])
 
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
 
        epoch_end = time.time()
 
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_valid_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
 
        torch.save(model, '/home/sh542/workspace/561/project/save/train-all/'+dataset+'_model_'+str(epoch+1)+'.pt')
    return model, history




num_epochs = 15
trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
torch.save(history, '/home/sh542/workspace/561/project/save/train-all/'+dataset+'_history.pt')

history = np.array(history)
plt.plot(history[:, 0:3])
plt.legend(['Train Loss', 'Val Loss', 'Test Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('/home/sh542/workspace/561/project/save/train-all/'+dataset+'_loss_curve.png')
plt.show()
plt.clf()
 
plt.plot(history[:, 3:6])
plt.legend(['Train Accuracy', 'Val Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('/home/sh542/workspace/561/project/save/train-all/'+dataset+'_accuracy_curve.png')
plt.show()




