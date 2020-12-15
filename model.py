import random
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import PIL.ImageOps
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
    	euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
    	loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

    	return loss_contrastive

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(pl.LightningModule):
    def __init__(self, margin, learning_rate, resize, imageFolderTrain, imageFolderTest, batch_size, should_invert):
        super().__init__()
        self.imageFolderTrain = imageFolderTrain
        self.imageFolderTest= imageFolderTest
        self.learning_rate = learning_rate
        self.criterion = ContrastiveLoss(margin=margin)
        self.batch_size = batch_size
        self.should_invert = should_invert
        self.transform = transforms.Compose([transforms.Resize((resize,resize)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor()])
        
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))
    
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    
    def training_step(self, batch, batch_idx):
        x0, x1 , y = batch
        output1,output2 = self(x0, x1)
        loss = self.criterion(output1,output2, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x0, x1 , y = batch
        output1,output2 = self(x0, x1)
        loss = self.criterion(output1,output2,y)

        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
            
    def prepare_data(self):
        self.DatasetFolder = dset.ImageFolder(self.imageFolderTrain)
        self.DatasetFolder_testing = dset.ImageFolder(self.imageFolderTest)
    
    def setup(self, stage=None):
        
        self.siamese_dataset_train = SiameseNetworkDataset(imageFolderDataset=self.DatasetFolder,
                                        transform=self.transform
                                       ,should_invert=self.should_invert)
        self.siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=self.DatasetFolder_testing,
                                        transform=self.transform
                                       ,should_invert=self.should_invert)

    def train_dataloader(self):
        return DataLoader(self.siamese_dataset_train, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.siamese_dataset_test, batch_size=self.batch_size)

if __name__=='__main__':

	parser = argparse.ArgumentParser(
        description='Siamese Network - Face Recognition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gpus', default=1, type=int)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--pretrain_epochs', default=5000, type=int)
	parser.add_argument('--margin', default=1.0, type=float)
	parser.add_argument('--should_invert', default=False)
	parser.add_argument('--imageFolderTrain', default=None)
	parser.add_argument('--imageFolderTest', default=None)
	parser.add_argument('--learning_rate', default=2e-2, type=float)
	parser.add_argument('--resize', default=100, type=int)


	args = parser.parse_args()
	print(args)

	model = SiameseNetwork(margin= args.margin, learning_rate=args.learning_rate, resize=args.resize, imageFolderTrain=args.imageFolderTrain,
							imageFolderTest=args.imageFolderTest, batch_size=args.batch_size, should_invert=args.should_invert)
	trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.pretrain_epochs, progress_bar_refresh_rate=20)
	trainer.fit(model)
	trainer.save_checkpoint("siamese_face_recognition.ckpt")
	trainer.test()