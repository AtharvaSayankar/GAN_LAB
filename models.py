import torch
import torch.nn as nn

class dcgan_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class dcgan_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)
    
class cgan_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 100)

        self.init_size = 64 // 4

        self.l1 = nn.Sequential(
            nn.Linear(100 * 2, 128 * self.init_size * self.init_size)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128,0.8),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.8),
            nn.ReLU(True),

            nn.Conv2d(64,3,3,stride=1,padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat((z,c),dim=1)

        out = self.l1(x)
        out = out.view(out.shape[0],128,self.init_size,self.init_size)
        img = self.conv_blocks(out)

        return img

class cgan_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 64*64)

        self.model = nn.Sequential(

            nn.Conv2d(4,64,4,2,1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128*(64//4)*(64//4),1),
            nn.Sigmoid()
        )

    def forward(self,img,labels):

        label = self.label_emb(labels)
        label = label.view(labels.size(0),1,64,64)

        x = torch.cat([img,label],1)

        return self.model(x)
