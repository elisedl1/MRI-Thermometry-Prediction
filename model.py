from torch import nn
import torch
#___________________________________UNET CLASS___________________________________________

class UNet(nn.Module):
    def __init__(self, kernel=4, num_filters=64,num_colours=3, num_in_channels=3):
        # first call parent's initialization function
        super(UNet,self).__init__()
       # padding = kernel // 2
        padding=2
        ###
        #_______________________________down sampple_____________________________________
        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel,padding=padding, stride=2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU())
          #  nn.MaxPool2d(2),)
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*2),
            nn.LeakyReLU())

        self.downconv3 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=kernel, padding=padding,stride=2),
            nn.BatchNorm2d(num_filters*4),
            nn.LeakyReLU())
        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=kernel, padding=padding,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        self.rfconv2 = nn.Sequential(
            nn.Conv2d(num_filters*8, num_filters*8, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        
        self.rfconv3 = nn.Sequential(
            nn.Conv2d(num_filters*8, num_filters*8, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        
        #_______________________________up sampple_____________________________________
  
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters*8,num_filters*4, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*4),
            nn.Dropout2d(0.1),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
                nn.ConvTranspose2d(256+512,num_filters*4, kernel_size=kernel,padding=1,stride=2),#, output_padding=1
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
        
        self.upconv3 = nn.Sequential(
                nn.ConvTranspose2d(256+512,num_filters*4, kernel_size=kernel,padding=2,stride=2,output_padding=1),
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
        
        self.upconv4 = nn.Sequential(
                nn.ConvTranspose2d(512,num_filters*4, kernel_size=kernel,padding=2,stride=2,output_padding=1),#, output_padding=1
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
        self.upconv5 = nn.Sequential(
                nn.ConvTranspose2d(256+128,num_filters*4, kernel_size=kernel,padding=1,stride=2),#, output_padding=1
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
        self.upconv6 = nn.Sequential(
                nn.ConvTranspose2d(256+64,num_filters*4, kernel_size=kernel,padding=2,stride=2,output_padding=0),#, output_padding=1
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
               # nn.Tanh())
                nn.Sigmoid())

        self.finalconv =nn.Sequential(
                nn.Conv2d(num_filters*4,1, kernel_size=kernel,padding=padding),
                
               # nn.Tanh())
                nn.ReLU())

    #sigmoid --- bce+dice   --- + binarize    
    

    def forward(self, x):
        #rint(x.shape)
        out1 = self.downconv1(x)
#         print("unet out1L: ", out1)
#         print("min: ", torch.min(out1))
#         print("max: ", torch.max(out1))
        #rint("1", out1.shape)
        out2 = self.downconv2(out1)
        #rint("2", out2.shape)
        out3 = self.downconv3(out2)
        #rint("3", out3.shape)
        out4 = self.rfconv(out3)
        #rint("4", out4.shape)
        out5 = self.rfconv2(out4)
        #rint("5", out5.shape)
        out6 = self.rfconv3(out5)
        #rint("6", out6.shape)
        out7= self.upconv1(out6)
        #rint("8", out7.shape)
        out8 = torch.cat((out5, out7), dim=1)
        #rint("9", out8.shape)
        
        out9 = self.upconv2(out8)
        #rint("2", out9.shape)
        out10=torch.cat((out4, out9), dim=1)
        #rint("3", out9.shape)
        out11= self.upconv3(out10)
        #rint("11",out11.shape)
        out12= torch.cat((out3, out11),dim=1)
        #rint("12",out12.shape)
        
        
        out13=self.upconv4(out12)
        #rint("13",out13.shape)
        out14= torch.cat((out2, out13),dim=1)
        #rint("14",out14.shape)
        
        out15=self.upconv5(out14)
        #rint("15",out15.shape)
        out16= torch.cat((out1, out15),dim=1)
        #rint("16",out16.shape)
        
        out17=self.upconv6(out16)
        out=self.finalconv(out17)
    
        return out
    
    #___________________________________REGRESSION UNET CLASS___________________________________________
    

class regression_UNet(nn.Module):
    def __init__(self, kernel=4, num_filters=64,num_colours=3, num_in_channels=3):
        # first call parent's initialization function
        super(regression_UNet,self).__init__()
       # padding = kernel // 2
        padding=2
        ###
        #_______________________________down sampple_____________________________________
        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel,padding=padding, stride=2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU())
          #  nn.MaxPool2d(2),)
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*2),
            nn.LeakyReLU())
          #  nn.MaxPool2d(2),)

        self.downconv3 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=kernel, padding=padding,stride=2),
            nn.BatchNorm2d(num_filters*4),
            nn.LeakyReLU())
        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=kernel, padding=padding,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        self.rfconv2 = nn.Sequential(
            nn.Conv2d(num_filters*8, num_filters*8, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        
        self.rfconv3 = nn.Sequential(
            nn.Conv2d(num_filters*8, num_filters*8, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        
        #_______________________________up sampple_____________________________________
  
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters*8,num_filters*4, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*4),
            nn.Dropout2d(0.1),
            nn.ReLU())
        self.upconv2 = nn.Sequential(
                nn.ConvTranspose2d(256,num_filters*4, kernel_size=kernel,padding=1,stride=2),#, output_padding=1
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
        
        self.upconv3 = nn.Sequential(
                nn.ConvTranspose2d(256,num_filters*4, kernel_size=kernel,padding=2,stride=2,output_padding=1),
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
        
        self.upconv4 = nn.Sequential(
                nn.ConvTranspose2d(256,num_filters*4, kernel_size=kernel,padding=2,stride=2,output_padding=1),#, output_padding=1
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
        self.upconv5 = nn.Sequential(
                nn.ConvTranspose2d(256,num_filters*4, kernel_size=kernel,padding=1,stride=2),#, output_padding=1
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
        self.upconv6 = nn.Sequential(
                nn.ConvTranspose2d(256,num_filters*4, kernel_size=kernel,padding=2,stride=2,output_padding=0),#, output_padding=1
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
               # nn.Tanh())
                nn.Sigmoid())

        self.finalconv =nn.Sequential(
                nn.Conv2d(num_filters*4,1, kernel_size=kernel,padding=padding),
                
               # nn.Tanh())
                nn.ReLU())

    #sigmoid --- bce+dice   --- + binarize    
    

    def forward(self, x):
        #rint(x.shape)
        out1 = self.downconv1(x)
        out2 = self.downconv2(out1)
        #rint("2", out2.shape)
        out3 = self.downconv3(out2)
        #rint("3", out3.shape)
        out4 = self.rfconv(out3)
        #rint("4", out4.shape)
        out5 = self.rfconv2(out4)
        #rint("5", out5.shape)
        out6 = self.rfconv3(out5)
        #rint("6", out6.shape)
        out7= self.upconv1(out6)
        out9 = self.upconv2(out7)
        out11= self.upconv3(out9)
        out13=self.upconv4(out11)
        out15=self.upconv5(out13)
        out17=self.upconv6(out15)
        out=self.finalconv(out17)

    
        return out
    #___________________________________ATTENTION UNET CLASS___________________________________________
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
    
class AttU_Net(nn.Module):
    def __init__(self, kernel=2, num_filters=64,num_colours=3, num_in_channels=1):
        # first call parent's initialization function
        super(AttU_Net,self).__init__()
       # padding = kernel // 2
        padding=2
        ###
        #_______________________________down sampple_____________________________________
        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=kernel,padding=2, stride=2),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU())
          #  nn.MaxPool2d(2),)
        self.downconv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel, padding=2,stride=2),
            nn.BatchNorm2d(num_filters*2),
            nn.LeakyReLU())
          #  nn.MaxPool2d(2),)

        self.downconv3 = nn.Sequential(
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=kernel, padding=padding,stride=2),
            nn.BatchNorm2d(num_filters*4),
            nn.LeakyReLU())
        self.rfconv = nn.Sequential(
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=kernel, padding=padding,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        self.rfconv2 = nn.Sequential(
            nn.Conv2d(num_filters*8, num_filters*8, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        
        self.rfconv3 = nn.Sequential(
            nn.Conv2d(num_filters*8, num_filters*8, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU())
        
        #_______________________________up sampple_____________________________________
  
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters*8,num_filters*8, kernel_size=kernel, padding=1,stride=2),
            nn.BatchNorm2d(num_filters*8),
            nn.Dropout2d(0.1),
            nn.ReLU())
        self.Att1 = Attention_block(F_g=num_filters*8,F_l=num_filters*8,F_int=num_filters*4)
        self.Up_conv1 = conv_block(ch_in=num_filters*16, ch_out=num_filters*8)
        self.upconv2 = nn.Sequential(
                nn.ConvTranspose2d(num_filters*8,num_filters*8, kernel_size=kernel,padding=1,stride=2),#, output_padding=1
                nn.BatchNorm2d(num_filters*8),
                nn.Dropout2d(0.1),
                nn.ReLU())
        self.Att2 = Attention_block(F_g=num_filters*8,F_l=num_filters*8,F_int=num_filters*4)
        self.Up_conv2 = conv_block(ch_in=num_filters*16 ,ch_out=num_filters*8)
        self.upconv3 = nn.Sequential(
                nn.ConvTranspose2d(num_filters*8,num_filters*4, kernel_size=kernel,padding=2,stride=2,output_padding=1),
                nn.BatchNorm2d(num_filters*4),
                nn.Dropout2d(0.1),
                nn.ReLU())
      #  print(type(num_filters/2))
        self.Att3 = Attention_block(F_g=num_filters*4,F_l=num_filters*4,F_int=num_filters*2)
        self.Up_conv3 = conv_block(ch_in=num_filters*8, ch_out=num_filters*4)
        self.upconv4 = nn.Sequential(
                nn.ConvTranspose2d(num_filters*4,num_filters*2, kernel_size=kernel,padding=2,stride=2,output_padding=1),#, output_padding=1
                nn.BatchNorm2d(num_filters*2),
                nn.Dropout2d(0.1),
                nn.ReLU())
        self.Att4 = Attention_block(F_g=num_filters*2,F_l=num_filters*2,F_int=num_filters)
        self.Up_conv4 = conv_block(ch_in=num_filters*4, ch_out=num_filters*2)
        self.upconv5 = nn.Sequential(
                nn.ConvTranspose2d(num_filters*2,num_filters, kernel_size=kernel,padding=2,output_padding=1,stride=2),#, output_padding=1
                nn.BatchNorm2d(num_filters),
                nn.Dropout2d(0.1),
                nn.ReLU())

        self.upconv6 = nn.Sequential(
                nn.ConvTranspose2d(num_filters,num_filters//2, kernel_size=kernel,padding=2,output_padding=1,stride=2),#, output_padding=1
                nn.BatchNorm2d(num_filters//2),
                nn.Dropout2d(0.1),
                nn.ReLU())
        


        self.finalconv =nn.Sequential(
                nn.Conv2d(num_filters//2,1, kernel_size=1,padding=0),
                
               # nn.Tanh())
                nn.ReLU())

    #sigmoid --- bce+dice   --- + binarize    
    

    def forward(self, x):
        #rint(x.shape)
     #   print("start",x.shape)
        out1 = self.downconv1(x)
#         print("unet out1L: ", out1)
#         print("min: ", torch.min(out1))
#         print("max: ", torch.max(out1))
        #rint("1", out1.shape)
        out2 = self.downconv2(out1)
        #rint("2", out2.shape)
        out3 = self.downconv3(out2)
        #rint("3", out3.shape)
        out4 = self.rfconv(out3)
        #rint("4", out4.shape)
        out5 = self.rfconv2(out4)
        #rint("5", out5.shape)
        out6 = self.rfconv3(out5)
        #rint("6", out6.shape)
   #     print("66",out6.shape)
        out7= self.upconv1(out6)
        #rint("8", out7.shape)
#        out8 = torch.cat((out5, out7), dim=1)
        #rint("9", out8.shape)
    #    print(out7.shape)
     #   print("HI",out6.shape)
        a9=self.Att1(g=out7,x=out5)
      #  out9 = self.upconv1(out7)
        d1 = torch.cat((a9,out7),dim=1)        
        d1 = self.Up_conv1(d1)

        out8= self.upconv2(d1)
        a10=self.Att2(g=out8,x=out4)
     #   out9 = self.upconv2(out7)
        d2 = torch.cat((a10,out8),dim=1)        
        d2 = self.Up_conv2(d2)

      #  print("out99999",d2.shape)
        out9 = self.upconv3(d2)
     #   print("bye",out9.shape,out3.shape)
        a11=self.Att3(g=out9,x=out3)
     #   out9 = self.upconv2(out7)
        d3 = torch.cat((a11,out9),dim=1)        
        d3 = self.Up_conv3(d3)

        out10 = self.upconv4(d3)
        a12=self.Att4(g=out10,x=out2)
        
        d4 = torch.cat((a12,out10),dim=1)        
        d4 = self.Up_conv4(d4)
       # print("last1",d4.shape)
        out11 = self.upconv5(d4)
        #print("last2",out11.shape)
        out12 = self.upconv6(out11)
        out12=self.finalconv(out12)
        return out12#d5
    #___________________________________PIX2PIX CLASS__________________________________________________