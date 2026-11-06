import torch, math
from blocks.ResNetBlocks import ResNetDecoderBlock, ResNetEncoderBlock, ResNetBottleneck
import torch.nn.functional as F

class UNET(torch.nn.Module):
    TIME_DIM=128
    
    def sin_embed(self, time):
        i = torch.range(0, self.TIME_DIM/2-1, )
        wI = torch.exp(-((math.log(10000)*i)/(self.TIME_DIM/2)))
        sines = torch.sin(time * wI) 
        cosines = torch.cos(time * wI) 
        return torch.concat([sines, cosines], dim=-1)
        
    def __init__(self) -> None: 
        # Input is 200x200
        
        self.r1 = ResNetEncoderBlock(3, 8, 2, self.TIME_DIM) # 200x200 -> 100x100
        self.r2 = ResNetEncoderBlock(8, 16, 2, self.TIME_DIM)  # 100x100 -> 50x50
        self.r3 = ResNetEncoderBlock(16, 32, 2, self.TIME_DIM) # 50x50 -> 25x25
        self.r4 = ResNetEncoderBlock(32, 64, 2, self.TIME_DIM) # 25x25 -> 13x13
        self.r5 = ResNetEncoderBlock(64, 128, 2, self.TIME_DIM) # 13x13 -> 6x6
        self.r6 = ResNetEncoderBlock(128, 128, 2, self.TIME_DIM) # 6x6 -> 3x3

        self.btl1 = ResNetBottleneck(128, 256, self.TIME_DIM) 
        self.btl2 = ResNetBottleneck(256, 256, self.TIME_DIM) 
        self.btl3 = ResNetBottleneck(256, 128, self.TIME_DIM) 

        self.rt1 = ResNetDecoderBlock(128, 64, 2, self.TIME_DIM, padding=1) # 3x3 -> 6x6
        self.rt2 = ResNetDecoderBlock(64, 32, 2, self.TIME_DIM, padding=1)  # 6x6 -> 13x13
        self.rt3 = ResNetDecoderBlock(32, 32, 2, self.TIME_DIM, padding=1, output_padding=0) # 13x13 -> 25x25
        self.rt4 = ResNetDecoderBlock(32, 16, 2, self.TIME_DIM, padding=1) # 25x25 -> 50x50
        self.rt5 = ResNetDecoderBlock(16, 8, 2, self.TIME_DIM, padding=1) # 50x50 -> 100x100
        self.rt6 = ResNetDecoderBlock(8, 3, 2, self.TIME_DIM, padding=1) # 100x100 -> 200x200

        self.tl1 = torch.nn.Linear(self.TIME_DIM, self.TIME_DIM*2)
        self.tl2 = torch.nn.Linear(self.TIME_DIM*2, self.TIME_DIM)

    def forward(self, x0, time):
        time_emb = self.sin_embed(time)
        t = self.tl2(F.silu(self.tl1(time_emb))) # introduces learnability and non-linearity
        x1 = self.r1(x0, t)
        x2 = self.r2(x1, t)
        x3 = self.r3(x2, t)
        x4 = self.r4(x3, t)
        x5 = self.r5(x4, t)
        x6 = self.r6(x5, t)

        
        