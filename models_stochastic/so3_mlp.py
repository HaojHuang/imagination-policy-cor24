import sys
sys.path.append('../')

import torch
import numpy as np

from escnn import gspaces
from escnn import nn as snn
from escnn import group

import torch.nn as nn

class SO3MLP(snn.EquivariantModule):
    
    def __init__(self, out_dim=3):
        
        super(SO3MLP, self).__init__()
        
        # the model is equivariant to the group SO(3)
        self.G = group.so3_group()
        
        # since we are building an MLP, there is no base-space
        self.gspace = gspaces.no_base_space(self.G)
        
        # the input contains the coordinates of a point in the 3D space
        std_repr = self.G.standard_representation()
        tri_repr = self.G.trivial_representation
        self.in_type = snn.FieldType(self.gspace, 1*[std_repr])
        # self.in_type = nn.FieldType(self.gspace, 3*[tri_repr] + 1*[std_repr] + 2*[tri_repr])
        # self.in_type = self.gspace.type(self.G.standard_representation())
        
        # Layer 1
        # We will use the representation of SO(3) acting on signals over a sphere, bandlimited to frequency 1
        # To apply a point-wise non-linearity (e.g. ELU), we need to sample the spherical signals over a finite number of points.
        # Note that this makes the equivariance only approximate.
        # The representation of SO(3) on spherical signals is technically a quotient representation,
        # identified by the subgroup of planar rotations, which has id=(False, -1) in our library
        
        # N.B.: the first this model is instantiated, the library computes numerically the spherical grids, which can take some time
        # These grids are then cached on disk, so future calls should be considerably faster.
        
        activation1 = snn.QuotientFourierELU(
            self.gspace,
            subgroup_id=(False, -1),
            channels=4, # specify the number of spherical signals in the output features
            irreps=self.G.bl_sphere_representation(L=1).irreps, # include all frequencies up to L=1
            grid=self.G.sphere_grid(type='thomson', N=16), # build a discretization of the sphere containing 16 equally distributed points            
            inplace=True
        )
        
        # map with an equivariant Linear layer to the input expected by the activation function, apply batchnorm and finally the activation
        self.block1 = snn.SequentialModule(
            snn.Linear(self.in_type, activation1.in_type),
            #nn.IIDBatchNorm1d(activation1.in_type),
            activation1,
        )
        
        # Repeat a similar process for a few layers        
        # 8 spherical signals, bandlimited up to frequency 3
        activation2 = snn.QuotientFourierELU(
            self.gspace,
            subgroup_id=(False, -1),
            channels=8, # specify the number of spherical signals in the output features
            irreps=self.G.bl_sphere_representation(L=3).irreps, # include all frequencies up to L=3
            grid=self.G.sphere_grid(type='thomson', N=40), # build a discretization of the sphere containing 40 equally distributed points            
            inplace=True
        )
        self.block2 = snn.SequentialModule(
            snn.Linear(self.block1.out_type, activation2.in_type),
            #nn.IIDBatchNorm1d(activation2.in_type),
            activation2,
        )

        
        activation3 = snn.QuotientFourierELU(
            self.gspace,
            subgroup_id=(False, -1),
            channels=8, # specify the number of spherical signals in the output features
            irreps=self.G.bl_sphere_representation(L=3).irreps, # include all frequencies up to L=3
            grid=self.G.sphere_grid(type='thomson', N=40), # build a discretization of the sphere containing 40 equally distributed points            
            inplace=True
        )
        self.block3 = snn.SequentialModule(
            snn.Linear(self.block2.out_type, activation3.in_type),
            #nn.IIDBatchNorm1d(activation2.in_type),
            activation3,
        )

        # self.in_type = nn.FieldType(self.gspace, 3*[tri_repr] + 1*[std_repr] + 2*[tri_repr])
        self.out_type = snn.FieldType(self.gspace, out_dim*[tri_repr])
        # print(self.out_type)
        self.block_out = snn.Linear(self.block3.out_type, self.out_type)
    
    def forward(self, x: snn.GeometricTensor):
        
        # check the input has the right type
        assert x.type == self.in_type
        x = self.block1(x)
        # print('layer1',x.shape)
        x = self.block2(x)
        # print('layer2',x.shape)
        x = self.block3(x)
        # print('layer3',x.shape)
        x = self.block_out(x)
     
        return x.tensor
    
    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) ==2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape
    
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = SO3MLP().to(device)

# np.set_printoptions(linewidth=10000, precision=4, suppress=True)

# model.eval()

# B = 10

# # generates B random points in 3D and wrap them in a GeometricTensor of the right type
# print(model.in_type)
# x = model.in_type(torch.randn(1, B, 3))


# print('##########################################################################################')
# with torch.no_grad():
#     y = model(x.to(device)).to('cpu')
#     print(y.shape)