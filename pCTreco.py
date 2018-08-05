# Andriy Zatserklyaniy <zatserkl@gmail.com> Aug 3, 2018

"""
Working example of Machine Learning approach to pCT image reconstruction.
"""

import numpy as np
import matplotlib.pyplot as plt

# % matplotlib inline


class Track:
    """
    Track consists of measured energy Edet and list of voxels' coordinates.
    The coordinates are the same for voxel weight and for voxel derivatives.
    Class Track has a global pointer to Phantom and to the Phantom filled
    by derivatives.
    """
    phantom_w = None
    phantom_d = None
    Einc = None  # incident energy

    def __init__(self, Edet, voxels):
        self.Edet = Edet  # Energy detector meausurement for this event
        self.voxels = voxels[:]  # use list copy

    def __repr__(self):
        s = 'Edet ' + str(Edet) + ' voxels'
        for voxel in self.voxels:
            s += ' ' + str(voxel)
        return s
    
    def loss(self):
        """
        --> Debug-purpose function
        Just calculates loss: not terrible useful w/o derivatives
        :return: loss function J
        """
        dE = self.Einc - self.Edet
        for voxel in self.voxels:
            row, col = voxel
            dE -= Track.phantom_w[row, col]
        return dE
    
    def loss_der(self):
        """
        --> Mainstream function
        Calculates the loss function and modifies derivatives
        :return: loss function
        NB: modifies phantom_d passed by references
        """
        dE = self.Einc - self.Edet
        for voxel in self.voxels:
            row, col = voxel
            dE -= Track.phantom_w[row, col]

        # modify derivative for this voxel
        for voxel in self.voxels:
            Track.phantom_d[row, col] = -dE  # minus because loss = (y - h(w))
        return dE

    def print(self):
        print(self.Edet, end='\t')
        for row_col in self.voxels:
            row, col = row_col
            # print('Track.phantom_w[row, col] =', Track.phantom_w[row, col])
            # print(Track.phantom_w[row, col], end='\t')  # print voxel value
            print(row_col, end='\t')                      # voxel coordinates
        print()


def generate_horizontal_rays(N):
    rays = []
    for row in range(N):
        ray = []
        for col in range(N):
            ray.append((row, col))
        rays.append(ray)
    return rays


def generate_vertical_rays(N):
    rays = []
    for col in range(N):
        ray = []
        for row in range(N):
            ray.append((row, col))
        rays.append(ray)
    return rays


def generate_diagonal_rays(N):
    """NB that the second coordinate in POSITIVE and NEGATIVE pairs is the same.
    The first one in NEGATIVE is related to first in POSITIVE as 4-x
    """

    # Generate positive slope rays

    rays_pos = []
    for row in range(N):
        i, j = row, 0
        ray_top = []
        ray_bot = []
        for k in range(row+1):
            # print((i-k, j+k), 'bottom:', (N-1-(j+k), N-1-(i-k)))
            ray_top.append((i-k, j+k))
            ray_bot.append((N-1-(j+k), N-1-(i-k)))
        # print()
        rays_pos.append(ray_top)
        rays_pos.append(ray_bot)

    # remove last ray - it's a duplicate of previous
    del rays_pos[-1]

    """Create negative slope rays changing the x-coordinate to (N-1) - x
    """

    rays_neg = []
    for ray_pos in rays_pos:
        ray_neg = []
        for cell in ray_pos:
            # append tuple with the first element changed to (N-1) - x
            ray_neg.append((N - 1 - cell[0], cell[1]))
        rays_neg.append(ray_neg)

    rays = rays_pos + rays_neg
    return rays


def generate_rays(N):
    rays = generate_horizontal_rays(N) +\
        generate_vertical_rays(N) +\
        generate_diagonal_rays(N)
    return rays


def print_phantom(phantom):
    for row in range(phantom.shape[0]):
        for col in range(phantom.shape[1]):
            print('{:8.2f}'.format(phantom[row, col]), end=' ')
        print()


def costFunction_calculate(tracks, phantom_w):
    """
    Calculates cost function, does not compute derivatives
    """
    J = 0.  # cost function

    for track in tracks:
        loss = track.loss()  # does not compute derivatives
        J += loss * loss

    return J / len(tracks) / 2


def costFunction(tracks, phantom_w, phantom_d):
    """
    Returns cost function.
    The derivatives phantom_d will be changed in-place as passed by reference
    """
    J = 0.  # cost function
    # clear derivatives
    phantom_d = np.zeros((phantom_d.shape[0], phantom_d.shape[1]))

    for track in tracks:
        loss = track.loss_der()  # calcs loss function J and modifies phantom_d
        J += loss * loss

    phantom_d /= len(tracks)
    return J / len(tracks) / 2


if __name__ == '__main__':

    N = 2
    phantom_v = np.zeros((N, N))  # true values
    phantom_w = np.zeros((N, N))  # weights
    phantom_d = np.zeros((N, N))  # derivatives
    
    # incident energy
    Einc = 100  # MeV

    #
    # initialize Track global variables
    #
    Track.phantom_w = phantom_w
    Track.phantom_d = phantom_d
    Track.Einc = Einc

    #
    # set the phantom density (in terms of energy loss)
    #
    phantom_v[1, 1] = 10  # loss 10 MeV
    phantom_v[1, 0] = 3   # loss 3 MeV
    phantom_v[0, 0] = 5   # loss 5 MeV

    # print('"real" phantom\n')
    # print_phantom(phantom_v)

    # Generate rays and create tracks
    tracks = []

    generated_rays = generate_rays(N)
    for ray in generated_rays:
        tracks.append(Track(Einc, ray))

    # measure the energy
    for track in tracks:
        Elost = 0
        for voxel in track.voxels:
            row, col = voxel
            Elost += phantom_v[row, col]
        track.Edet = Einc - Elost

    print('\nMeasured tracks:')

    for track in tracks:
        track.print()


    ###############################################


    cost = []
    cost.append(costFunction_calculate(tracks, phantom_w))

    iter_max = 3
    for iter in range(1, iter_max+1):
        print('\n--------------- iteration', iter, '---------------')

        Ji = costFunction(tracks, phantom_w, phantom_d)
        print('\nInitial Cost function Ji =', Ji)

        print('\n"real" phantom\n')
        print_phantom(phantom_v)

        # print('\nInitial "weights" phantom: should be empty\n')
        # print_phantom(phantom_w)

        print('\n"derivatives" phantom\n')
        print_phantom(phantom_d)

        # introduce the correction
        phantom_w -= phantom_d

        print('\n"weights" phantom after correction\n')
        print_phantom(phantom_w)

        Jf = costFunction_calculate(tracks, phantom_w)
        print('\nFinal Cost function Jf =', Jf)
        cost.append(Jf)

    plt.figure()
    plt.plot(cost, marker='o', linestyle='None')
    plt.grid()
    plt.show()
