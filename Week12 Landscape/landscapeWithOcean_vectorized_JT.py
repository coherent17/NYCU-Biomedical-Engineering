import numpy as np
from numba import jit

class LandscapeWithOcean(object):
    @jit(forceobj=True)
    def __init__(self,NX,NY):

        # These were all global variables in the matlab code
        self.A = np.zeros((NX,NY))

        self.pool     = np.zeros((NX,NY)) #matrix of pooled areas
        self.drain    = np.zeros((NX,NY)) #matrix of draining points
        self.drainage = np.zeros((NX,NY)) #matrix of drainage points(points connecting drains and pools)
        self.ocean    = np.zeros((NX,NY)) #matrix of ocean points

        self.VOcean = 0.0
        self.AOcean = 0.0
        self.ZBeachLevel = 0.0
        
    def make_periodic_boundaries(self, Z):
        """
        Helper function that produces just the boundary conditions for inspection/use. 
        """
        Z_pbc = np.zeros((np.shape(Z)[1] + 2, np.shape(Z)[0] + 2))
        Z_pbc[1:-1, 1:-1] = Z
        Z_pbc[0, 1:-1] = Z[0,:]
        Z_pbc[-1, 1:-1] = Z[-1,:]
        Z_pbc[1:-1, 0] = Z[:,0]
        Z_pbc[1:-1, -1] = Z[:,-1]
        Z_pbc[0, 0] = Z[-1, 1]
        Z_pbc[0, -1] = Z[-1, 0]
        Z_pbc[-1, 0] = Z[0, -1]
        Z_pbc[-1, -1] = Z[0, 0]
        
        return Z_pbc
    
    def make_RLUD(self, Z):
        """
        Helper function that produces right-left-up-down shifts
        """
        Z_pbc = np.zeros((np.shape(Z)[1] + 2, np.shape(Z)[0] + 2))
        Z_pbc[1:-1, 1:-1] = Z
        Z_pbc[0, 1:-1] = Z[0,:]
        Z_pbc[-1, 1:-1] = Z[-1,:]
        Z_pbc[1:-1, 0] = Z[:,0]
        Z_pbc[1:-1, -1] = Z[:,-1]
        
        Z_R = Z_pbc[1:-1, 2:]
        Z_L = Z_pbc[1:-1, :-2]
        Z_U = Z_pbc[2:, 1:-1]
        Z_D = Z_pbc[:-2, 1:-1]
        
        return Z_R, Z_L, Z_U, Z_D
    
    def make_diag(self, Z):
        """
        Helper function that produces diagonal shifts. 
        """
        Z_pbc = np.zeros((np.shape(Z)[1] + 2, np.shape(Z)[0] + 2))
        
        Z_pbc[1:-1, 1:-1] = Z
        Z_pbc[0, 1:-1] = Z[0,:]
        Z_pbc[-1, 1:-1] = Z[-1,:]
        Z_pbc[1:-1, 0] = Z[:,0]
        Z_pbc[1:-1, -1] = Z[:,-1]
        Z_pbc[0, 0] = Z[-1, 1]
        Z_pbc[0, -1] = Z[-1, 0]
        Z_pbc[-1, 0] = Z[0, -1]
        Z_pbc[-1, -1] = Z[0, 0]
        
        Z_RD = Z_pbc[:-2, 2:]
        Z_RU = Z_pbc[2:, 2:]
        Z_LU = Z_pbc[2:, :-2]
        Z_LD = Z_pbc[:-2, :-2]
        
        return Z_RU, Z_RD, Z_LU, Z_LD
        

    @jit(forceobj=True)
    def ComputeOceanVolumeFromOceanLevelParameter(self, Z ,NX ,NY, oceanLevelParameter):
        Zmin = np.min(Z)
        Zmax = np.max(Z)
        self.ZBeachLevel = Zmin+oceanLevelParameter*(Zmax-Zmin)
        self.VOcean=0.0
        self.AOcean=0
                    
        Z_ocean_ind = np.where(Z <= self.ZBeachLevel)
        add_ocean = np.sum(Z <= self.ZBeachLevel)
        self.VOcean = np.sum(self.ZBeachLevel - Z[Z_ocean_ind])
        self.AOcean = add_ocean
                    
        print('Minimum elevation          ',Zmin)
        print('Maximum elevation          ',Zmax)
        print('Beach level                ',self.ZBeachLevel)
        print('Ocean volume               ',self.VOcean)
        print('Percentage of ocean surface',self.AOcean/(NX*NY)*100)
    @jit(forceobj=True)

    def calculate_collection_area(self, Z, NX, NY):

        flowU = np.zeros((NX,NY)) # there is some flow that goes up UP    from (i,j) to (i,j+1)
        flowD = np.zeros((NX,NY)) # there is some flow that goes up DOWN  from (i,j) to (i,j-1)
        flowL = np.zeros((NX,NY)) # there is some flow that goes up LEFT  from (i,j) to (i-1,j)
        flowR = np.zeros((NX,NY)) # there is some flow that goes up RIGHT from (i,j) to (i+1,j)
        
        self.A = np.zeros((NX, NY)) 
        
        Z_Flow = np.zeros((NX, NY))
        
        Z_R, Z_L, Z_U, Z_D = self.make_RLUD(Z)
        
        A_R, A_L, A_U, A_D = self.make_RLUD(self.A)
        
        # Find differences between each cell and each adjacent cell
        Z_DiffR = Z - Z_R
        Z_R_ind = np.where(Z_DiffR > 0)
        Z_Flow[Z_R_ind] += Z_DiffR[Z_R_ind]

        Z_DiffL = Z - Z_L
        Z_L_ind = np.where(Z_DiffL > 0)
        Z_Flow[Z_L_ind] += Z_DiffL[Z_L_ind]

        Z_DiffU = Z - Z_U
        Z_U_ind = np.where(Z_DiffU > 0)
        Z_Flow[Z_U_ind] += Z_DiffU[Z_U_ind]
        
        Z_DiffD = Z - Z_D
        Z_D_ind = np.where(Z_DiffD > 0)
        Z_Flow[Z_D_ind] += Z_DiffD[Z_D_ind]
        
        A_DiffR = self.A - A_R
        A_DiffL = self.A - A_L
        A_DiffU = self.A - A_U
        A_DiffD = self.A - A_D
        
        Z_DiffR[Z_R_ind] /= Z_Flow[Z_R_ind]
        Z_DiffL[Z_L_ind] /= Z_Flow[Z_L_ind]
        Z_DiffU[Z_U_ind] /= Z_Flow[Z_U_ind]
        Z_DiffD[Z_D_ind] /= Z_Flow[Z_D_ind]
        
        drain_ind = np.where(self.drain > 0)
        pool_ind = np.where(self.pool > 0)
        drainage_ind = np.where(self.drainage > 0)
        pool_drainage_ind = np.where((self.drainage > 0) & (self.pool > 0))
        drain_ind = np.where(self.drain > 0)
        
        ZA_R = Z_DiffR*A_DiffR
        ZA_L = Z_DiffR*A_DiffL
        ZA_U = Z_DiffU*A_DiffU
        ZA_D = Z_DiffD*A_DiffD        
        
        self.A[drain_ind] = (1. + ZA_R[drain_ind] + ZA_L[drain_ind] + ZA_D[drain_ind] + ZA_U[drain_ind])
        self.A[pool_drainage_ind] += (ZA_R[pool_drainage_ind] + 
                                      ZA_L[pool_drainage_ind] + 
                                      ZA_D[pool_drainage_ind] + 
                                      ZA_U[pool_drainage_ind])
        self.A[drainage_ind] = 1. + ZA_R[drain_ind] + ZA_L[drain_ind] + ZA_D[drain_ind] + ZA_U[drain_ind]

    @jit(forceobj=True)
    def pool_check(self, Z, NX, NY):
        # input  Z(current elevation profile)
        # output ZS, drain, pool, drainage

        ###The matrices below have a border of zeros to make checking the boundary easier
        self.pool     = np.zeros((NX,NY)) #matrix of pooled areas
        self.drain    = np.zeros((NX,NY)) #matrix of draining points
        self.drainage = np.zeros((NX,NY)) #matrix of drainage points(points connecting drains and pools)
        self.ocean = np.zeros((NX,NY)) 
        self.AOcean = 0
        
        sub_ocean_points = np.where(Z < self.ZBeachLevel)
        self.AOcean += np.sum(Z < self.ZBeachLevel)
        self.ZBeachLevel = np.max(Z[sub_ocean_points])
        self.ocean[sub_ocean_points] = 1
        
        # We want to know which Z points touch a pool, are a pool, touch drainage, are drainage, etc
        # This means their associated left/right up/down arrays are non zero at those points. 
        # For pools that are not connected to drains or drainage, we say that they are drainage
        # For pools that are connected to drains or drainage, we say they are draining

        pool_R, pool_L, pool_U, pool_D = self.make_RLUD(self.pool)
        drain_R, drain_L, drain_U, drain_D = self.make_RLUD(self.drain)
        drainage_R, drainage_L, drainage_U, drainage_D = self.make_RLUD(self.drainage)
        ocean_R, ocean_L, ocean_U, ocean_D = self.make_RLUD(self.ocean)
        
        p_R = np.where(pool_R > 0)
        P_L = np.where(pool_L > 0)
        p_U = np.where(pool_U > 0)
        P_D = np.where(pool_D > 0)
        
        d_R = np.where(drain_R > 0)
        d_L = np.where(drain_L > 0)
        d_U = np.where(drain_U > 0)
        d_D = np.where(drain_D > 0)
        
        dr_R = np.where(drainage_R > 0)
        dr_L = np.where(drainage_L > 0)
        dr_U = np.where(drainage_U > 0)
        dr_D = np.where(drainage_D > 0)
        
        o_R = np.where(ocean_R > 0)
        o_L = np.where(ocean_L > 0)
        o_U = np.where(ocean_U > 0)
        o_D = np.where(ocean_D > 0)
        
        # Find isolated pools, no drainage, ocean, or drains touching. 
        only_P = np.where( ( (pool_R > 0) & ( (drain_R <= 0) & (drainage_R <= 0) & (ocean_R <= 0) ) ) &
                           ( (pool_L > 0) & ( (drain_L <= 0) & (drainage_L <= 0) & (ocean_L <= 0) ) ) &
                           ( (pool_U > 0) & ( (drain_U <= 0) & (drainage_U <= 0) & (ocean_U <= 0) ) ) &
                           ( (pool_D > 0) & ( (drain_D <= 0) & (drainage_D <= 0) & (ocean_D <= 0) ) ) )
        
        # Find pools that have a drain, drainage, or ocean cell touching
        P_and_or_D = np.where( ( (pool_R > 0) & ( (drain_R > 0) | (drainage_R > 0) | (ocean_R > 0) ) ) |
                               ( (pool_L > 0) & ( (drain_L > 0) | (drainage_L > 0) | (ocean_L > 0) ) ) |
                               ( (pool_U > 0) & ( (drain_U > 0) | (drainage_U > 0) | (ocean_U > 0) ) ) |
                               ( (pool_D > 0) & ( (drain_D > 0) | (drainage_D > 0) | (ocean_D > 0) ) ) )
        
        # Find pools that have a drain, drainage, and ocean cell touching
        P_and_D = np.where( ( (pool_R > 0) & (drainage_R <= 0) & (pool_L > 0) & (drainage_L <= 0) & 
                             (pool_U > 0) & (drainage_U <= 0) & (pool_D > 0) & (drainage_D <= 0) ) &
                           ( (drain_L > 0) | (ocean_L > 0) | (drain_U > 0) | (ocean_U > 0) | 
                            (drain_D > 0) | (ocean_D > 0)) | (drainage_R > 0) | (drainage_L > 0) | 
                           (drainage_U > 0) | (drainage_D > 0) )
        
        # Assign status to each cell
        self.drainage[P_and_or_D] = 1
        self.drain[P_and_D] = 1
        self.pool[only_P] = 1
#             #print(ZBeachLevel,self.VOcean,self.AOcean,np.sum(self.ocean))