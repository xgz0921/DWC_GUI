from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from zernike import RZern
import gym
import random
import matplotlib.image as imm
import pandas as pd
from scipy.ndimage import zoom
#import cupy as cp
import h5py
from functions import *
scale_factor = (128 / 256, 128 / 256)

co_range = 4 #Random Zernike coefficient range in micrometers
obs_bias = 3 # Observation bias in micrometers
obs_biases = []
cr_range = 2
co_ranges = [2.5,2.5,2.5,2,2,2,2,1,1,1,1,1,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8]
 #Wavefront resolution
n_modes = 9#Number of target modes
co_ranges = co_ranges[0:n_modes].copy()
n_modes_all = 25 #Number of all modes, including higher order noise modes
obs_c_p = np.array([[i/n_modes] for i in range(n_modes+1)])
obs_c_n = np.array([[-i/n_modes] for i in range(n_modes+1)])
def merge_element_wise(arr1, arr2):
    # Using zip to pair elements from both arrays
    merged = [item for pair in zip(arr1, arr2) for item in pair]
    return np.array(merged)
obs_c = merge_element_wise(obs_c_n, obs_c_p)[1:]
#%% Set Zernike coefficient ranges in micrometers.
#Coefficient ranges for training.
# the 6th and 7th radial order Zernike coefficients are simulated for higher order noise.
co_ranges_train = np.array([0.3,0.3,\
                            1,1,1,\
             0.7, 0.7, 0.7, 0.7,\
             0.2, 0.2, 0.2, 0.2, 0.2,\
             0.1,0.1,0.1,0.1,0.1,0.1,\
             0.1,0.1,0.1,0.1,0.1,0.1,0.1])*2
#Decreasing coefficient ranges to simulate mouse eye
co_ranges_decreasing = np.array([2, 2, 2,\
             2, 2, 2, 2,\
              1, 1, 1, 1, 1,\
            0.025,0.025,0.025,0.025,0.025,0.025,
            0.025,0.025,0.025,0.025,0.025,0.025,0.025])

#Uniform coefficient ranges for testing, without the effect of higher order modes.
co_ranges_clean = np.array([2, 2, 2,\
             2, 2, 2, 2,\
              1, 1, 1, 1, 1,\
            0,0,0,0,0,0,
            0,0,0,0,0,0,0])

#Wavefront scale calibration. The scale of the Zernike coefficients for simulating the real system to generate the same wavefront phase shape as the real system.
coefficient_scale = 1
co_scales = np.array([coefficient_scale for _ in range(n_modes_all)]) 

#%% Generate tuples containing mode indices and coefficient ranges & Other useful variables. 
crDM_all = tuple((-cr_range,cr_range,i)for i in range(2,n_modes_all+2))
crDM_obs = tuple((-obs_bias,obs_bias,i)for i in range(2,n_modes+2))
crDM_obs_signs_only = tuple((-1,1,i)for i in range(2,n_modes+2)) #This is for generating observation matrix
abDM_all = tuple((-co_ranges_decreasing[i-2],co_ranges_decreasing[i-2],i)for i in range(2,n_modes_all+2))
abDM = tuple((-co_range,co_range,i)for i in range(2,n_modes+2))

rda = .0,1.0 #Reward dynamic range [0,1]
iobs = 2*len(abDM)+1 #number of observations

#focus_alignment = np.load('focus_alignment_sharpness_normalized_dspd_2.npy')

#%%
def load_images(directory_path):
    with h5py.File(directory_path, 'r') as mat_data:
    # List all datasets in the file
        print(list(mat_data.keys()))
        
        # Access a specific dataset (replace 'your_dataset' with the actual dataset name)
        data1 = np.array(mat_data['processedPatch'])#.reshape((513,512,512))
    # Print the keys of the dictionary to see the structure of the data
    #datax = data1[0]
    data = data1#.reshape((513,512,512))
    data = data.view(np.complex128)[250:400,250:400,0:500]#[100:400,100:400,:]
    data = np.transpose(data, (2,0,1))
    return data#_projection

            
#trgtim = load_images('C:\\Users\\GX\\Documents\\STOC-T\\defocusFitCorrVol_s0006.mat')
#trgtim = load_images('C:\\Users\\GX\\Documents\\STOC-T\\H004-OS-02-FF\\volume_00_267-129-1.mat')
print(os.getlogin())
keywords_slice = ['295','296','297','298','299','300','301','302','303','304','305']
keywords_slice = None#['285','286','287','288','289','290','291','292','293','294','295','296','297','298','299','300','301','302','303','304','305']
#keywords_slice = ['294','295','296']
#keywords_slice = None
keywords_slice = [str(i) for i in range(265,321)]
keywords_slice = [str(i) for i in range(267,276)]
if os.getlogin() == 'GX':
    locs,trgtim = mat_img_load('C:\\Users\\GX\\Documents\\STOC-T\\H004-actually-ai-whole\\',keywords_slice=None,return_loc=True)
    #trgtim = mat_load('C:\\Users\\GX\\Documents\\STOC-T\\40004\\40004\\')
else:
    locs,trgtim = mat_img_load('/Users/gx/Documents/STOC-T/H004-actually-ai-whole/',keywords_slice = None,return_loc=True)

#Fourier domain upsampling
print(len(trgtim))
'''
trgtims = []
for tgt in trgtim:
    temp = np.fft.fftshift(np.fft.fft2(tgt[10:110,10:110]))
    plt.imshow(abs(temp))
    plt.show()
    temp_zp = zero_pad_image(temp,[500,500])
    trgtims.append(np.fft.ifft2(temp_zp))
trgtim = trgtims
'''
#print(locs)
#%%
idx = np.random.randint(0, len(trgtim))
test = trgtim[idx]#[0:70,0:70]
# plt.imshow(abs(test)**2,cmap='gray',vmax = 8e7)
# plt.title(f'{idx}')
# plt.show()
num_imgs = len(trgtim)
trgtim_size = trgtim[0].shape
print(trgtim_size)
#trgtim = np.array(trgtim)
#plt.imshow(np.abs(trgtim[1])**2)

x_s_org = 0
y_s_org = 0
figsize = 128#int(trgtim[0].shape[1]/2)
wfres = 128#figsize#int(x_e_org-x_s_org)#int(trgtim[0].shape[1]/2)
final_size = 128
input_res = 1024
input_range = 30
#%%
class IIDAO_Env(gym.Env):
    class cr: #Correction DM
        class sim:pass
    class ab: #Aberration DM
        class sim:pass
    class dq:pass #Data acquisition
    class sim:pass #Simulated SAO ingredients
    class vr:
        def __init__(self):
            self.psi = []
            self.psf = []
            self.intensity_positive = []
            self.intensity_negative = []
            self.ifft_intensity_positive = []
            self.ifft_intensity_negative = []
            self.ratio_phase = []
            self.ratio_ifft = []
    def __init__(self):
        #Observation & Action Space Definition
        self.observation_space = gym.spaces.Box(np.array([np.full(iobs,mnob) for mnob in [rda[0]]+[ob[0] for ob in crDM_obs]]).T,
												np.array([np.full(iobs,mxob) for mxob in [rda[1]]+[ob[1] for ob in crDM_obs]]).T)

        self.action_space = gym.spaces.Box(np.array([ob[0] for ob in crDM_obs]),
									 np.array([ob[1] for ob in crDM_obs]))
        
        self.mxi = self.mni = 0
        self.cr.c = np.zeros(len(crDM_all))
        self.cr.ar = crDM_all
        self.ab.c = np.zeros(len(abDM_all))
        self.ab.ar = abDM_all
        
        # Define coordinates for wavefront, both illumination and detection path.
        self.xpr_i = np.linspace(-1,1,wfres) 
        self.ypr_i = self.xpr_i

        # Zernike modes
        self.cart_i = RZern(7)
        self.xv_i,self.yv_i = np.meshgrid(self.xpr_i,self.ypr_i)
        self.cart_i.make_cart_grid(self.xv_i, self.yv_i)
        
        # Unit wavefront lists for all modes
        self.sim.wflst_i = [0]*int(self.cart_i.nk)

        #Initialization
        self.SimInit() 

        # Observation matrix components definition
        self.obs_r = np.zeros((iobs,1))
        self.obs_c = obs_c#np.zeros((iobs,1))
        
        # Other parameters
        self.trtem = 0 # Count the steps already performed for each episode.
        self.rststp = 1 # Steps before reset, single step per episode for this scenario.
        self.step_count = 0 # Count total steps experienced
        self.episode_count = 0 # Count number of episodes experienced
        self.rwd_record = [] # Record of rewards during training
        self.wfe_record = [] # Record of wavefront errors during training

        self.rdar = []
        self.corrected_coefficients = np.load('C:\\Users\\GX\\Documents\\STOC-T\\H004-actually-ai-whole\\results_entropy_zmhc_all_patch_6_7m.npy')
    def step(self,action):
        
        self.trtem += 1                                 
        self.step_count+=1
        corrections = np.zeros(n_modes)
        corrections[2:] = np.clip(np.array(action),-co_range,co_range) #Clip the actions to be within effective range.
        self.rwd_temp = self.crSet(corrections) #Apply the actor network's predictions to the DM, correct wavefront aberration, and calculate the reward.

        reward = np.power(self.rwd_temp/self.flat,1)

        rst = (self.trtem >= self.rststp) #End episode when steps exceed the range (only 1 step here since rststp=0)
        
        if rst:
            self.rwd_record.append(reward) #Record the rewards
            
        return [], reward, rst, {}
    
    def reset(self,*n_slice, trgtim_assign = None,aber = None, training = True, ab_mode = 'uniform',obs_bias = obs_bias, new_ab = True,x_s = x_s_org\
              ,y_s = y_s_org, random_patch = False,bias_am = 1,bias_md = 4,pre_cor = None):
        '''
        Reset function of DDPG. 
        training: determine if it is training or not. If training, use uniformly distributed Zernike coefficients to form random aberration.
        ab_mode: random aberration mode for testing, including "decreasing", "normal","uniform", and "nonoise"
        '''

        self.focus = 0
        if n_slice:
            self.n_slice = np.squeeze(n_slice)
        else:
            self.n_slice = np.random.randint(0,num_imgs)
        #self.trgtim = trgtim[128:384,128:384,self.n_slice]
        self.cor_coeff = self.corrected_coefficients[self.n_slice // 16]
        if random_patch:
            self.x_s = 0
            self.y_s = 0
            while locs[self.n_slice]['x'] == 0 or locs[self.n_slice]['y'] == 0:
                self.n_slice = np.random.randint(0, num_imgs)
                #np.random.randint(0,trgtim_size[1]-figsize-1)
                #self.trgtim = trgtim[self.n_slice][self.x_s:self.x_s+figsize,self.y_s:self.y_s+figsize].copy()
            self.trgtim = locs[self.n_slice]['data'][self.x_s:self.x_s + figsize, self.y_s:self.y_s + figsize].copy()

        else:
            self.x_s = x_s
            self.y_s = y_s
            self.trgtim = trgtim[self.n_slice][self.x_s:self.x_s+figsize,self.y_s:self.y_s+figsize].copy()
        if trgtim_assign is not None:
            self.trgtim = trgtim_assign.copy()
        #self.trgtim = trgtim[:,:,self.n_slice]
        #self.trgtim_zp = centralized_padding(self.trgtim, (128,128))

        #self.trgtim = trgtim[:,:,self.n_slice]
        #self.trgtim_fft = np.fft.fft2(self.trgtim)#zoom(np.fft.fft2(self.trgtim),scale_factor,order = 5)
        self.trgtim_zeropad = zero_pad_image(self.trgtim,[wfres,wfres])
        self.trgtim_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.trgtim_zeropad)))
        
        self.trtem = 0 #Set the number of steps performed per episode to zero
        #Different aberration generation schemes, depending on training or not. (customizable)
        #self.focus = focus_alignment[self.n_slice]
        #print(self.focus)
        if new_ab:
            #select a random part of a random image from the test images
            self.max_intensity = 1
            self.ab.c = np.zeros(n_modes_all)
            self.cr.c = np.zeros(n_modes_all)
            self.abSet(focus_correction=False,aberration_correction=False)
            self.raw = np.abs(self.crSet(save_fig=True))**2
            self.raw_org = self.crSet(original=True)
            self.flat_img = np.abs(self.crSet(save_fig = True))**2 #Acquire the reference perfect image (flat wavefront)
            #self.max_intensity = np.max(self.flat_img.flatten()) # Get max image pixel intensity
            #self.flat_img = self.flat_img/self.max_intensity #Normalize perfect image intensity from 0 to 1
            self.flat = self.crSet()#sum(self.flat_img.flatten()**2)#/(sum(self.flat_img.flatten()))**2 #Acquire the perfect image metric (flat wavefront)
            
        if training:
            if new_ab:
            #Uniform aberration amount for all modes is used for training.
                aberrations = np.zeros(n_modes_all)
                self.ab.c[:n_modes] = np.array([np.round(np.random.uniform(-co_ranges_train[md[2]-2],co_ranges_train[md[2]-2]),3) for md in abDM])
                self.ab.c[n_modes:] = np.clip(np.array([np.round(np.random.normal(0,0.4*co_ranges_train[md[2]-2]),3) for md in abDM_all[n_modes:]]),-0.2,0.2)
        else:
            if new_ab:
                #Generate higher order mode noise.
                #self.ab.c[n_modes:] = np.clip(np.array([np.round(np.random.normal(0,0.4*co_ranges_train[md[2]-4]),3) for md in abDM_all[n_modes:]]),-0.025,0.025)
                
                if ab_mode == 'decreasing':#Gaussian distributed coefficient values with decreasing limits
                    self.ab.c[:n_modes] = np.array([np.round(np.clip(np.random.normal(0,0.4*co_ranges_decreasing[md[2]-4]),-2,2),3) for md in abDM])

                elif ab_mode == 'normal':#Gaussian distributed coefficient values with uniform limits
                    self.ab.c[:n_modes] = np.array([np.round(np.random.normal(0,0.4*co_ranges_train[md[2]-4]),3) for md in abDM])
                
                elif ab_mode == 'nonoise':#Uniform distributed coefficient values with uniform limits, higher order mode noise removed.
                    self.ab.c[:n_modes] = np.array([np.round(np.random.normal(0,0.4*co_ranges_decreasing[md[2]-4]),3) for md in abDM_all])
                    self.ab.c[n_modes:] = np.array([0 for md in abDM_all[n_modes:]])
                    
                elif ab_mode == 'uniform':#Uniformly distributed coefficients, same as training.
                    self.ab.c[:n_modes] = np.array([np.round(np.random.uniform(-co_ranges_train[md[2]-4],co_ranges_train[md[2]-4]),3) for md in abDM]) 

        if aber is not None:
            self.ab.c = np.zeros(n_modes_all)
            self.ab.c[:n_modes] = aber
        self.abSet() #Set aberrations on the DM
        #Acquire the metrics for 2N+1 observations: self.obs_r

        self.episode_count += 1 # Record a new episode.
        obs_raw = self.get_observation(bias_md,bias_am)[int(input_res/2-input_range):int(input_res/2+input_range),int(input_res/2-input_range):int(input_res/2+input_range)]

        self.vr.ratio_ifft = obs_raw.copy()
        obs = obs_raw/np.max(obs_raw)
        return obs

    def get_observation(self,md,amplitude):
        intensity_pad_size = 512
        self.cr.c = np.zeros(n_modes_all)
        self.cr.c[md-2] = -amplitude
        self.vr.intensity_negative = self.crSet(return_intensity=True)
        I_1 = zero_pad_image(self.vr.intensity_negative,[intensity_pad_size,intensity_pad_size])
        self.cr.c = np.zeros(n_modes_all)
        self.cr.c[md-2] = amplitude
        self.vr.intensity_positive = self.crSet(return_intensity=True)
        I_2 = zero_pad_image(self.vr.intensity_positive,[intensity_pad_size,intensity_pad_size])
        crop_fft = 100
        #I_1_fft = np.fft.fftshift(np.fft.fft2(I_1))
        self.vr.ifft_intensity_negative = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(I_1)))[int(intensity_pad_size/2-crop_fft):int(intensity_pad_size/2+crop_fft),int(intensity_pad_size/2-crop_fft):int(intensity_pad_size/2+crop_fft)]
        #I_2_fft = np.fft.fftshift(np.fft.fft2(I_2))
        self.vr.ifft_intensity_positive = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(I_2)))[int(intensity_pad_size/2-crop_fft):int(intensity_pad_size/2+crop_fft),int(intensity_pad_size/2-crop_fft):int(intensity_pad_size/2+crop_fft)]
        #ratio = zero_pad_image(I_1_fft/I_2_fft,[input_res,input_res])
        self.vr.ratio_phase = self.vr.ifft_intensity_negative/self.vr.ifft_intensity_positive #I_1_fft/I_2_fft

        ratio = zero_pad_image(self.vr.ratio_phase,[input_res,input_res])

        return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ratio))))**2


    def SimInit(self):
        #Initialization of Zernike polynomials.
        def setzer(n):
            c_i = np.zeros(self.cart_i.nk)
            c_i[n-1] = 1
            self.sim.wflst_i[n] = self.cart_i.eval_grid(c_i,matrix=True)
            
            return
        [setzer(int(md)) for md in list(dict.fromkeys([md[2] for md in crDM_all+abDM_all]))]
        return
    
    def abSet(self,focus_correction = False,aberration_correction = True,corr = None):
        #Set wavefront error, assuming illumination path and collection path have the same wavefront pattern. 
        self.ab.sim.wf_i = sum([self.sim.wflst_i[self.ab.ar[md][2]]*self.ab.c[md] for md in range(len(self.ab.ar))]) #illumination path wavefront
        if focus_correction:
            self.ab.sim.wf_i = self.ab.sim.wf_i.copy()+self.sim.wflst_i[4]*self.focus
        if aberration_correction:
            if corr is not None:
                self.cor_coeff = corr
            else:
                self.cor_coeff = self.corrected_coefficients[self.n_slice//16]
            self.ab.sim.wf_i = self.ab.sim.wf_i.copy()+sum([self.sim.wflst_i[self.ab.ar[md][2]]*self.cor_coeff[md-2] for md in range(2,len(abDM))])

    
    def crSet(self,*mdar,save_fig = False,original = False,return_intensity = False):
        #Set wavefront correction signal (Zernike coefficients) to the DM
        if mdar: #If given a target array of coefficients
            self.cr.c[0:n_modes] = np.squeeze(mdar) #Set the first 12 modes to desired values
            self.cr.c[n_modes:] = np.zeros(n_modes_all-n_modes) # Do not correct for the noise modes.
            
        return self.ZK(self.cr,save_fig=save_fig,original=original,return_intensity=return_intensity)


    def ZK(self,dm,metric = 'sharpnes',save_fig = False, original = False,return_intensity = False):
        '''
        Reward acquisition (Image generation) based on current aberration and correction DM shapes.
        '''
        dm.sim.wf_i = sum([self.sim.wflst_i[dm.ar[md][2]]*dm.c[md]*co_scales[md] for md in range(len(dm.ar))]) #Wavefront superposition, illumination path
        self.psi_i = np.exp(-1j*(self.cr.sim.wf_i+self.ab.sim.wf_i)) #PSI
        #plt.imshow(np.real(psi_i))
        #plt.show()
        self.psi_i[np.isnan(self.psi_i)]=0
        self.vr.psi = self.psi_i.copy()
        #self.vr.psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(zero_pad_image(self.psi_i,[256,256])))))[64:192,64:192]**2
        #psi_i_cp = cp.asarray(psi_i)
        
        corrected = np.multiply(self.trgtim_fft,self.psi_i)
        #corrected = corrected[int(wfres/2-figsize/2):int(wfres/2+figsize/2),int(wfres/2-figsize/2):int(wfres/2+figsize/2)]
        #corrected_cp = self.trgtim_fft_cp*psi_i_cp
        #rdar = np.abs(cp.fft.ifft2(corrected_cp).get())**2
        corrected_padded = zero_pad_image(corrected,[final_size,final_size])
        self.rdar_complex = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(corrected_padded)))#[int(final_size/2-wfres/2):int(final_size/2+wfres/2),int(final_size/2-wfres/2):int(final_size/2+wfres/2)]
        self.rdar_all = np.abs(self.rdar_complex)**2#/self.max_intensity
        self.rdar = self.rdar_all.copy()#[int(wfres/2-figsize/2):int(wfres/2+figsize/2),int(wfres/2-figsize/2):int(wfres/2+figsize/2)]
        if return_intensity:
            return self.rdar
        if original:
            return np.fft.ifft2(corrected)
        #plt.imshow(rdar,vmax = 1e4)
        #plt.show()
        #self.dq.rdar = np.abs(cp.fft.ifft2(corrected_cp).get())**2
        if save_fig: #If this is true, return the actual figure, otherwise, return the metric
            return self.rdar_complex.copy()
        #Return the sharpness metric.
        if metric == 'sharpnes':
            return np.sum(self.rdar**2)/(np.sum(self.rdar))**2
            #return image_quality_Q(self.rdar)
        else:
            return np.sum(self.rdar*np.log(self.rdar))
    
    def expand_dims(self,In):
        # Add an additional dimension to an array.
        m = np.expand_dims(In, axis=0)
        # = np.expand_dims(m, axis=0)
        return m
    
    def entropy(self,img):
        return -sum(img.flatten()*np.log(img.flatten()))
    
    def focus_corr(self,slc):
        slc_fft = np.fft.fft2(slc)
        co = np.linspace(-15,15,31)
        metrics = []
        self.cr.c = np.zeros(n_modes)
        for c in co:
            self.ab.c = np.zeros(n_modes)
            self.ab.c[0] = c
            self.abSet()
            metrics.append(self.crSet())
        
        metrics = np.array(metrics)
        focus_cor = co[np.argmin(metrics)]
        return focus_cor
    
    def upsample_fft(self,In,factor = 2):
        x = In.shape[0]
        y = In.shape[1]
        In_fft = np.fft.fftshift(np.fft.fft2(In))
        In_fft_padded = zero_pad_image(In_fft,[int(factor*x),int(factor*y)])
        return np.fft.ifft2(In_fft_padded)

    def reset_zmhc(self,trgtim_assign, pre_cor=None):

        self.trgtim = trgtim_assign.copy()
        # self.trgtim = trgtim[:,:,self.n_slice]
        # self.trgtim_zp = centralized_padding(self.trgtim, (128,128))

        # self.trgtim = trgtim[:,:,self.n_slice]
        # self.trgtim_fft = np.fft.fft2(self.trgtim)#zoom(np.fft.fft2(self.trgtim),scale_factor,order = 5)
        self.trgtim_zeropad = zero_pad_image(self.trgtim, [wfres, wfres])
        self.trgtim_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.trgtim_zeropad)))
        if pre_cor is not None:
            self.corrected_coefficients = pre_cor
        else:
            self.corrected_coefficients = np.zeros(n_modes)
            # select a random part of a random image from the test images
        self.ab.c = np.zeros(n_modes_all)
        self.cr.c = np.zeros(n_modes_all)
        self.abSet_zmhc(focus_correction=False, aberration_correction=False)
        self.flat_img = np.abs(self.crSet(save_fig=True))**2
        self.flat = self.crSet()

        self.abSet_zmhc()  # Set aberrations on the DM
        return
    def abSet_zmhc(self,focus_correction = False,aberration_correction = True):
        #Set wavefront error, assuming illumination path and collection path have the same wavefront pattern.
        self.ab.sim.wf_i = sum([self.sim.wflst_i[self.ab.ar[md][2]]*self.ab.c[md] for md in range(len(self.ab.ar))]) #illumination path wavefront
        if focus_correction:
            self.ab.sim.wf_i = self.ab.sim.wf_i.copy()+self.sim.wflst_i[4]*self.focus
        if aberration_correction:
            self.ab.sim.wf_i = self.ab.sim.wf_i.copy()+sum([self.sim.wflst_i[self.ab.ar[md][2]]*self.corrected_coefficients[md-2] for md in range(2,len(abDM))])

#%%

        
        
        
# self.psi = []
# self.psf = []
# self.intensity_positive = []
# self.intensity_negative = []
# self.ifft_intensity_positive = []
# self.ifft_intensity_negative = []
# self.ratio_phase = []
# self.ratio_ifft = []