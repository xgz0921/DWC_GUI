from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from zernike import RZern
from scipy.ndimage import zoom
from PIL import Image
import h5py
from functions import *

#%%
class IIDAO_Env():
    class cr: #Correction DM
        class sim:pass
    class ab: #Aberration DM
        class sim:pass
    class dq:pass #Data acquisition
    class sim:pass #Simulated SAO ingredients
    class vr:
        def __init__(self):
            self.psi = []
            self.psf = np.zeros(self.wfres,self.wfres)
            self.intensity_positive = []
            self.intensity_negative = []
            self.ifft_intensity_positive = []
            self.ifft_intensity_negative = []
            self.ratio_phase = []
            self.ratio_ifft = []
    def __init__(self,trgtim,n_modes,zero_padding = None):
        #Observation & Action Space Definition
        #self.DM_all = tuple((-1,1,i)for i in range(n_modes))
        self.n_modes = n_modes
        self.trgtim = trgtim
        if zero_padding is not None:
            self.trgtim_zeropad = zero_pad_image(self.trgtim, [zero_padding, zero_padding])
            self.trgtim_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.trgtim_zeropad)))
            self.wfres = zero_padding
        else:
            self.trgtim_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.trgtim)))
            self.wfres = self.trgtim.shape[0]

        self.cr.c = np.zeros(n_modes)
        self.cr.ar = tuple((-1,1,i+2)for i in range(n_modes))
        self.ab.c = np.zeros(n_modes)
        self.ab.ar = tuple((-1,1,i+2)for i in range(n_modes))
        
        # Define coordinates for wavefront, both illumination and detection path.
        self.xpr_i = np.linspace(-0.8,0.8,self.wfres)
        self.ypr_i = self.xpr_i

        # Zernike modes
        self.cart_i = RZern(7)
        self.xv_i,self.yv_i = np.meshgrid(self.xpr_i,self.ypr_i)
        self.cart_i.make_cart_grid(self.xv_i, self.yv_i)
        
        # Unit wavefront lists for all modes
        self.sim.wflst_i = [0]*int(self.cart_i.nk)

        #Initialization
        self.SimInit()

        self.sharpness_org = 1
        self.entropy_org = 1
        self.img_complex = np.zeros((self.wfres,self.wfres))
        self.img_intensity = np.abs(self.img_complex)**2


        self.ab.c = np.zeros(n_modes)
        self.abSet()
        self.cr.c = np.zeros(n_modes)
        self.img_intensity = self.crSet(return_intensity=True)
        zoomed = zoom(self.img_intensity,2,order=1)
        self.mx = np.max(zoomed)

        self.img_intensity_display = self.zoom_and_convert_to_unit8(self.img_intensity)
        self.sharpness_org = self.sharpness(self.img_intensity)
        self.entropy_org = self.entropy(self.img_intensity)
        self.sharpness_metric = 1
        self.entropy_metric = 1

    def get_observation(self,md,amplitude):
        intensity_pad_size = 1024
        input_res = 1024
        input_range = 32
        self.ab.c = np.zeros(self.n_modes)
        self.ab.c[md-2] = -amplitude
        self.abSet()
        self.vr.intensity_negative = self.crSet(return_intensity=True)
        I_1 = zero_pad_image(self.vr.intensity_negative,[intensity_pad_size,intensity_pad_size])
        self.ab.c = np.zeros(self.n_modes)
        self.ab.c[md-2] = amplitude
        self.abSet()
        self.vr.intensity_positive = self.crSet(return_intensity=True)
        I_2 = zero_pad_image(self.vr.intensity_positive,[intensity_pad_size,intensity_pad_size])
        crop_fft = 150
        #I_1_fft = np.fft.fftshift(np.fft.fft2(I_1))
        self.vr.ifft_intensity_negative = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(I_1)))[int(intensity_pad_size/2-crop_fft):int(intensity_pad_size/2+crop_fft),int(intensity_pad_size/2-crop_fft):int(intensity_pad_size/2+crop_fft)]
        #I_2_fft = np.fft.fftshift(np.fft.fft2(I_2))
        self.vr.ifft_intensity_positive = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(I_2)))[int(intensity_pad_size/2-crop_fft):int(intensity_pad_size/2+crop_fft),int(intensity_pad_size/2-crop_fft):int(intensity_pad_size/2+crop_fft)]
        #ratio = zero_pad_image(I_1_fft/I_2_fft,[input_res,input_res])
        self.vr.ratio_phase = self.vr.ifft_intensity_negative/self.vr.ifft_intensity_positive #I_1_fft/I_2_fft

        ratio = zero_pad_image(self.vr.ratio_phase,[1024,1024])
        obs = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ratio))))**2
        feature_map = obs[int(input_res/2-input_range):int(input_res/2+input_range),int(input_res/2-input_range):int(input_res/2+input_range)]

        self.ab.c = np.zeros(self.n_modes)
        self.abSet()
        return zoom(feature_map,2,order=1)

    def SimInit(self):
        #Initialization of Zernike polynomials.
        def setzer(n):
            c_i = np.zeros(self.cart_i.nk)
            c_i[n-1] = 1
            self.sim.wflst_i[n] = self.cart_i.eval_grid(c_i,matrix=True)
            
            return
        [setzer(int(md)) for md in list(dict.fromkeys([md[2] for md in self.ab.ar+self.cr.ar]))]
        return
    
    def abSet(self):
        #Set wavefront error, assuming illumination path and collection path have the same wavefront pattern. 
        self.ab.sim.wf_i = sum([self.sim.wflst_i[self.ab.ar[md][2]]*self.ab.c[md] for md in range(len(self.ab.ar))]) #illumination path wavefront
        return

    
    def crSet(self,*mdar,return_intensity = False):
        #Set wavefront correction signal (Zernike coefficients) to the DM
        if mdar: #If given a target array of coefficients
            self.cr.c[0:self.n_modes] = np.squeeze(mdar) #Set the first 12 modes to desired values
        return self.ZK(self.cr,return_intensity=return_intensity)


    def ZK(self,dm,return_intensity = False):
        '''
        Reward acquisition (Image generation) based on current aberration and correction DM shapes.
        '''
        dm.sim.wf_i = sum([self.sim.wflst_i[dm.ar[md][2]]*dm.c[md] for md in range(len(dm.ar))]) #Wavefront superposition, illumination path
        self.psi_i = np.exp(-1j*(self.cr.sim.wf_i+self.ab.sim.wf_i)) #PSI

        self.psi_i[np.isnan(self.psi_i)]=0
        temp_psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(zero_pad_image(self.psi_i,[3*self.wfres,3*self.wfres])))))**2
        self.vr.psf = self.convert_image_to_PIL(temp_psf[int(self.wfres): int(2*self.wfres),int(self.wfres): int(2*self.wfres)])
        self.vr.psi = np.real(self.psi_i.copy())

        corrected = np.multiply(self.trgtim_fft,self.psi_i)
        self.img_complex = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(corrected)))#[int(final_size/2-wfres/2):int(final_size/2+wfres/2),int(final_size/2-wfres/2):int(final_size/2+wfres/2)]
        if return_intensity:
            return np.abs(self.img_complex)**2
        self.img_intensity = np.abs(self.img_complex)**2
        self.img_intensity_display = self.zoom_and_convert_to_unit8(self.img_intensity)
        self.sharpness_current = self.sharpness(self.img_intensity)
        self.entropy_current = self.entropy(self.img_intensity)
        self.sharpness_metric = np.round(self.sharpness_current/self.sharpness_org,4)
        self.entropy_metric = np.round(self.entropy_current/self.entropy_org,4)
        return

    def sharpness(self,In):
        return np.sum(In**2)/np.sum(In)**2
    
    def entropy(self,In):
        return -np.sum(In*np.log(In))

    def zoom_and_convert_to_unit8(self,Img):
        Img = zoom(Img, 4, order=1)
        Img = Img/(1.1*self.mx)*255
        #print(Img)
        if Img.dtype != np.uint8:
            arr = np.clip(Img, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="L")  # grayscale

        return pil_img

    def convert_image_to_PIL(self,arr,eps = 1e-12):
        # load new image into env
        arr = zoom(arr, 2, order=1)
        arr = arr/np.max(arr+eps)*255
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        display_img = Image.fromarray(arr, mode="L")
        return display_img
    # def reset_zmhc(self,trgtim_assign, pre_cor=None):
    #
    #     self.trgtim = trgtim_assign.copy()
    #     # self.trgtim = trgtim[:,:,self.n_slice]
    #     # self.trgtim_zp = centralized_padding(self.trgtim, (128,128))
    #
    #     # self.trgtim = trgtim[:,:,self.n_slice]
    #     # self.trgtim_fft = np.fft.fft2(self.trgtim)#zoom(np.fft.fft2(self.trgtim),scale_factor,order = 5)
    #     self.trgtim_zeropad = zero_pad_image(self.trgtim, [wfres, wfres])
    #     self.trgtim_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.trgtim_zeropad)))
    #     if pre_cor is not None:
    #         self.corrected_coefficients = pre_cor
    #     else:
    #         self.corrected_coefficients = np.zeros(n_modes)
    #         # select a random part of a random image from the test images
    #     self.ab.c = np.zeros(n_modes_all)
    #     self.cr.c = np.zeros(n_modes_all)
    #     self.abSet_zmhc(focus_correction=False, aberration_correction=False)
    #     self.flat_img = np.abs(self.crSet(save_fig=True))**2
    #     self.flat = self.crSet()
    #
    #     self.abSet_zmhc()  # Set aberrations on the DM
    #     return
    # def abSet_zmhc(self,focus_correction = False,aberration_correction = True):
    #     #Set wavefront error, assuming illumination path and collection path have the same wavefront pattern.
    #     self.ab.sim.wf_i = sum([self.sim.wflst_i[self.ab.ar[md][2]]*self.ab.c[md] for md in range(len(self.ab.ar))]) #illumination path wavefront
    #     if focus_correction:
    #         self.ab.sim.wf_i = self.ab.sim.wf_i.copy()+self.sim.wflst_i[4]*self.focus
    #     if aberration_correction:
    #         self.ab.sim.wf_i = self.ab.sim.wf_i.copy()+sum([self.sim.wflst_i[self.ab.ar[md][2]]*self.corrected_coefficients[md-2] for md in range(2,len(abDM))])

