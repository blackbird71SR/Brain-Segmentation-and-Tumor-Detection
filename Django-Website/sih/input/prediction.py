import numpy as np
import nibabel as nib
import cv2 as cv
import torch
from torch.utils import data
from torchvision.transforms import transforms
from .data_loader.preprocess import readVol,to_uint8,IR_to_uint8,histeq,preprocessed,get_stacked,rotate,calc_crop_region,calc_max_region_list,crop,get_edge
import SimpleITK as sitk
import scipy.spatial
import difflib

import os
import argparse
from torch.autograd import Variable
from .models_x.fcn_xu import fcn_mul
# use_cuda = False

class MR18loader_test(data.Dataset):
    def __init__(self,T1_path,IR_path,T2_path,is_transform,is_crop,is_hist,forest):
        self.T1_path=T1_path
        self.IR_path=IR_path
        self.T2_path=T2_path
        self.is_transform=is_transform
        self.is_crop=is_crop
        self.is_hist=is_hist
        self.forest=forest
        self.n_classes=11 
        self.T1mean=0.0
        self.IRmean=0.0
        self.T2mean=0.0
        #read data
        T1_nii=to_uint8(readVol(self.T1_path))
        IR_nii=IR_to_uint8(readVol(self.IR_path))
        T2_nii=to_uint8(readVol(self.T2_path))
        #histeq
        if self.is_hist:
            T1_nii=histeq(T1_nii)
        #stack 
        T1_stack_list=get_stacked(T1_nii,self.forest)
        IR_stack_list=get_stacked(IR_nii,self.forest)
        T2_stack_list=get_stacked(T2_nii,self.forest)
        #crop
        if self.is_crop:
            region_list=calc_max_region_list(calc_crop_region(T1_stack_list,50,5),self.forest)
            self.region_list=region_list
            T1_stack_list=crop(T1_stack_list,region_list)
            IR_stack_list=crop(IR_stack_list,region_list)
            T2_stack_list=crop(T2_stack_list,region_list)
        #get mean
        T1mean,IRmean,T2mean=0.0,0.0,0.0
        for samples in T1_stack_list:
            for stacks in samples:
                T1mean=T1mean+np.mean(stacks)
        self.T1mean=T1mean/(len(T1_stack_list)*len(T1_stack_list[0]))
        for samples in IR_stack_list:
            for stacks in samples:
                IRmean=IRmean+np.mean(stacks)
        self.IRmean=IRmean/(len(IR_stack_list)*len(IR_stack_list[0]))
        for samples in T2_stack_list:
            for stacks in samples:
                T2mean=T2mean+np.mean(stacks)
        self.T2mean=T2mean/(len(T2_stack_list)*len(T2_stack_list[0]))

        #transform
        if self.is_transform:
            for stack_index in range(len(T1_stack_list)):
                T1_stack_list[stack_index],  \
                IR_stack_list[stack_index],  \
                T2_stack_list[stack_index]=  \
                self.transform(               \
                T1_stack_list[stack_index],  \
                IR_stack_list[stack_index],  \
                T2_stack_list[stack_index]) 

        # data ready
        self.T1_stack_list=T1_stack_list
        self.IR_stack_list=IR_stack_list
        self.T2_stack_list=T2_stack_list

    def __len__(self):
        return 48
    def __getitem__(self,index):
        return self.region_list[index],self.T1_stack_list[index],self.IR_stack_list[index],self.T2_stack_list[index]
    
    def transform(self,imgT1,imgIR,imgT2):
        imgT1=torch.from_numpy((imgT1.transpose(2,0,1).astype(np.float)-self.T1mean)/255.0).float()
        imgIR=torch.from_numpy((imgIR.transpose(2,0,1).astype(np.float)-self.IRmean)/255.0).float()
        imgT2=torch.from_numpy((imgT2.transpose(2,0,1).astype(np.float)-self.T2mean)/255.0).float()
        return imgT1,imgIR,imgT2

labels = {1: 'Cortical gray matter',
          2: 'Basal ganglia',
          3: 'White matter',
          4: 'White matter lesions',
          5: 'Cerebrospinal fluid in the extracerebral space',
          6: 'Ventricles',
          7: 'Cerebellum',
          8: 'Brain stem',
          # The two labels below are ignored:
          #9: 'Infarction',
          #10: 'Other',
        }

def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and pathology masked."""
    testImage   = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)
    
    # Check for equality
    assert testImage.GetSize() == resultImage.GetSize()
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)    
    
    # Remove pathology from the test and result images, since we don't evaluate on that
    pathologyImage = sitk.BinaryThreshold(testImage, 9, 11, 0, 1)  # pathology == 9 or 10
    
    maskedTestImage   = sitk.Mask(testImage,   pathologyImage)     # tissue    == 1 --  8
    maskedResultImage = sitk.Mask(resultImage, pathologyImage)
    
    # Force integer
    if not 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        maskedResultImage = sitk.Cast(maskedResultImage, sitk.sitkUInt8)
            
    return maskedTestImage, maskedResultImage

def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""        
    dsc = dict()
    for k in labels.keys():
        testArray   = sitk.GetArrayFromImage(sitk.BinaryThreshold(  testImage, k, k, 1, 0)).flatten()
        resultArray = sitk.GetArrayFromImage(sitk.BinaryThreshold(resultImage, k, k, 1, 0)).flatten()
        
        # similarity = 1.0 - dissimilarity
        # scipy.spatial.distance.dice raises a ZeroDivisionError if both arrays contain only zeros.
        try:
            dsc[k] = 1.0 - scipy.spatial.distance.dice(testArray, resultArray)
        except ZeroDivisionError:
            dsc[k] = None
    
    return dsc

def getHausdorff(testImage, resultImage):
    """Compute the 95% Hausdorff distance."""    
    hd = dict()
    for k in labels.keys():
        lTestImage   = sitk.BinaryThreshold(  testImage, k, k, 1, 0)
        lResultImage = sitk.BinaryThreshold(resultImage, k, k, 1, 0)
        
        # Hausdorff distance is only defined when something is detected
        statistics = sitk.StatisticsImageFilter()
        statistics.Execute(lTestImage)
        lTestSum = statistics.GetSum()
        statistics.Execute(lResultImage)
        lResultSum = statistics.GetSum()
        if lTestSum == 0 or lResultSum == 0:
            hd[k] = None
            continue
                                
        # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
        eTestImage   = sitk.BinaryErode(lTestImage, (1,1,0))
        eResultImage = sitk.BinaryErode(lResultImage, (1,1,0))
        
        hTestImage   = sitk.Subtract(lTestImage, eTestImage)
        hResultImage = sitk.Subtract(lResultImage, eResultImage)    
        
        hTestArray   = sitk.GetArrayFromImage(hTestImage)
        hResultArray = sitk.GetArrayFromImage(hResultImage)   
            
        # Convert voxel location to world coordinates. Use the coordinate system of the test image
        # np.nonzero   = elements of the boundary in numpy order (zyx)
        # np.flipud    = elements in xyz order
        # np.transpose = create tuples (x,y,z)
        # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
        # (Simple)ITK does not accept all Numpy arrays; therefore we need to convert the coordinate tuples into a Python list before passing them to TransformIndexToPhysicalPoint().
        testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
        resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]
                
        # Use a kd-tree for fast spatial search
        def getDistancesFromAtoB(a, b):    
            kdTree = scipy.spatial.KDTree(a, leafsize=100)
            return kdTree.query(b, k=1, eps=0, p=2)[0]
        
        # Compute distances from test to result and vice versa. 
        dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
        dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)
        hd[k] = max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
        
    return hd

def getVS(testImage, resultImage):   
    """Volume similarity.
    
    VS = 1 - abs(A - B) / (A + B)
    
    A = ground truth in ML
    B = participant segmentation in ML
    """    
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    vs = dict()
    for k in labels.keys():
        testStatistics.Execute(sitk.BinaryThreshold(testImage, k, k, 1, 0))
        resultStatistics.Execute(sitk.BinaryThreshold(resultImage, k, k, 1, 0))
        
        numerator = abs(testStatistics.GetSum() - resultStatistics.GetSum())
        denominator = testStatistics.GetSum() + resultStatistics.GetSum()               
        
        if denominator > 0:        
            vs[k] = 1 - float(numerator) / denominator
        else:
            vs[k] = None
        
    return vs

def prediction(folder_path):
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    path = folder_path
    pre_path = folder_path + '/pre/'
    for i in os.listdir(pre_path):
        if os.path.isfile(os.path.join(pre_path,i)) and 'reg_T1' in i:
            T1_path = os.path.join(pre_path,i)
        if os.path.isfile(os.path.join(pre_path,i)) and 'FLAIR' in i:
            T2_path = os.path.join(pre_path,i)
        if os.path.isfile(os.path.join(pre_path,i)) and 'reg_IR' in i:
            IR_path = os.path.join(pre_path,i)
    print(T1_path)
    print(T2_path)
    print(IR_path)

    # io vols
    srcvol=nib.load(T1_path)
    outvol=np.zeros((240,240,48),np.uint8)
    # data loader
    loader=MR18loader_test(T1_path=T1_path,IR_path=IR_path,T2_path=T2_path,is_transform=True,is_crop=True,is_hist=True,forest=3)
    testloader=data.DataLoader(loader,batch_size=1,num_workers=1,shuffle=False)

    # model setup
    model_path_1 = '/home/blackbird98/Downloads/SIH/prediction_1/sih/sih/input/model1.pkl'
    n_classes = loader.n_classes
    model_1=fcn_mul(n_classes=n_classes)
    model_1.cuda()
    #torch.load(model_1, map_location=lambda storage, loc: storage)
    state_1 = torch.load(model_path_1)['model_state']
    model_1.load_state_dict(state_1)
    model_1.eval()

    for i_t,(regions_t,T1s_t,IRs_t,T2s_t) in enumerate(testloader):
        T1s_t,IRs_t,T2s_t=Variable(T1s_t.cuda()),Variable(IRs_t.cuda()),Variable(T2s_t.cuda())
        with torch.no_grad():
            out_1=model_1(T1s_t,IRs_t,T2s_t)[0,:,:,:]
        pred_1 = out_1.data.max(0)[1].cpu().numpy()
        h,w=pred_1.shape[0],pred_1.shape[1]
        pred=np.zeros((h,w),np.uint8)
        # vote in 7 results
        for y in range(h):
            for x in range(w):
                #pred_list=np.array([pred_1[y,x],pred_2[y,x],pred_3[y,x],pred_4[y,x],pred_5[y,x],pred_6[y,x],pred_7[y,x]])
                pred_list=np.array([pred_1[y,x]])
                pred[y,x]=np.argmax(np.bincount(pred_list))
        # padding to 240x240
        pred_pad=np.zeros((240,240),np.uint8)
        pred_pad[regions_t[0]:regions_t[1],regions_t[2]:regions_t[3]]=pred[0:regions_t[1]-regions_t[0],0:regions_t[3]-regions_t[2]]
        outvol[:,:,i_t]=pred_pad.transpose()
    # write nii.gz
    nib.Nifti1Image(outvol, srcvol.affine, srcvol.header).to_filename('/home/blackbird98/Downloads/SIH/prediction_1/sih/sih/media/input/pred.nii')
    pred_path = '/home/blackbird98/Downloads/SIH/prediction_1/sih/sih/media/input/pred.nii'
    truth_path = folder_path + '/segm.nii.gz'

    return pred_path, truth_path

def scores(pred_path, truth_path):
    print("Output predicted...")
    testImage, resultImage = getImages(truth_path, pred_path)
    
    dsc = getDSC(testImage, resultImage)
    h95 = getHausdorff(testImage, resultImage)
    vs  = getVS(testImage, resultImage)

    # print('Dice',                dsc,       '(higher is better, max=1)')
    # print('HD',                  h95, 'mm',  '(lower is better, min=0)')
    # print('VS',                   vs,       '(higher is better, max=1)')

    scores = []
    for i in range(1,9):
        scores.append([labels[i], dsc[i], h95[i], vs[i]])

    return scores

# print('Back: Background,')
# print('GM: Cortical GM(red), Basal ganglia(green),')
# print('WM: WM(yellow), WM lesions(blue),')
# print('CSF: CSF(pink), Ventricles(light blue),')
# print('Back: Cerebellum(white), Brainstem(dark red)')
