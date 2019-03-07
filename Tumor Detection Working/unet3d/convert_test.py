import SimpleITK as sitk

x = input("Enter file name:")

reader = sitk.ImageFileReader()
reader.SetImageIO("NiftiImageIO")
reader.SetFileName(x)
image = reader.Execute()

# basic_transform = sitk.HDF5TransformIO()

# from medpy.io import load
# from medpy.io import save

# i, h = load("Brats18_2013_3_1_flair.nii.gz")
# print (i.shape, i.dtype)
# save(i,"test.h5",hdr=h,force=True)