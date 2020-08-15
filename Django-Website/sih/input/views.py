import os
from django.core.files.base import File
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views.generic import FormView, DetailView
from .models import InputImage
from .forms import InputImageForm
from .prediction import prediction, scores
import gzip
import shutil
import zipfile


class InputImageView(FormView):
    template_name = 'image_upload.html'
    form_class = InputImageForm

    def form_valid(self, form):
        input_image = InputImage(
            image=self.get_form_kwargs().get('files')['image'])
        input_image.save()
        
        s = input_image.image.path
        arrlist = s.split('/')

        s = ''

        folder_name = arrlist[-1]
        for i in range(len(arrlist)-1):
            s+= arrlist[i]
            s+= '/'

        print("s is  : ", s)

        print("folder name is : ", folder_name)
        self.id = input_image.id

        if(input_image.extension() == ".zip" ):
            with zipfile.ZipFile(input_image.image.path, 'r') as zip:
                zip.printdir() 
                zip.extractall(s)

        predict_str = s+folder_name[:-4]
        pred_path, truth_path = prediction(predict_str)
        print(pred_path)
        print(truth_path)

        input_image_path = 'input/' + folder_name[:-4]+'/'+ 'pre/FLAIR.nii.gz'

        input_image = InputImage(image = input_image_path)
        input_image.save()
        s =  input_image.image.path[:-3]

        if(input_image.extension() == ".gz"):
            with gzip.open(input_image.image.path,'rb') as f_in:
                with open(s,'wb') as f_out:
                    shutil.copyfileobj(f_in,f_out)

        arrlist = s.split('/')
        path_final = arrlist[-3] + '/' + arrlist[-2] + '/' + arrlist[-1]
        # print("path final is : ",path_final)

        input_image = InputImage( image = path_final)
        input_image.save()
        self.id = input_image.id

        return HttpResponseRedirect(self.get_success_url())

    def get_success_url(self):
        return reverse('input_image', kwargs={'pk': self.id})

class InputDetailView(DetailView):
    model = InputImage
    template_name = 'image_detail.html'
    context_object_name = 'image'

class OutputView(DetailView):
    model = InputImage
    template_name = 'image_output.html'
    context_object_name = 'image'
