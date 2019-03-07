import sys
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import warnings
warnings.filterwarnings("ignore")
from nilearn import plotting
from nilearn import image
from PIL import Image as img
from PIL import ImageOps
import os
import csv
from prediction import scores, prediction

class MyPopup2(QWidget):
   def __init__(self):
      QWidget.__init__(self)
      
      layout=QVBoxLayout()
      self.setLayout(layout)
      self.setGeometry(50,50,500,300)
      self.setWindowTitle("Score Predictions")

      self.addHead=QLabel()
      layout.addWidget(self.addHead)
      self.addHead.setText('Segment of Brain:     Dice Similarity Coefficient    95% Hausdorff distance      Volume similarity')
      self.addHead.setStyleSheet("QLabel { color : skyblue;}")
      
      self.scores=QLabel()
      layout.addWidget(self.scores)

      score=scores(pred_path,truth_path)
      print (score)
      #score=[['Cortical gray matter', 0.8264493931976515, 1.9166649580001898, 0.9796933750991367], ['Basal ganglia', 0.7360081489445984, 19.805929672962634, 0.9134740535340389], ['White matter', 0.8207025704533785, 2.87499743700027, 0.99131603588365], ['White matter lesions', 0.7386807181889149, 5.749998450279245, 0.7603922716627635], ['Cerebrospinal fluid in the extracerebral space', 0.8072240305876914, 2.7105744221204677, 0.9905379750432934], ['Ventricles', 0.9158469133969338, 3.6867338695570457, 0.9462860536768254], ['Cerebellum', 0.9212352860878057, 23.338083765926005, 0.9874308275492739], ['Brain stem', 0.8131455399061033, 14.648693973517284, 0.8730046948356808]]


      #self.le.setText(score[0][0]+score[0][1]+score[0][2]+score[0][3])
      for i in range (0,8):
         for j in range(0,4):
            self.scores.setText(self.scores.text() +str(score[i][j])+'   ')
         self.scores.setText(self.scores.text() + "\n" )
      self.setLayout(layout)

class MyPopup(QWidget):
   def __init__(self):
      QWidget.__init__(self)
      
      layout=QVBoxLayout()
      self.setLayout(layout)
      self.setGeometry(50,50,500,300)
      self.setWindowTitle("Color Chart")

      self.label=QLabel()
      self.label.setFixedHeight(20)
      self.label.setFixedWidth(20)
      self.label.setStyleSheet("QLabel { background-color : darkred;}")
      self.label1=QLabel()
      self.label1.setText('Brain Stem')
      line=QtGui.QHBoxLayout()
      line.addWidget(self.label)
      line.addWidget(self.label1)
      layout.addLayout(line)

      self.label2=QLabel()
      self.label2.setFixedHeight(20)
      self.label2.setFixedWidth(20)
      self.label2.setStyleSheet("QLabel { background-color : white;}")
      self.label3=QLabel()
      self.label3.setText('Cerebellum')
      line1=QtGui.QHBoxLayout()
      line1.addWidget(self.label2)
      line1.addWidget(self.label3)
      layout.addLayout(line1)

      self.label4=QLabel()
      self.label4.setFixedHeight(20)
      self.label4.setFixedWidth(20)
      self.label4.setStyleSheet("QLabel { background-color :lightblue;}")
      self.label5=QLabel()
      self.label5.setText('Ventricles')
      line3=QtGui.QHBoxLayout()
      line3.addWidget(self.label4)
      line3.addWidget(self.label5)
      layout.addLayout(line3)

      self.label6=QLabel()
      self.label6.setFixedHeight(20)
      self.label6.setFixedWidth(20)
      self.label6.setStyleSheet("QLabel { background-color : pink;}")
      self.label7=QLabel()
      self.label7.setText('CSF')
      line4=QtGui.QHBoxLayout()
      line4.addWidget(self.label6)
      line4.addWidget(self.label7)
      layout.addLayout(line4)

      self.label8=QLabel()
      self.label8.setFixedHeight(20)
      self.label8.setFixedWidth(20)
      self.label8.setStyleSheet("QLabel { background-color : blue;}")
      self.label9=QLabel()
      self.label9.setText('WM lesions')
      line5=QtGui.QHBoxLayout()
      line5.addWidget(self.label8)
      line5.addWidget(self.label9)
      layout.addLayout(line5)

      self.label10=QLabel()
      self.label10.setFixedHeight(20)
      self.label10.setFixedWidth(20)
      self.label10.setStyleSheet("QLabel { background-color : yellow;}")
      self.label11=QLabel()
      self.label11.setText('WM')
      line6=QtGui.QHBoxLayout()
      line6.addWidget(self.label10)
      line6.addWidget(self.label11)
      layout.addLayout(line6)

      self.label12=QLabel()
      self.label12.setFixedHeight(20)
      self.label12.setFixedWidth(20)
      self.label12.setStyleSheet("QLabel { background-color : green;}")
      self.label13=QLabel()
      self.label13.setText('Basal ganglia')
      line7=QtGui.QHBoxLayout()
      line7.addWidget(self.label12)
      line7.addWidget(self.label13)
      layout.addLayout(line7)

      self.label14=QLabel()
      self.label14.setFixedHeight(20)
      self.label14.setFixedWidth(20)
      self.label14.setStyleSheet("QLabel { background-color : red;}")
      self.label15=QLabel()
      self.label15.setText('Cortical GM')
      line8=QtGui.QHBoxLayout()
      line8.addWidget(self.label14)
      line8.addWidget(self.label15)
      layout.addLayout(line8)

      self.setLayout(layout)


class filedialogdemo(QWidget):
   def __init__(self, parent = None):
      super(filedialogdemo, self).__init__(parent)
      
      
      layout = QVBoxLayout()
      self.setLayout(layout)
      self.setGeometry(50, 50, 500, 300)
      self.setWindowFlags(Qt.WindowTitleHint)
      self.setWindowTitle("NeuralNets")
      self.setWindowIcon(QtGui.QIcon('logo1.png'))
      self.setWindowFlags( Qt.WindowTitleHint |  Qt.WindowMinimizeButtonHint |Qt.WindowCloseButtonHint | Qt.WindowSystemMenuHint);

      menu_bar=QMenuBar()
      layout.addWidget(menu_bar)

      #Adding Menus into Menubar
      file_menu=menu_bar.addMenu('File')
      edit_menu=menu_bar.addMenu('Edit')
      help_menu=menu_bar.addMenu('Help')
      theme_menu=menu_bar.addMenu('Themes')
      data_menu=menu_bar.addMenu('Data Analysis')
      #color_table_menu=menu_bar.addMenu('Color Table')

      score_opt=QtGui.QAction("Predict Scores",self)
      score_opt.setShortcut("Ctrl+P")
      score_opt.triggered.connect(self.newtwo)
      data_menu.addAction(score_opt)

      upload_opt=QtGui.QAction("Upload a File",self)
      upload_opt.setShortcut("Ctrl+U")
      upload_opt.triggered.connect(self.getfile)
      file_menu.addAction(upload_opt)

      colortab_opt=QtGui.QAction("Color chart",self)
      colortab_opt.setShortcut("Ctrl+S")
      colortab_opt.triggered.connect(self.newone)
      help_menu.addAction(colortab_opt)




      exit_menu = QtGui.QAction("Quit", self)
      exit_menu.setShortcut("Ctrl+Q")
      exit_menu.setStatusTip('Leave The App')
      exit_menu.triggered.connect(self.close_application)

      file_menu.addAction(exit_menu)
     
      
      enlarge_opt=QtGui.QAction("Enlarge Window",self)
      enlarge_opt.setShortcut("Ctrl+E")
      enlarge_opt.triggered.connect(self.enlarge_window)
      edit_menu.addAction(enlarge_opt)

      reduce_opt=QtGui.QAction("Reduce Window",self)
      reduce_opt.setShortcut("Ctrl+R")
      reduce_opt.triggered.connect(self.reduce_window)
      
      edit_menu.addAction(reduce_opt)

      #Adding Motif Theme in Theme Menu
      theme_opt1=QtGui.QAction("Motif",self)
      theme_opt1.setShortcut("Ctrl+M")
      theme_opt1.triggered.connect(self.theme_motif) 
      theme_menu.addAction(theme_opt1)

      #Adding Cde Theme in Theme Menu
      theme_opt2=QtGui.QAction("Cde",self)
      theme_opt2.setShortcut("Ctrl+C")
      theme_opt2.triggered.connect(self.theme_cde)
      theme_menu.addAction(theme_opt2)

      #Adding Plastique Theme in Theme Menu
      theme_opt3=QtGui.QAction("Plastique",self)
      theme_opt3.setShortcut("Ctrl+P")
      theme_opt3.triggered.connect(self.theme_plat)
      theme_menu.addAction(theme_opt3)

      #Adding Clean Looks Theme in Theme Menu
      theme_opt4=QtGui.QAction("Cleanlooks",self)
      theme_opt4.setShortcut("Ctrl+L")
      theme_opt4.triggered.connect(self.theme_cleanlooks)
      theme_menu.addAction(theme_opt4)

      #Adding windowsvista Theme in Theme Menu
      theme_opt5=QtGui.QAction("windowsvista",self)
      theme_opt5.setShortcut("Ctrl+V")
      theme_opt5.triggered.connect(self.theme_windowsvista)
      theme_menu.addAction(theme_opt5)

      #Adding Mac Theme in Theme Menu
      '''theme_opt6=QtGui.QAction("Mac",self)
      theme_opt6.setShortcut("Ctrl+A")
      theme_opt6.triggered.connect(self.theme_mac)
      theme_menu.addAction(theme_opt6)'''


      #Adding View1 to Color Menu
      '''color1_opt=QtGui.QAction("View 1",self)
      color1_opt.setShortcut("Ctrl+1")
      color1_opt.triggered.connect(self.color_view1)
      color_table_menu.addAction(color1_opt)

      #Adding View2 to Color Menu
      color2_opt=QtGui.QAction("View 2",self)
      color2_opt.setShortcut("Ctrl+1")
      color2_opt.triggered.connect(self.color_view2)
      color_table_menu.addAction(color2_opt)'''
    
      self.btn = QPushButton("Upload a file")
      self.btn.clicked.connect(self.getfile)
      
      layout.addWidget(self.btn)

      
      self.le = QLabel()
      self.le.setStyleSheet("background-color: white;")
      pixmap=QPixmap('logo_sih.jpg')
      self.le.setPixmap(pixmap.scaled(600,600,QtCore.Qt.KeepAspectRatio))
      self.le.setAlignment(Qt.AlignCenter)
      

      self.le1 = QLabel()
      #self.le1.setStyleSheet("background-color: white;")

      #pixmap=QPixmap('logo_sih.png')
      #self.le1.setPixmap(pixmap.scaled(400,400,QtCore.Qt.KeepAspectRatio))
      #self.le1.setAlignment(Qt.AlignRight)


      #self.le2 = QLabel()
      #self.le2.setStyleSheet("background-color: white;")

      #self.le3 = QLabel()
      #self.le3.setStyleSheet("background-color: white;") 
      hbox=QtGui.QHBoxLayout()
      
      hbox.addWidget(self.le)
      hbox.addWidget(self.le1)
      #hbox.addWidget(self.le2)
      #hbox.addWidget(self.le3)
      layout.addLayout(hbox)
      
    
      self.le4=QLabel()
      

      self.le5=QLabel()
      self.le5.setAlignment(Qt.AlignCenter)

      hbox1=QtGui.QHBoxLayout()
      hbox1.addWidget(self.le4)
      hbox1.addWidget(self.le5)

      layout.addLayout(hbox1)

      self.progress=QtGui.QProgressBar(self)
      layout.addWidget(self.progress)


      self.btn = QPushButton("Analyze")
      self.btn.clicked.connect(self.analyze)


      layout.addWidget(self.btn)
      layout.setMargin(0)
      layout.setSpacing(2)
      
      

      #self.btn = QPushButton("Save Picture")
      #self.btn.clicked.connect(self.save_file)
      
      #layout.addWidget(self.btn)
  
   def theme_motif(self):
      print ("Theme Motif")
      QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('motif'))

   def theme_cde(self):
      print ("Theme Cde")
      QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('cde'))

   def theme_plat(self):
      print ("Theme Plastique")
      QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Plastique'))

   def theme_cleanlooks(self):
      print ("Theme Cleanlooks")
      QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))

   def theme_windowsvista(self):
      print ("Theme Windows Vista")
      QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('windowsvista'))

   #def theme_mac(self):
   #   print ("Theme Mac")
   #   QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('sgi'))

   def color_view1(self):
      global color1
      global color2
      color1=[0,0,0]
      color2=[242,245,5]
      self.model_selection()

   def color_view2(self):
      global color1
      global color2
      color1=[ 180, 253, 75 ]
      color2=[ 244, 60, 6 ]
      self.model_selection()

   def newone(self):
      self.w=MyPopup()
      self.w.setGeometry(100,100,400,200)
      self.w.show()

   def newtwo(self):
      self.w=MyPopup2()
      self.w.setGeometry(100,100,400,200)
      self.w.show()
      

   def model_selection(self):
      image_file= img.open(fname.split('.')[0]+'_1.png')
      image_file=ImageOps.grayscale(image_file)
      image_file=ImageOps.colorize(image_file,color1,color2).save((fname.split('.')[0])+'_1_view.png')   
      pixmap=QPixmap(fname.split('.')[0]+'_1_view.png')
      self.le.setPixmap(pixmap.scaled(650,650,QtCore.Qt.KeepAspectRatio))

      os.remove(fname.split('.')[0]+'_1_view.png')

      '''image_file= img.open(fname.split('.')[0]+'_2.png')
      image_file=ImageOps.grayscale(image_file)
      ImageOps.colorize(image_file,color1,color2).save((fname.split('.')[0])+'_2_view.png')   
      pixmap=QPixmap(fname.split('.')[0]+'_2_view.png')
      self.le1.setPixmap(pixmap.scaled(200,200,QtCore.Qt.KeepAspectRatio))   
      os.remove(fname.split('.')[0]+'_2_view.png')

      image_file= img.open(fname.split('.')[0]+'_3.png')
      image_file=ImageOps.grayscale(image_file)
      ImageOps.colorize(image_file,color1,color2).save((fname.split('.')[0])+'_3_view.png')   
      pixmap=QPixmap(fname.split('.')[0]+'_3_view.png')
      self.le2.setPixmap(pixmap.scaled(200,200,QtCore.Qt.KeepAspectRatio))   
      os.remove(fname.split('.')[0]+'_3_view.png')

      image_file= img.open(fname.split('.')[0]+'_4.png')
      image_file=ImageOps.grayscale(image_file)
      ImageOps.colorize(image_file,color1,color2).save((fname.split('.')[0])+'_4_view.png')   
      pixmap=QPixmap(fname.split('.')[0]+'_4_view.png')
      self.le3.setPixmap(pixmap.scaled(200,200,QtCore.Qt.KeepAspectRatio))   
      os.remove(fname.split('.')[0]+'_4_view.png')'''

   def getfile(self):
      global fname 
      fname = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
      print (fname)
      i = 1
      if fname=='':
         pass
      else:
         self.btn.setText('Uploading')
         self.completed =0

         while self.completed<99:
            self.completed+=0.0001
            self.progress.setValue(self.completed)

      self.completed =0
      self.progress.setValue(self.completed)
      self.btn.setText('Analyze')
      print(fname)
      #filename = fname.split('.')[0] + '_'+str(i) + '.png'
      #plotting.plot_anat(fname,output_file=filename)
      #for img in image.iter_img(fname):
      #   filename = fname.split('.')[0] + '_'+str(i) + '.png'
      #   plotting.plot_anat(img, threshold = 3, display_mode = 'z', cut_coords=1, colorbar = False, output_file=filename)
      #   i = i + 1

      
      #pixmap=QPixmap(fname.split('.')[0]+'_1.png')
      #self.le.setPixmap(pixmap.scaled(600,600,QtCore.Qt.KeepAspectRatio))
      #self.le.setAlignment(Qt.AlignCenter) 

      #pixmap=QPixmap(fname.split('.')[0]+'_2.png')
      #self.le1.setPixmap(pixmap.scaled(200,200,QtCore.Qt.KeepAspectRatio))
      #self.le1.setAlignment(Qt.AlignCenter)
      
      #pixmap=QPixmap(fname.split('.')[0]+'_3.png')
      #self.le2.setPixmap(pixmap.scaled(200,200,QtCore.Qt.KeepAspectRatio))
      #self.le2.setAlignment(Qt.AlignCenter) 

      #pixmap=QPixmap(fname.split('.')[0]+'_4.png')
      #self.le3.setPixmap(pixmap.scaled(200,200,QtCore.Qt.KeepAspectRatio))
      #self.le3.setAlignment(Qt.AlignCenter) 

      return fname
   #def style_choice(self,text):
   #   self.setText(text)
   #   QtGui.QApplication.setStyle(QtGui.QStyleFactory.create(text))

   '''def score(self):
      #score=scores(pred_path,truth_path)
      #print (score)

      score=[['Cortical gray matter', 0.8264493931976515, 1.9166649580001898, 0.9796933750991367], ['Basal ganglia', 0.7360081489445984, 19.805929672962634, 0.9134740535340389], ['White matter', 0.8207025704533785, 2.87499743700027, 0.99131603588365], ['White matter lesions', 0.7386807181889149, 5.749998450279245, 0.7603922716627635], ['Cerebrospinal fluid in the extracerebral space', 0.8072240305876914, 2.7105744221204677, 0.9905379750432934], ['Ventricles', 0.9158469133969338, 3.6867338695570457, 0.9462860536768254], ['Cerebellum', 0.9212352860878057, 23.338083765926005, 0.9874308275492739], ['Brain stem', 0.8131455399061033, 14.648693973517284, 0.8730046948356808]]


      #self.le.setText(score[0][0]+score[0][1]+score[0][2]+score[0][3])
      #for i in range (0,8):
      #   for j in range(0,4):
      #      self.le.setText(self.le.text() +str(score[i][j])+' ')
      #   self.le.setText(self.le.text() + "\n" )
         

      self.le.setStyleSheet("QLabel { background-color : none;color:blue;}")
      self.le.setAlignment(Qt.ALignLeft)
      self.le1.setText(' ')
      self.le4.setText(' ')
      self.le5.setText(' ')'''
      
   def analyze(self,getfile):
      color1=[0,0,0]
      color2=[242,245,5]
      if fname == 0:
         print("Error no fname")
      else:
         self.completed =0

         while self.completed<50:
            self.completed+=0.0001
            self.progress.setValue(self.completed)

      global pred_path,truth_path
      pred_path,truth_path=prediction(fname)
      print (pred_path,truth_path)
     
      #image_file = img.open(fname.split('.')[0]+'_1.png') # open colour image
      
      
      filename = 'sih_pred_out.png'
      plotting.plot_epi(pred_path,output_file=filename,colorbar=True)

      filename = 'sih_truth_out.png'
      plotting.plot_epi(truth_path,output_file=filename,colorbar=True)




      #image_file = image_file.convert('RGB',matrix) 
      #image_file=image_file.save(fname.split('.')[0]+'_out_1.png')
      #print (image_file)
      #file=fname.split('/')[0]+'/'+fname.split('/')[1]+'/'+fname.split('/')[2]+'/'+fname.split('/')[3]+'/'+fname.split('/')[4]+'/'+fname.split('/')[5]+'/prediction.nii.gz'

      #filename = fname.split('.')[0]+ '_out_1.png'
      #plotting.plot_anat(file,output_file=filename)
      #image_file= img.open(fname.split('.')[0]+'_out_1.png')
      #image_file=ImageOps.grayscale(image_file)
      #image_file=ImageOps.colorize(image_file,color1,color2).save((fname.split('.')[0])+'_1_view.png')   
      while self.completed<100:
         self.completed+=0.0001
         self.progress.setValue(self.completed)
      
      pixmap=QPixmap('sih_pred_out.png')
      self.le.setPixmap(pixmap.scaled(500,500,QtCore.Qt.KeepAspectRatio))
      pixmap=QPixmap('sih_truth_out.png')
      self.le1.setPixmap(pixmap.scaled(500,500,QtCore.Qt.KeepAspectRatio))


      
   

      '''name=fname.split('/')[5]
      print (name)
      csvFile='/root/Desktop/Pyqt/prediction_epoch31/brats_scores.csv'
      csv_file = csv.reader(open(csvFile, "r"), delimiter=",")


      #loop through csv list
      for row in csv_file:
          #if current rows 2nd value is equal to input, print that row
          for value in row:
             if name == value:

               data=row
               print (row)
               data[1]=float(data[1])
               data[1]=float("{0:.4f}".format(data[1]))
               data[1]=str(data[1])
               print (data[1])

               data[2]=float(data[2])
               data[2]=float("{0:.4f}".format(data[2]))
               data[2]=str(data[2])
               print (data[2])

               data[3]=float(data[3])
               data[3]=float("{0:.4f}".format(data[3]))
               data[3]=str(data[3])
               print (data[3])
               self.le4.setText("Whole Tumor: "+data[1]+'  '+'Tumor Core: '+data[2]+'  '+'Enhancing Tumor: '+ data[3]+'')
               self.le4.setStyleSheet("color:'blue';")

               print (row[3])'''
      
      #for k,v in scores:
      #   print (k,v)
      self.le4.setText("Prediction")
      self.le4.setFixedWidth(250)
      self.le5.setText("Truth")
      self.le5.setFixedWidth(250)


      self.btn = QPushButton("Save")
      self.btn.clicked.connect(self.analyze)
      
      QVBoxLayout().addWidget(self.btn)
 
   def color_table(self):
      print ('working!!')
      image_file = img.open(fname.split('.')[0]+'_1.png') # open colour image
      
      image_file = image_file.convert('RGBA') # convert image to black and white
      
      image_file=image_file.save(fname.split('.')[0]+'_color1_1.png')
      #print (image_file)

      pixmap=QPixmap(fname.split('.')[0]+'_color1_1.png')
      self.le.setPixmap(pixmap.scaled(200,200,QtCore.Qt.KeepAspectRatio))
      

   def close_application(self):
        choice=QtGui.QMessageBox.question(self,'Quit','Are you sure?',QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        
        if choice == QtGui.QMessageBox.Yes:
            print("whooaaaa so custom!!!")
            sys.exit()
        else:
            pass
   def enlarge_window(self):
      self.setGeometry(0,200,2000,300)

   def reduce_window(self):
      self.setGeometry(50, 50, 500, 300)
      
   def save_file(self):
      name=QtGui.QFileDialog.getSaveFileName(self,'Save File',"Image files (*.png)")
            
def main():
   app = QApplication(sys.argv)
   ex = filedialogdemo()
   ex.show()
   sys.exit(app.exec_())
   
if __name__ == '__main__':
   main()