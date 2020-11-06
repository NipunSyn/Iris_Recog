import numpy as np
import cv2
import os
import math
from scipy import interpolate
from matplotlib import pyplot as plt

#______________________________________________________________________________
im1 = cv2.imread("001_1_1.jpg")
im2 = cv2.imread("001_1_2.jpg")
#______________________________________________________________________________

"""
Loading Images from folder
"""

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    #print ("Number of photos in the folder: ")
    #print (len(images))
    return images

#______________________________________________________________________________

"""
Processing
"""

def processing(image):
    c_image = image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    result = image.copy()
    image = cv2.medianBlur(image,19)
    circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1.4,10,param1=50,param2=120,minRadius=0,maxRadius=0)
    height=20
    width=240
    r=0
    mask = np.zeros((height,width),np.uint8)
    if circles is not None:
        for i in circles[0,:]:
            cv2.circle(c_image,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(c_image,(i[0],i[1]),int(i[2]+(2400/i[2])),(0,255,0),2)
            cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=0)
            r=i[2]
        
            pupil_X=i[0]
            pupil_Y=i[1]
            pupil_R=i[2]
        
            iris_X=i[0]
            iris_Y=i[1]
            iris_R=i[2]+(2400/i[2])
    
    plt.title("Iris Detection")
    plt.imshow(c_image,cmap='gray')
    plt.show()    
    
    angledivisions=239
    radiuspixels=22
    
    r=range(0,(radiuspixels-1),1)
    theta=np.linspace(0,360,num=240)
    theta=list(theta)

    ox=float(pupil_X-iris_X)
    oy=float(pupil_Y-iris_Y)
    
    if ox<=0:
        sgn=-1
    elif ox>0:
        sgn=1
    
    if ox==0 and oy>0:
        sgn=1
    ap=np.ones([1,240])
    ap=list(ap[0])
    a=[i* ((ox**2)+(oy**2)) for i in ap]
    if ox==0:
        phi=90
    else:
        phi=math.degrees(math.atan(float(oy/ox)))
    b=[(math.cos(math.pi-math.radians(phi)-math.radians(i))) for i in theta]


    term1=[(math.sqrt(i)*j) for i,j in zip(a,b)]
    term2=[i*(j**2) for i,j in zip(a,b)]
    term3=[i-(iris_R**2) for i in a]

    rk=[i + math.sqrt(j-k) for i,j,k in zip(term1,term2,term3)]

    r=[i-pupil_R for i in rk]
    r=np.asmatrix(r)

    term1=np.ones([1,radiuspixels])
    term1=np.asmatrix(term1)
    term1=term1.transpose()

    rmat2=np.matmul(term1,r)



    term1=np.ones(((angledivisions+1),1))
    term1=np.asmatrix(term1)
    term2= np.linspace(0,1,(radiuspixels))
    term2=np.asmatrix(term2)
    term3=np.matmul(term1,term2)
    term3=np.asmatrix(term3)
    term3=term3.transpose()

    rmat3=np.multiply(rmat2,term3)
    rmat4=rmat3+pupil_R

    rmat=rmat4[1:radiuspixels-1]




    term1=np.ones(((radiuspixels-2),1))
    term2=[math.cos(math.radians(i)) for i in theta]
    term2=np.asmatrix(term2)

    term3=[math.sin(math.radians(i)) for i in theta]
    term3=np.asmatrix(term3)

    xcosmat=np.matmul(term1,term2)
    xsinmat=np.matmul(term1,term3)

    xot=np.multiply(rmat,xcosmat)
    yot=np.multiply(rmat,xsinmat)

    xo=pupil_X+xot
    yo=pupil_Y-yot



    xt=np.linspace(0,c_image.shape[0]-1,c_image.shape[0])
    yt=np.linspace(0,c_image.shape[1]-1,c_image.shape[1])
    x,y=np.meshgrid(xt,yt)
    
    ip=interpolate.RectBivariateSpline(xt,yt,result)
    polar_array=ip.ev(yo,xo)
    #polar_array = np.asarray(polar_array,dtype=np.uint8)
    plt.title("Normalised")
    plt.imshow(polar_array,cmap='gray')
    plt.show()
    return polar_array

#______________________________________________________________________________
#p1 = processing(im1)
#p2 = processing(im2)
#______________________________________________________________________________    
"""
Template Generation
"""
def temp_gen(polar_array ):
    kernel = cv2.getGaborKernel((240, 20), 0.05, 20, 18, 1, 0, cv2.CV_64F)
    h, w = kernel.shape[:2]
    g_kernel = cv2.resize(kernel, (240, 20), interpolation=cv2.INTER_CUBIC)
    g_kernel_freq=np.fft.fft2(g_kernel)

    freq_image=np.fft.fft2(polar_array)

    mul_image=np.multiply(g_kernel_freq,freq_image)

    inv_image=np.fft.ifft2(mul_image)


    inv_image_ravel=inv_image.ravel()

    pre_template=[];
    for i in inv_image_ravel:
        real_part=np.real(i)
        imaginary_part=np.imag(i)
    
        if((real_part>=0) and (imaginary_part>=0)):
            pre_template.append('11')
        elif ((real_part>=0) and (imaginary_part<0)):
            pre_template.append('10')
        elif ((real_part<0) and (imaginary_part<0)):
            pre_template.append('00')
        elif ((real_part<0) and (imaginary_part>=0)):
            pre_template.append('01')
    
    Template=''.join(pre_template)
    Template=list(Template)
    Template=np.asarray(Template)
    Template=np.reshape(Template,[20,480])
    Template=Template.astype(int)
    
    return Template
#______________________________________________________________________________
#en_Template = temp_gen(p1)
#qu_Template = temp_gen(p2)
#______________________________________________________________________________
"""
Mask Generation
"""

def mask_gen(polar_array):
    polar_array = np.asarray(polar_array,dtype=np.uint8)
    #clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(2,2))
    #cl1 = clahe.apply(polar_array)
    ad_th = cv2.adaptiveThreshold(polar_array,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,17)
    
    _,ot = cv2.threshold(polar_array,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    #ad_th = cv2.medianBlur(ad_th,1)
    #plt.imshow(cl1,cmap="gray")
    a = cv2.bitwise_and(ad_th,polar_array, mask = None)
    plt.title("Iris pixels")
    plt.imshow(a,cmap="gray")
    plt.show()
    a = a>0
    a_ravel = a.ravel()
    pre_a=[]
    for i in a_ravel:
        if i==True:
            pre_a.append('11')
        else:
            pre_a.append('00')
    new_a = ''.join(pre_a)
    new_a = list(new_a)
    new_a = np.asarray(new_a)        
    new_a = np.reshape(new_a,[20,480]) #Same as the normalised image
    new_a=new_a.astype(int)
    return new_a
#______________________________________________________________________________    
#en_Mask = mask_gen(p1)
#qu_Mask = mask_gen(p2)
#______________________________________________________________________________    
"""
Score Calculation
"""
def CalculateScore3(en_Template,en_Mask,qu_Template,qu_Mask): 
    Num1=np.bitwise_xor(en_Template,qu_Template)
    Num2=np.bitwise_and(Num1,en_Mask)
    Num3=np.bitwise_and(Num2,qu_Mask)
    Numerator=np.count_nonzero(Num3)
    
    Den1=np.bitwise_and(en_Mask,qu_Mask)
    Denomenator=np.count_nonzero(Den1)
    
    mask_scor=float(Numerator)/float(Denomenator)

    return mask_scor
#______________________________________________________________________________
#result = CalculateScore3(en_Template,en_Mask,qu_Template,qu_Mask)
#print (result)
#______________________________________________________________________________
"""
Generating tuples of images
"""    


def make_tuple():
    images_a = load_images(folder = 'C:/Users/Tewari\'s/Documents/Database/a')
    images_b = load_images(folder = 'C:/Users/Tewari\'s/Documents/Database/b')
    zipp = zip(images_a,images_b)
    zipp = list(zipp)
    return zipp
#______________________________________________________________________________

def main(zipp):
    total = 0
    im_count = 0
    for tup in zipp: 
        im1,im2 = tup
        c = int(input("Press 1 to process: "))
        if c == 1:
            p1 = processing(im1)
            p2 = processing(im2)
        else: break
        d = int(input("Press 1 to check score: "))
        if d==1:
            en_Template = temp_gen(p1)
            qu_Template = temp_gen(p2)
            en_Mask = mask_gen(p1)
            qu_Mask = mask_gen(p2)
            
            score_de = CalculateScore3(en_Template,en_Mask,qu_Template,qu_Mask)
            print ("The score is: ", round(score_de,3))
            if score_de>0 and score_de<=0.15:
                print ("Accurate Match")
            elif score_de>0.25:
                print ("Inaccurate Match")
        else: break
        im_count +=1
        total+=score_de
        avg = total/im_count
        print("The average score is: ",round(avg,3))
    print("**End of Iteration**")

main(make_tuple())
            
























#______________________________________________________________________________
