#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 19:22:48 2020

@author: amir niaraki
"""
import numpy as np
import argparse
import cv2
import pandas as pd
import timeit
import csv
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics




def adaboos(data):
    # data = pd.read_csv("res.csv")
    y = data[data.columns[-1]]
    print(y)
    X = data.drop(data.columns[-1], axis=1)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print(y_train)
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))




def dataGenerator(image, image_LAB, circle_centers, diam):
    circle_centers_df = pd.read_csv(circle_centers, header=None)
    rows, cols, _ = image.shape
    print(rows, " ", cols)
    df_result = pd.DataFrame(0, np.arange(rows * cols), np.arange(7))
    rows_circles, cols_circles = circle_centers_df.shape
    for i in range(rows_circles):
        for j in range(cols_circles):
            elements = tuple(circle_centers_df.iloc[i, j][1:-1].split(", "))
            c_i = int(elements[0])
            c_j = int(elements[1])

            for temp_i in range(c_i - diam // 2, c_i + diam //2):
                for temp_j in range(c_j - diam // 2, c_j + diam //2):
                    if temp_i >= 0 and temp_i < rows and temp_j >= 0 and temp_j < cols:
                        if (c_i - temp_i) ** 2 + (c_j - temp_j) ** 2 <= (diam//2) ** 2:
                            df_result.iloc[temp_i * cols + temp_j, 6] = 1

    for i in range (rows):
        for j in range(cols):
            [B, G, R] = image[i, j]
            df_result.iloc[i * cols + j, 0] = B
            df_result.iloc[i * cols + j, 1] = G
            df_result.iloc[i * cols + j, 2] = R
            [L, a, b] = image_LAB[i,j]
            df_result.iloc[i * cols + j, 3] = L
            df_result.iloc[i * cols + j, 4] = a
            df_result.iloc[i * cols + j, 5] = b
    df_result.to_csv("res.csv", index=False)
    
    return df_result






def template_drawing(original_image, df, diam):
    theta_corr=0.65
    overlap_neighors=5
    image=original_image

    rows, columns= df.shape
    thickness=1
    # center_coordinates=(100,100)
    radius=int(diam/2)
    color = (255, 0, 0)
    step=int(diam//4)
    tree_log=[]
    center_list_row=[]
    print(" drawing the templates")
    for i in range(0,rows-overlap_neighors,overlap_neighors+1):
        for j in range(0, columns-overlap_neighors, overlap_neighors+1):
            best_score=df.iloc[i,j]
            c_i = step * (i) + diam / 2
            c_j = step * (j) + diam / 2
            center_coordinates = (int(c_j), int(c_i))
            for i_sqr in range(overlap_neighors):
                for j_sqr in range(overlap_neighors):
                    #check the condition if the score is above the theta_corr and if it is maximum between the neighbors
                    if df.iloc[i+i_sqr,j+j_sqr]>theta_corr:

                        if df.iloc[i+i_sqr,j+j_sqr]>best_score:
                            df.iloc[i,j]=0
                            best_score=df.iloc[i+i_sqr,j+j_sqr]
                            c_i = step * (i + i_sqr) + diam / 2
                            c_j = step * (j + j_sqr) + diam / 2
                            center_coordinates = (int(c_j), int(c_i))
                        else:
                            df.iloc[i + i_sqr, j + j_sqr]=0
                    else:
                        df.iloc[i + i_sqr, j + j_sqr] = 0

            if best_score>theta_corr:
                image = cv2.circle(image, center_coordinates, radius, color, thickness)
                
                center_list_row.append(center_coordinates)
            
        #some operations here for the center_list_row to find the line
        # image=cv2.line(image, center_list_row[0],center_list_row[-1],color, thickness)
        tree_log.append(center_list_row)
        center_list_row=[]
            
                
    
    return df , image, tree_log
    # print(df)
                
    

    

    






def localization(blk_img, diam, scale=20):
   # from offline calculations we found out that an image with scale 1:20 has 224 meter width
   # Each pixel is 0.23 m
   # Our assumption is that each tree contains 16 to 24 pixels in the transfered image
    height, width, _ = blk_img.shape
    dataframe_list = []
    print("Template matching for diam="+str(diam))
    step = int(diam / 4)
    num_circles_row = (height-diam)// step+1
    num_circles_col = (width-diam)// step+1
    df = pd.DataFrame(0, np.arange(num_circles_row), np.arange(num_circles_col))
    df_binary=df
    print("number of templates along the rows and columns:")
    print(num_circles_row,num_circles_col)
    for i in range(num_circles_row):
        for j in range(num_circles_col):
            c_i=step*i+diam/2
            c_j=step*j+diam/2
            total_pixels=0
            white_pixels=0
    
    
            for x in range (int(c_i-diam/2),int(c_i+1+diam/2)-1):
                for y in range (int(c_j-diam/2),int(c_j+1+diam/2)-1):
                    # print("x",x,"y",y)
                    if ((x - c_i) ** 2 + (y - c_j) ** 2 <= (diam / 2) ** 2):
                        if (blk_img[x,y]==[255,255,255]).all():
                            white_pixels+=1
                        
                        total_pixels+=1
            # print(total_pixels)
                        
            template_score=white_pixels/total_pixels
            # print("i"+str(i)+"j"+str(j))
            # print(template_score)
    
            df.iloc[i,j]=template_score
              
   
    # print(df)
    return df



def masking(img_name):
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", help = img_name)
# 	args = vars(ap.parse_args())
	# load the image
	image = cv2.imread(img_name)
	image2=image.copy()

	height, width,_=image.shape
	print(height, width)
	upper=[86,255,255]
	lower=[36,0,0]
	# print(image[50,50])
	# cv2.imshow('pseudocolor',image)
# 	cv2.waitKey(0)

	for i in range(0, height):
		for j in range(0,width):
			if (image2[i,j]<=upper).all() and (image2[i,j]>=lower).all():
				image2[i, j]= [255, 255, 255]
			else:
				image2[i, j]=[0,0,0]
	return image, image2





def main():
    img_name = "sample_image2.png"
    img_name = "sample_image_croped.png"
    start=timeit.default_timer()
    diam_list=[16]
    original_image, masked_image=masking(img_name)
    for diam in diam_list:
        scores=localization(masked_image, diam)
        filtered_scores, image_circled, tree_log =template_drawing(original_image, scores, diam)


    # cv2.imshow("1", original_image)
    #cv2.waitKey(0)

    # cv2.imshow("2", masked_image)
    # cv2.waitkey(0)

    
    for row in tree_log:
        with open('tree_log.csv','a') as tree_log:
            wr=csv.writer(tree_log, dialect='excel')
            wr.writerow(row)
# Creating our dataset

    image = cv2.imread(img_name)
    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # prepared_data=dataGenerator(image, image_LAB, "tree_log.csv", 16)
    cv2.imshow("tree detection", image_circled)
    cv2.waitKey(0)
    # adaboos(prepared_data)
    end=timeit.default_timer()
    print("time:  "+str(end-start))
if __name__=="__main__":
	main()
