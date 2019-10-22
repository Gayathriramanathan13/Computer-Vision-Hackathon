# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:37:38 2019

@author: Gayathri
"""

import numpy as np
#from skimage.draw import line
#from skimage.morphology import thin
import matplotlib.pyplot as plt
#from PIL import Image
import xml.etree.ElementTree as ET
#from io import StringIO
#import shutil
import glob
#from datetime import datetime
import os
import re

def get_label(inkml_file_abs_path):
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    
    for child in root:
        if (child.tag == doc_namespace + 'annotation') and (child.attrib == {'type': 'truth'}):
            return child.text

def get_traces_data(inkml_file_abs_path):
    
    traces_data = []
    
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    
    'Stores traces_all with their corresponding id'
    traces_all = [{'id': trace_tag.get('id'),
                    'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
                                    for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                                else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
                                    for axis_coord in coord.split(' ')] \
                            for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                            for trace_tag in root.findall(doc_namespace + 'trace')]
    
    'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
    
    'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')
    
    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):
            
            label = traceGroup.find(doc_namespace + 'annotation').text
            
            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):
                
                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))
                
                'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)
            
            
            traces_data.append({'label': label, 'trace_group': traces_curr})
    
    else:
        'Consider Validation data that has no labels'
        [traces_data.append({'trace_group': [trace['coords']]}) for trace in traces_all]
    
    return traces_data

def createDirectory(dirPath):
    if not os.path.exists(dirPath): 
        os.mkdir(dirPath)


def inkml2img(input_path, output_path,filename):
    if 'train' in output_path.split('.')[0]:
        fout = open(output_path.split('.')[0] + '.txt', 'w+')
        text =get_label(input_path)
        print(text)
        fout.write(text)
        fout.close()
    traces = get_traces_data(input_path)
    for idx, elem in enumerate(traces):
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().spines['top'].set_visible(False)
        plt.axes().spines['right'].set_visible(False)
        plt.axes().spines['bottom'].set_visible(False)
        plt.axes().spines['left'].set_visible(False)
        ls = elem['trace_group']
        folder = output_path
        if 'train' in output_path.split('.')[0]:
            label = elem['label']
            folder = output_path+'/'+label
        for subls in ls:
            data = np.array(subls)
            x,y=zip(*data)
            plt.plot(x,y,linewidth=2,c='black')
            #print(output_path)
            #print(output_path+'_'+idx+'.png')
        createDirectory(folder)
        digit_file_name = folder+'/'+filename+'_'+str(idx)+'.png'
        print(digit_file_name)
        plt.savefig(digit_file_name, bbox_inches='tight', dpi=100)
        plt.show()
        plt.gcf().clear()
    #print(output_path)
    #plt.savefig(output_path, bbox_inches='tight', dpi=100)
    
def inkml2img_full_eq(input_path, output_path):
    if 'train' in output_path.split('.')[0]:
        fout = open(output_path.split('.')[0] + '.txt', 'w+')
        text =get_label(input_path)
        print(text)
        fout.write(text)
        fout.close()
    traces = get_traces_data(input_path)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.axes().spines['top'].set_visible(False)
    plt.axes().spines['right'].set_visible(False)
    plt.axes().spines['bottom'].set_visible(False)
    plt.axes().spines['left'].set_visible(False)
    for idx,elem in enumerate(traces):
        ls = elem['trace_group']
        for subls in ls:
            data = np.array(subls)
            x,y=zip(*data)
            plt.plot(x,y,linewidth=2,c='black')
    plt.savefig(output_path+'.png', bbox_inches='tight', dpi=100)
    plt.show()
    #print(output_path)
    #plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.gcf().clear()


dataPath = 'gayathri-ramanathan-predict-mathnotations-master/hack_data/train'
filesPath = glob.glob(dataPath + '*/*.inkml')
print("There are " + str(len(filesPath)) + " files in " + dataPath)
targetFolder = 'gayathri-ramanathan-predict-mathnotations-master/hack_data/train_imgs/'
targetFolder1 = 'gayathri-ramanathan-predict-mathnotations-master/hack_data/train_imgs2/'
cnt = 0
createDirectory(targetFolder)
numberOfFile = len(filesPath)
for idx,fileInkml in enumerate(filesPath):
        print(fileInkml)
        cnt = cnt + 1
        fileName = fileInkml.split('\\')[2]
        fileName = re.sub('.inkml','',fileName)
        print(fileName)
        print("Processing %s [%d/%d]" % (fileName, cnt, numberOfFile))
        print("[" + str(cnt) + "/" + str(numberOfFile) + "]" + "Processed " + fileInkml + " --> " + targetFolder + fileName + ".png")
        inkml2img_full_eq(fileInkml, targetFolder + fileName)
        inkml2img(fileInkml, targetFolder1,fileName )

dataPath2 = 'gayathri-ramanathan-predict-mathnotations-master/hack_data/test'
filesPath2 = glob.glob(dataPath2 + '*/*.inkml')
print("There are " + str(len(filesPath)) + " files in " + dataPath)       
targetFolder2 = 'gayathri-ramanathan-predict-mathnotations-master/hack_data/test_imgs/'
targetFolder3 = 'gayathri-ramanathan-predict-mathnotations-master/hack_data/test_imgs2/'
count = 0
createDirectory(targetFolder2)
numberOfFile2 = len(filesPath2)
for fileInkml in filesPath2:
        print(fileInkml)
        count = count + 1
        fileName = fileInkml.split('\\')[2]
        fileName = re.sub('.inkml','',fileName)
        print(fileName)
        print("Processing %s [%d/%d]" % (fileName, count, numberOfFile2))
        print("[" + str(count) + "/" + str(numberOfFile2) + "]" + "Processed " + fileInkml + " --> " + targetFolder + fileName + ".png")
        #try:
        inkml2img_full_eq(fileInkml, targetFolder2 + fileName)
        inkml2img(fileInkml, targetFolder3,fileName )

