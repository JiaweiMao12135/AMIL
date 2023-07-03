import os
import time
import random
import shutil
from pandas import DataFrame
import pandas as pd
import numpy as np
import csv
import re
from tqdm import tqdm
import jieba
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
import argparse
import gensim
import logging
from sklearn.model_selection import train_test_split

#Function for Pfam file cleaning
def clean_pfam_csv(pfam_path,new_pfam_path):
    """
    :param pfam_path: Original Pfam annotation file
    :param new_pfam_path: Clean Pfam annotation file
    :return: None
    """
    taxons = os.listdir(pfam_path)
    for taxon in taxons:
        folder_path = pfam_path+taxon+'/'
        new_folder_path = new_pfam_path+taxon+'/'
        csves = os.listdir(folder_path)
        if os.path.isdir(new_folder_path) is False:
            os.makedirs(new_folder_path)
        for csv_ in tqdm(csves):
            if csv_== '.ipynb_checkpoints':
                continue
            result=[]
            file = open(folder_path+csv_,'r')
            infoes = file.readlines()[28:]
            for info in infoes :
                clean_info = []
                info = info.split(' ')
                for i in info:
                    if i != '':
                        clean_info.append(i)
                clean_info = clean_info[:-1]
                if clean_info not in result:
                    result.append(clean_info)
                    with open(new_folder_path + folder_path[9:] + csv_, 'a+',newline='') as new_file:
                        writer = csv.writer(new_file)
                        writer.writerow(clean_info)
                    new_file.close()
    print('Cleaning of Pfam annotation files is Done!')
#Function for doc1 generating(v=1)
def doc1_generator(folder_path, label, output_folder_path):
    """
    :param folder_path: Path to the cleaned pfam annotation file for each taxon
    :param label: Classification labels for each taxon
    :param output_folder_path:Output a csv file with three columns, which are the species name, taxonomic label and the functional domain name contained in all protein sequences
    :return:
    """
    if os.path.isdir(output_folder_path) is False:
        os.makedirs(output_folder_path)
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name =='.ipynb_checkpoints':
            continue
        doc = ''
        file_path = os.path.join(folder_path, file_name)
        specie_data = pd.read_csv(file_path, header=None)
        specie_data = specie_data.sort_values(specie_data.columns[0])
        specie_data.drop_duplicates(subset = None, keep = 'first',inplace=True)
        specie_data = specie_data[[0,1,6]]
        file = specie_data.groupby(specie_data[0],sort=False)
        for gene,domain in file:
            sentence = ""
            gene_name = str(gene)
            domain = domain.sort_values([1])
            domain_list = domain[6] #一个sentence
            for s in domain_list:
                sentence+=s
                sentence+=' '
            sentence+='\n'
            doc+=sentence
        with open (output_folder_path+'doc.csv','a+') as doc_file:
            writer = csv.writer(doc_file)
            writer.writerow([file_name, label, doc])
        doc_file.close()
#Function for cleaning original .fa to remove sequences without Pfam annotation.
def clean_fa(new_pfam_path, folder_path, output_folder_path):

    """
    :param new_pfam_path: Path to the cleaned pfam annotation file for each taxon
    :param folder_path: Path to the original .fa
    :param output_folder_path: Path to the cleaned .fa
    :return: None
    """
    if os.path.isdir(output_folder_path) is False:
        os.makedirs(output_folder_path)
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name == '.ipynb_checkpoints':
            continue
        file_path = os.path.join(folder_path, file_name)
        try:
            pfam_path = os.path.join(new_pfam_path+'ALL/', file_name.split('.')[0]+'.anno.pep.fa.csv')
            pfam_file = pd.read_csv(pfam_path, header=None)
        except:
            pfam_path = os.path.join(new_pfam_path+'ALL/', file_name.split('.')[0]+'.anno.pep.fa.pfamout.csv')
            pfam_file = pd.read_csv(pfam_path, header=None)
        new_path = output_folder_path+file_name+'.csv'
        p1 = re.compile(r'>(.*?)\n', re.S)  # 用于筛选蛋白质ID
        p2 = re.compile(r'\n(.*?)\n', re.S)  # 用于筛选出蛋白质序列
        pfam_file = pfam_file.sort_values(pfam_file.columns[0])
        pfam_file.drop_duplicates(subset = None, keep = 'first',inplace=True)
        pfam_ids = sorted(list(set(pfam_file[0].tolist())))

        with open(file_path, 'r') as fa_file:
            fa_file_info = fa_file.read()
            protein_ids = re.findall(p1, fa_file_info)
            proteins = re.findall(p2, fa_file_info)
            # print(file_name,len(protein_ids),len(proteins))
            for index, protein_id in enumerate(protein_ids):
                protein_id = protein_id.split(' ')[0]
                if protein_id in pfam_ids:
                    #原始的序列
                    sequence = ''
                    protein = proteins[index]
                    pfam_infos = pfam_file[pfam_file[0]==protein_id]
                    pfam_infos = pfam_file[pfam_file[0]==protein_id].sort_values([1])
                    for pfam_info in pfam_infos.iterrows():
                        info = list(list(pfam_info)[1])
                        start = info[1]-1
                        end =info[2]
                        sub_protein = protein[start:end]
                        sequence+= sub_protein
                    with open(new_path, 'a+') as new_fa_file:
                        writer = csv.writer(new_fa_file)
                        writer.writerow([protein_id, sequence])
                    new_fa_file.close()
        fa_file.close()
#Function for collecting kmers and frequencies
def cal_words(fa_folder_path, k ,output_folder_path):
    """
    :param fa_folder_path: Path to new .fa
    :param k: Size of Sliding Window
    :param output_folder_path: Path to save dict
    :return: None
    """
    if os.path.isdir(output_folder_path) is False:
        os.makedirs(output_folder_path)
    word_count= {}
    for file_name in tqdm(os.listdir(fa_folder_path)):
        if file_name == '.ipynb_checkpoints':
            continue
        Fa_file_path = os.path.join(fa_folder_path, file_name)
        Fa_file = pd.read_csv(Fa_file_path , header=None)
        Fa_file = Fa_file.sort_values(Fa_file.columns[0])
        for seq_list in Fa_file[1]:
            seq = seq_list
            for d in range(0, len(seq) - k + 1):
                word = seq[d:d+k]
                if word in word_count.keys():
                    word_count[word]+=1
                else:
                    word_count[word]=1
    top25 = sorted(list(word_count.values()))[int(len(word_count.keys())/4)]
    print('Number of collected of kmer:',len(word_count.keys()))
    print('Top 25% frequency of collected of kmer:', top25)
    dict_path = os.path.join(output_folder_path,str(k)+'.txt')
    with open(dict_path,'a') as dict_file:
        for key, value in word_count.items():
            dict_file.write(key + '\t' + str(value) + '\n')
    dict_file.close()
    with open(os.path.join(output_folder_path,str(k)+'mer.txt'),'a') as dict_file:
        for key, value in word_count.items():
            if value > top25:
                dict_file.write(key+'\n')
    dict_file.close()

def merge_folders(new_fa_path,determination):
    folders = os.listdir(new_fa_path)
    determination = determination
    if not os.path.exists(determination):
        os.makedirs(determination)
    for folder in folders:
        dir = new_fa_path + '\\' + str(folder)
        files = os.listdir(dir)
        for file in files:
            source = dir + '\\' + str(file)
            deter = determination + str(file)
            shutil.copyfile(source, deter)
#Function for doc2 generating(v=2)
def doc2_generator(fa_folder_path, label,dict_path,output_folder_path):
    """
    :param Path to new_fa of each taxon:
    :param label: Classification labels for each taxon
    :param dict_path: Path to User dictionary
    :param output_folder_path: Path to doc2
    :return: None
    """
    if os.path.isdir(output_folder_path) is False:
        os.makedirs(output_folder_path)
    jieba.load_userdict(dict_path)
    for file_name in tqdm(os.listdir(fa_folder_path)):
        if file_name == '.ipynb_checkpoints':
            continue
        doc = ''
        Fa_file_path = os.path.join(fa_folder_path, file_name)
        Fa_file = pd.read_csv(Fa_file_path , header=None)
        Fa_file = Fa_file.sort_values(0)
        doc_path = output_folder_path+'doc.csv'
        for seq_list in Fa_file.iterrows():
            seq = seq_list[1][1]
            word_list = jieba.cut(seq)
            sentence = ' '.join(word_list) + ' '
            doc+=sentence
            doc+='\n'
        with open(doc_path, 'a+') as doc_file:
            writer = csv.writer(doc_file)
            writer.writerow([file_name,label,doc])
        doc_file.close()
#Function for dataset1 generating(v=1)
def dataset1_generator(doc_path, var_number, taxon):
    """
    :param doc_path: Path to doc(v=1)
    :param var_number:
    :param taxon:
    :return: None
    """
    df = pd.read_csv(doc_path+'doc.csv', header=None)
    labels = df[1].tolist()
    species = df[0].tolist()
    counter = dict(Counter(labels))
    print('Original data info:',counter)
    for key, value in counter.items():
        var_number_taxon = var_number//value
        print('Number of mutations will be generated from', taxon[key-1],'taxon:',var_number_taxon)
        original_taxon = df[2][df[1]==key].tolist()
        original_species = df[0][df[1] == key].tolist()
        # print(original_taxon)
        print('Generating!')
        for i in range(0, len(original_taxon)):
            print(i)
            origin_list = original_taxon[i].split('\n')
            specie_name = original_species[i]
            length = len(origin_list)
            split_list = []
            for j in range(0, length, int(0.1 * length)):
                split_list.append(origin_list[j: j + int(0.1 * length)])
            variation = 1
            new_docs = []
            while variation <= var_number_taxon:
                random.shuffle(split_list)
                new_doc = sum(split_list, [])
                if new_doc not in new_docs:
                    new_docs.append(new_doc)
                    doc = ""
                    for n in new_doc:
                        if n != '':
                            sentence = n + '\n'
                            doc += sentence
                    with open(doc_path+'data.csv', 'a+') as file:
                        writer = csv.writer(file)
                        writer.writerow([specie_name, key, doc])
                    file.close()
                    # time.sleep(1)
                    variation += 1
#Fnction for dataset2 generating(v=2)
def dataset2_generator(doc_path, var_number, taxon):
    df = pd.read_csv(doc_path + 'doc.csv', header=None)
    labels = df[1].tolist()
    species = df[0].tolist()
    counter = dict(Counter(labels))
    print('Original data info:', counter)
    for key, value in counter.items():
        var_number_taxon = var_number // value
        print('Number of mutations will be generated from', taxon[key - 1], 'taxon:', var_number_taxon)
        original_taxon = df[2][df[1] == key].tolist()
        original_species = df[0][df[1] == key].tolist()
        # print(original_taxon)
        print('Generating!')
        for i in range(0, len(original_taxon)):
            origin_list = original_taxon[i].split('\n')
            specie_name = original_species[i]
            length = len(origin_list)
            split_list = []
            for j in range(0, length, int(0.1 * length)):
                split_list.append(origin_list[j: j + int(0.1 * length)])
            variation = 1
            new_docs = []
            while variation <= var_number_taxon:
                is_extra_varation = random.randint(0, 10)
                if is_extra_varation == 0:
                    random.shuffle(split_list)
                    new_doc = sum(split_list, [])
                else:
                    v = 1
                    vartion_length = int(0.2 * length)
                    vartion_loc = []
                    random.shuffle(split_list)
                    new_doc = sum(split_list, [])
                    while v < vartion_length:
                        vartion = random.randint(0, len(new_doc) - 1)
                        if vartion not in vartion_loc:
                            w_list = list(new_doc[vartion].split(' '))
                            l = random.randint(0, len(w_list) - 1)
                            try:
                                w = random.randint(0, len(w_list[l]) - 1)
                                replace_str = random.choice(
                                    ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
                                     'T', 'W', 'Y', 'V', 'U', 'O', 'X'])
                                new_doc[vartion].replace(w_list[l][w], replace_str, 1)
                                vartion_loc.append(vartion)
                                v += 1
                            except:
                                continue
                if new_doc not in new_docs:
                    new_docs.append(new_doc)
                    doc = ""
                    for n in new_doc:
                        sentence = n + '\n'
                        doc += sentence
                    with open(doc_path+'data.csv', 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow([specie_name, key, doc])
                    file.close()
                    variation += 1

#Function for splitting dataset
def train_test_val_split(fa_folder_path,ratio_train):
    """
    :param fa_folder_path: Path to data.csv
    :param ratio_train: Ratio for splitting
    :return:
    """
    df = pd.read_csv(fa_folder_path+'data.csv')
    train,test = train_test_split(df,test_size=1-ratio_train)
    test.to_csv(fa_folder_path+'test.csv', header=None,index=None)
    train.to_csv(fa_folder_path+'train.csv' ,header=None,index=None)
    return train,test

class data_pretrement:
    def __init__(self,args):
        self.pfam_path = args.pfam_path
        self.new_pfam_path = args.new_pfam_path
        self.taxon = args.taxon
        self.doc1_path =  args.doc1_path
        self.original_fa_path = args.original_fa_path
        self.new_fa_path = args.new_fa_path
        self.kmer_size = args.kmer_size
        self.dict_path = args.dict_path
        self.dict = self.dict_path+str(self.kmer_size)+'mer.txt'
        self.doc2_path = args.doc2_path
        self.var_number = args.var_number
        self.ratio = args.train_ratio
    def data_pretrement(self):
        print('Cleaning your Pfam annotation file!')
        clean_pfam_csv(self.pfam_path, self.new_pfam_path)
        time.sleep(1)
        print('Generating doc1(v=1)')
        for t in self.taxon:
            label = self.taxon.index(t)+1
            taxon_file_path = self.new_pfam_path+t+'/'
            t_fa_path = self.original_fa_path+t+'/'
            clean_fa(self.new_pfam_path, t_fa_path, self.new_fa_path+t+'/')
            doc1_generator(taxon_file_path, label, self.doc1_path)
        time.sleep(1)
        merge_folders(self.new_fa_path,self.new_fa_path+'ALL/')
        print('Scanning kmers and Generating words hierarchies!')
        cal_words(self.new_fa_path+'ALL/', self.kmer_size, self.dict_path)
        print('Generating doc2(v=2)')
        for t in self.taxon:
            label = self.taxon.index(t)+1
            taxon_fa_path = self.new_fa_path + t + '/'
            doc2_generator(taxon_fa_path, label, self.dict, self.doc2_path)
        print('Generating dataset1(v=1)!')
        dataset1_generator(self.doc1_path,self.var_number, self.taxon)
        print('Generating dataset2(v=2)!')
        dataset2_generator(self.doc2_path, self.var_number, self.taxon)
        print('Splitting datasets!')
        train_test_val_split(self.doc1_path,self.ratio)
        train_test_val_split(self.doc2_path, self.ratio)
        print('ALL done!')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='params for data-pretreatment')
    parser.add_argument("--pfam_path", type=str, default='pfam/')
    parser.add_argument("--new_pfam_path", type=str, default='new_pfam/')
    parser.add_argument("--taxon", type=list, default=['D','N'])
    parser.add_argument("--doc1_path", type=str, default='doc1/')
    parser.add_argument("--original_fa_path", type=str, default='data/original_fa/')
    parser.add_argument("--new_fa_path", type=str, default='data/new_fa/')
    parser.add_argument("--kmer_size", type=int, default=3)
    parser.add_argument("--dict_path", type=str, default='dict/')
    parser.add_argument("--doc2_path", type=str, default='doc2/')
    parser.add_argument("--var_number", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.7)



    args = parser.parse_args()
    # print(args.taxon)
    dp = data_pretrement(args)
    dp.data_pretrement()



