##############################################################################
# Problem 
# Description       :  Word Sense Disambiguation (WSD) is a technique to 
#                      used in finding the meaning of a word in a sentence. 
#                      A word can have multiple meanings and the exact meaning
#                      of word is decided based upon context by humans. 
#                      Computers also can follow similar technique i.e. decide 
#                      meaning of an ambiguous word in sentence using context 
#                      information. WSD is a technique for the same. In this 
#                      technique, we build certain characteristic (called as 
#                      features) from a set of data (training set) 
#                      which has sense of ambiguous words decided beforehand.
#                      Using this feature a naive Bayesian classifier is 
#                      trained. This classifier can take be used on other set
#                      of data (test set), which contains ambiguous words, to 
#                      determine their senses.
#
# Usage             : This program takes following inputs:
#                     1) -tr = the name of training file used for WSD. 
#                     2) -ts = the name of test file which needs to be tagged 
#                              with senses.
#                     3) -tk = the name of gold std file/ manually tagged file.
#                     e.g. to run this program for a training file 
#                     "hard-a.xml", test file "hard-a1.xml" and gold std,
#                      file "hard-a.key." use 
#                      following command
# 
# python WSD_naive_bayes.py -tr hard-a.xml -ts hard-a1.xml -tk hard-a.key 
#
#                     Please note that sequence of the inputs SHOULD be same as
#                     shown above. i.e. -tr <training file name> 
#                     -ts <test file name> -tk <key file name> 
#                    
#                     Also, this program used MontyLingua NLP toolkit developed
#                     by Hugo Liu at MIT Media Lab. This program must be 
#                     present in python directory of MontyLingua installation
#                     for proper working.
#                                
#                     MontyLingua can be downloaded from this link:
#                     http://web.media.mit.edu/~hugo/montylingua/
#
#                     This program creates an output file with name 
#                     "op_file", which contains the tagged senses for 
#                     words in test file. 
#                     It also creates a csv file containing confusion
#                     matrix of WSD, which denotes inaccuracies happened
#                     in the WSD i.e. percentage of times a sense is 
#                     incorrectly assigned to a word instead of other.
#                   
#                     Sample of train file data: 
#
#                     <instance id="hard-a.sjm-044_1:">
#                     <answer instance="hard-a.sjm-044_1:" senseid="HARD1"/>
#                     <context>
#                     <s> It is a sea of modernist vignettes -- 
#                     and while many of them are superb in themselves , it 's 
#                     <head>HARD</head> not to wonder if the </s> 
#                     </context>
#                     </instance>
#
#                     This sample has ambiguous word HARD tagged with sense 
#                     HARD1.
#
#                     Sample of test file data:
#
#                     <instance id="hard-a.sjm-259_8:">
#                     <context>
#                     <s> Eagles redux ; `` It 's <head>HARD</head> 
#                     to deny people like the beat of just plain rock , 
#                     " adds Olson . </s> 
#                     </context>
#                     </instance>
#                   
#                     This test sample requires ambiguous word HARD to be 
#                     tagged with correct sense.
#           
#                     Sample output file entry:
#           
#                     hard-a hard-a.sjm-259_8: HARD1
#                   
#                     This entry has the ambiguous word, instance id and
#                     actual sense decide by this program.
#                      
# Algorithm         : 1) This program first reads the training and test files 
#                        entered by user.
#                       
#                     2) It then extracts the WSD data like instance ids, 
#                        context sentences from training file. 
#               
#                     3) Using WSD data extracted above, this program extracts 
#                        collocational features and uses it train naive 
#                        Bayesian classifier.
#
#                     4) It then extracts WSD data from test file. And it 
#                        builds feature vectors for each context sentence
#                        in test data.
#                           
#                     5) It then calculates the prior and likelihood prob
#                        based upon the features extracted from training data
#                        and feature vectors of test data. These two prob are
#                        used to calculate the sense for ambiguous word. 
#
#                     6) The word-senses for test file is written into
#                        an output file. 
#
#                     7) This program then evaluates the accuracy of the WSD
#                        by comparing the tagging done against a manually 
#                        tagged gold std. file. It calculates the overall
#                        accuracy and confusion matrix for the WSD. 
#
# Author            : Swapnil Nawale 
#
# Date              : 11/06/2012
#
# Version           : 1.0
#
# Prog. Language    : Programming Language used for this program is Python
#                     (Version 2.7.3).
#                     The basic Python code, used in this program, is
#                     learnt from the book "Think Python - How to Think Like 
#                     a Computer Scientist (by Allen B. Downey)" and from
#                     Google's Python Class , present online at 
#                     http://code.google.com/edu/languages/google-python-class/
#                       
# Text-Editor used  : vim editor on Linux Platform
#
# Notes             : (A) OVERALL ACCURACY OF THE PROGRAM FOR:
#                        
#                       1) HARD           : 90.5457 %
#                       2) INTEREST       : 88.7640 %
#                       3) SERVE          : 84.1825 %
#                       4) LINE           : 78.2329 %
#                       5) MICROSOFTIBM   : 76.8215 %
#           
#                    (B) Bag-of-words feature are not used by this program.
#                        Since I got significant accuracies with collocational 
#                        features, I opted out not to go for Bag-of-words 
#                        features.
#                   
#                    (C) Excluding stop-words during feature extraction did not
#                        help much with accuracies, so stop-words are not 
#                        excluded by this program.
#
#                    (D) I got best accuracies for the window size of 2. As a 
#                        general trend with all words, increase in window size
#                        decreases the accuracy.
###############################################################################

#!/usr/bin/python

'''
import statements to include Python's in-built module functionalities in the
program
'''
# sys module is used to access command line argument, exit function etc.
import sys

# re module is used to access regular expression related facilities
import re

# os module is used to access file manipulation features
import os

# python csv module is used for pretty printing of confusion matrix
import csv

# math module for logarithmic functionalities
import math

# collections module is used for creating ordered hash tables / dicts 
import collections

# MontyLingua packages are used for features like POS-tagging & lemmatization
from MontyLingua import *

# time module for time related functionality
import time


'''
Set the value of debug flag. debug flag is used to decide whether to print
debug information in the output or not. This flag will be a global variable.
'''
debug = False

###############################################################################
# Function      : evaluate_tagging(op_file_name,  gold_std_file_name)
# Description   : This function calculates the overall accuracy of classifier
#                 done by comparison against manually tagged gold std file.
#                 It also produces a confusion matrix to show percentage
#                 of times a sense is wrongly tagged with another.
# Arguments     : op_file_name - The name of file tagged with word senses by
#                                this program
#                 gold_std_file_name - The name of manually tagged file
# Returns       : None.
###############################################################################

def evaluate_tagging(op_file_name,  gold_std_file_name):

    '''
    Open tagging output and gold standard file and read lines from them and
    close them.
    '''

    tag_op_file_handle = open(op_file_name,'r')
    gold_std_file_handle = open(gold_std_file_name, 'r')

    tagged_lines = tag_op_file_handle.readlines()
    gold_std_lines = gold_std_file_handle.readlines()

    tag_op_file_handle.close()
    gold_std_file_handle.close()

    '''
    Iterate over the tagged_lines and gold_std_line to search for
    wrongly tagged senses. If a sense is incorrectly tagged, then the correct 
    sense from gold std. file and wrong sense from output file will be found
    and put into a pair. This pair will be the keys in a dict object called
    as confusion_matrix_dict. The values for these keys in the dict will be
    the number of times the incorrect sense tagging is observed for 
    the key pair.
    '''
    # initialize the confusion_matrix_dict 
    confusion_matrix_dict = {}

    # maintain a counter to store the total number of incorrect tags
    incorrect_tags_count = 0

    for i in range(0, len(tagged_lines)):
        if debug:
            print tagged_lines[i]
            print gold_std_lines[i]

        tag_line = tagged_lines[i]
        gold_line = gold_std_lines[i] 
        
        if debug:
            print tag_line
            print gold_line
        
        '''
        Split the tag_line and gold_line by white spaces 
        '''
        word_tag_pair_1 = tag_line.split()
        word_tag_pair_2 = gold_line.split()

        if debug:
            print word_tag_pair_1
            print word_tag_pair_2
        
        word_sense_1 = word_tag_pair_1[2]
        word_sense_2 = word_tag_pair_2[2]
   
        
        '''
        If both tags are not equal then add an entry for the pair of senses to
        confusion_matrix_dict
        '''
        if word_sense_1 != word_sense_2:
                
            if (word_sense_2,word_sense_1) not in confusion_matrix_dict.keys():
                confusion_matrix_dict[(word_sense_2,word_sense_1)] = 1
            else:
                prev_count = confusion_matrix_dict[(word_sense_2,word_sense_1)]
                new_count = prev_count + 1
                confusion_matrix_dict[(word_sense_2,word_sense_1)] = new_count

            incorrect_tags_count =  incorrect_tags_count +1
        
    
    '''
    Print overall accuracy.
    '''
    print float(100) -((float(incorrect_tags_count) / \
    float(len(gold_std_lines))) *100)

    '''
    Get all the senses which appear in confusion_matrix dict. These senses only
    will be displayed in rows and columns of evaluation output. 
    These senses will be stored in a list tags_list.
    '''
    tags_list = [] 

    for tag in confusion_matrix_dict.keys():
        
        if tag[0] not in tags_list:
            tags_list.append(tag[0])
        
        if tag[1] not in tags_list:
            tags_list.append(tag[1])

    if debug:
        print tags_list
        print len(tags_list)

    '''
    Iterate over the tags_list and start printing confusion matrix.
    Calculate the percentage errors for each sense pair in confusion_matrixdict
    to display it in output. The output will be stored as a .csv file.
    Python provides an elegant csv module for creation of csv file. I have
    found that csv is an easiest way to get table like pretty printing of
    confusion matrix. The usage of csv module was learnt from an answer
    to a question on stackoverflow forum. It can be found here:

    http://stackoverflow.com/questions/2084069/create-a-csv-file-with-values-
    from-a-python-list

    I have followed the code in answer by stackoverflow user vy32.

    Python lists can directly be written into a csv file. So, form the list 
    that will represent the confusion matrix along with table row and col
    header.
    '''
    
    csv_list = []

    col_header = [' ']

    # Iterate over tags_list to get the tags as table col headers 

    for tag in tags_list:
        col_header.append(tag)

    # create csv file
    out = csv.writer(open("conf_matrix" + str(time.time()) + ".csv","w"),\
                     delimiter=',')
    
    if debug:
        print confusion_matrix_dict
        print col_header

    out.writerow(col_header)

    row_data = []
    # iterate over the tags_list to get table row headers and data
    for i in range(0,len(tags_list)):
        row_data = [tags_list[i]]
        for j in range(0,len(tags_list)):
            try:
                row_data.append((float(\
                          confusion_matrix_dict[(tags_list[i],tags_list[j])])/\
                            float(incorrect_tags_count))*float(100))
            except KeyError:
                row_data.append('-')

        out.writerow(row_data) 
    
    if debug:
        print incorrect_tags_count
        print csv_list
    
###############################################################################
# End of evaluate function
###############################################################################

###############################################################################
# Function      : get_WSD_data(file_name)
# Description   : This function WSD data (like word to disambiguated, its 
#                 instances, senses of the instances and contexts) from the 
#                 training and test files 
# Arguments     : file_name - Name of training / test file
# Returns       :  1) The word to be tagged, 
#                  2) A list containing all instance ids from training file
#                  3) A list containing all tagged senses for each instance
#                  (This list will be empty for the test file as senses will be
#                   tagged later.)
#                  4) A list containing all context sentences for each instance
###############################################################################

def get_WSD_data(file_name):

    # open the file in read mode
    file_handle = open(file_name, 'r')

    # read the contents of file into a list
    wsd_data_lines = file_handle.readlines()

    # close the file
    file_handle.close()

    '''
    Initialize three variables to hold :
    1) A list containing all instance ids from training file
    2) A list containing all tagged senses for each instance
    3) A list containing all context sentences for each instance
    '''
    instance_id_list = []
    sense_id_list = []
    context_sent_list = []
 
    '''
    Initialize a variable to hold the value of word to be tagged
    '''
    ambiguous_word = ""
    
    context_flag = False
    context_sent = ""

    '''
    Iterate over the wsd_data_lines list to separate out the word to be 
    disambiguated, instance ids, senses for those instances and the 
    context sentence for each instance. Fill in the three lists initialized
    above with the data separated out.
    '''

    for wsd_data_line in wsd_data_lines:

        if debug:
            print wsd_data_line


        '''
        Get the word to be disambiguated from the file. For this, check if
        the line starts with "<lexelt"  tag. If yes , then get the value of
        item attribute for this tag.
        '''
        
        if wsd_data_line.startswith('<lexelt'):
            
            ambiguous_word =  wsd_data_line[wsd_data_line.find("\"") + 1 :\
                                            wsd_data_line.rfind("\"")]
            
    
        '''
        Get the instance id of a word instance  from the file. 
        If a line starts with "<instace" tag, then get its id 
        attribute value.
        '''

        if wsd_data_line.startswith('<instance'):
            
            instance_id =  wsd_data_line[wsd_data_line.find("\"") + 1 :\
                                         wsd_data_line.rfind("\"")]
            
            '''
            If an instance id has a space in it then remove the characters
            starting from the space till end. This is required to get only
            the instance ids. This is a special case handling for 
            the training files like MicrosoftIBM file, where instance tag 
            has some other additional attributes like docsrc along with id
            attribute.
            '''

            if " " in instance_id:
                
                instance_id = instance_id[1 : instance_id.find(" ")-1]

            '''
            Add the retrieved instance id to instance_id_list
            '''
            instance_id_list.append(instance_id)

        '''
        Get the sense ids for each word instance from the file. For this, 
        check if the line starts with tag "<answer" tag. If yes, then get the
        value of senseid attribute. This processing won't happen for test 
        file as it does not have answer tags.
        '''

        if wsd_data_line.startswith('<answer'):
    
            sense_id_substr = wsd_data_line[wsd_data_line.find("senseid"):]

            sense_id = sense_id_substr[sense_id_substr.find("\"") + 1 :\
                                       sense_id_substr.rfind("\"")]
     
            '''  
            Add the retrieved sense id to sense_id_list
            '''
            sense_id_list.append(sense_id)

        '''
        Get the context sentences for each word instances. For this,retrieve
        all sentences which occur between "<context>" and "</context>" tags.
        '''

        if  wsd_data_line.startswith('<context>'):
            context_flag  =  True
        
        if context_flag == True:
            context_sent = context_sent + wsd_data_line
        
        
        if wsd_data_line.startswith('</context>'):
            '''
            Strip <context> start and end tags from context sentences and
            add the context sentences to context_sent_list
            '''
            context_sent_list.append(context_sent.replace("\n","").\
                                                  replace("<context>","").\
                                                  replace("</context>",""))
            context_sent = ""
            context_flag = False

    if debug:
        print context_sent_list
        print instance_id_list
        print sense_id_list
        print len(context_sent_list)
        print len(instance_id_list)
        print len(sense_id_list)
        print ambiguous_word

    # return the WSD data retrieved
    return ambiguous_word, instance_id_list, sense_id_list, context_sent_list

###############################################################################
# End of get_WSD_data function
###############################################################################

###############################################################################
# Function      : get_coll_features(sense_id_list, context_sent_list, 
#                                   window_size )
# Description   : This function extracts the collocation features from the
#                 training data. These features are used in learning the
#                 naive Bayesian classifier 
# Arguments     : sense_id_list - list containing all tagged senses 
#                                 for each instance
#
#                 context_sent_list - list containing all context sentences 
#                                     for each instance
#
#                 window-size - size of window to be considered to find 
#                               context words i.e. value for
#
#
# Returns       : 1) One dict object that has :
#
#                   i) The word-senses as the keys of dict  and 
#                   ii) The values for these keys are the lists of lists 
#                   showing lemmas of 'N1' context words on both right and 
#                   left sides of the ambiguous target word
#
#                 2) One dict object that has:
#
#                   i) The word-senses as the keys of dict  and 
#                   ii) The values for these keys are the lists of lists 
#                   showing POS tags of 'N1' context words on both right 
#                   and left sides of the ambiguous target word
###############################################################################

def get_coll_features(sense_id_list, context_sent_list, window_size):

    '''
    The steps involved in deriving collocation features from training data are
    as follows:

    1) Iterate over the list of context sentences derived from training data.

    2) Convert each context sentence to lower case and mark  the head word with
    a specific identifier @. The head words are already marked by <head> tag in
    context sentences but I want to remove all xml tags from context sentences
    in order to facilitate lemmatization process (explained below).

    3) Then remove all xml tags from the context sentences.

    4) After all this preprocessing, lemmatize and POs-tag the 
    context sentences using MontyLingua. MontyLingua is a NLP 
    toolkit developed by Hugo Liu at MIT Media Lab.

    This toolkit can be accessed from the link: 
    http://web.media.mit.edu/~hugo/montylingua
    
    This toolkit can be used to tokenize, lemmatize and POS-tag raw text.

    I was originally planning to use Standford CoreNLP for processing the
    context sentences, but the output files produced by Stanford CoreNLP 
    require more processing than MontyLingua. So, I chose MontyLingua over
    Standford CoreNLP. Also, as MontyLingua is developed in python, it could be
    easily integrated with this application.

    5) Extract the context word lemmas and their tags occurring within window 
    around head word. And put these lemmas and pos tags into 
    sense_context_words_mapping_dict and sense_pos_tags_mapping_dict dict 
    objects.

    These two dict objects represent our collocation feature for naive Bayes 
    classifier.
    '''

    # initialize the dict objects to be returned from this function
    sense_context_words_mapping_dict = {}
    sense_pos_tags_mapping_dict = {}

    query_obj = MontyLingua()
    instance_counter = 0 

    for context_sent in context_sent_list :

        # convert the sent into lower case
        context_sent =  context_sent.lower()

        # replace <head> tag into an identifier @
        context_sent = context_sent.replace(" <head>", " @").\
                                    replace("<head> ", "@ ")
        
        # remove all xml tags from the sent
        context_sent =  re.sub(r'<[/]?[\w\s\d=@]+>', r'', context_sent)
        
        # get the lemmas and pos tags for all words in sentence
        tagged_sent = query_obj.tag_tokenized(query_obj.tokenize(context_sent))
        lemmatized_sent = query_obj.lemmatise_tagged(tagged_sent)
   
        '''
        Get the context words and their tags which fall within the window size 
        '''
        # split the lemmatized sentence by spaces
        tags_list = lemmatized_sent.split() 

        # get the position of @ tag into tags_list
        identifier_pos = tags_list.index("@/IN/@")
        
        # split the tags_list to get left and right window list
        left_list = tags_list[0:identifier_pos]
        right_list =  tags_list[identifier_pos+2:]

        '''
        Add dummy elements to the left and right lists if their length is
        less than the window size
        '''

        if len(left_list) < window_size:
            temp_list  = left_list
            left_list = []
            dummy_element_count = window_size - len(temp_list)
            for i in range(0, dummy_element_count):
                left_list.append("dummyWord/DUMMY/dummyLemma")

            for tag in temp_list:
                left_list.append(tag)
    
            
        if len(right_list) < window_size:
            dummy_element_count = window_size - len(right_list)
            for i in range(0, dummy_element_count):
                right_list.append("dummyWord/DUMMY/dummyLemma")

       
        '''
        Extract the context words and their POS tags which fall inside
        the window size on both sides of target word. 
        
        If we consider the window size to be N1, then we need to extract
        last N1 elements from left list and first N1 elements from the 
        right list.

        These extracted elements will then be split to get the lemmas and 
        POS tags and they will be inserted into the dict objects 
        sense_context_words_mapping_dict & sense_pos_tags_mapping_dict. 
        These two dict objects will actually represent the collocational
        features for our WSD naive Bayesian Classifier.
        
        To extract the last N1 elements from left list, it can be reversed 
        first with reversed() built-in function. The usage of reversed function
        was referred from the answer given by "Greg Hewgill" for a related 
        question on the stackoverflow.com forum. The detailed question can be
        found here:

        http://stackoverflow.com/questions/529424/
        traverse-a-list-in-reverse-order-in-python
        '''
        lemma_list = []
        pos_tags_list = []

        loop_counter = 0

        for tag in reversed(left_list):
            
            if loop_counter == window_size:
                break

            tag_elements = tag.split("/")
            
            if tag_elements[2] != '':
                lemma_list.append(tag_elements[2])
            else:
                lemma_list.append(tag_elements[0])

            pos_tags_list.append(tag_elements[1])
        
            loop_counter = loop_counter + 1 
        
        lemma_list = list(reversed(lemma_list))
        pos_tags_list = list(reversed(pos_tags_list))
      
        

        right_tags_list  = right_list[0 : window_size]

        for tag in right_tags_list:

            tag_elements = tag.split("/")
     
            if tag_elements[2] != '':
                lemma_list.append(tag_elements[2])
            else:
                lemma_list.append(tag_elements[0])

            pos_tags_list.append(tag_elements[1])


        '''
        Insert the extracted lemma_list and  pos_tags_list into 
        sense_context_words_mapping_dict & sense_pos_tags_mapping_dict.
        The keys of these two dict objects will be sense ids for the current
        context sentence.
        '''
        curr_instance_sense = sense_id_list[instance_counter] 

        if curr_instance_sense not in sense_context_words_mapping_dict.keys():
            value_dict = []
            value_dict.append(lemma_list)
            sense_context_words_mapping_dict[curr_instance_sense] = value_dict
        
        else:
            value_dict = sense_context_words_mapping_dict[curr_instance_sense]
            value_dict.append(lemma_list)
            sense_context_words_mapping_dict[curr_instance_sense] = value_dict

        
        if curr_instance_sense not in sense_pos_tags_mapping_dict.keys():
            value_dict = [] 
            value_dict.append(pos_tags_list)
            sense_pos_tags_mapping_dict[curr_instance_sense] = value_dict
     
        else:
            value_dict = sense_pos_tags_mapping_dict[curr_instance_sense]
            value_dict.append(pos_tags_list)
            sense_pos_tags_mapping_dict[curr_instance_sense] = value_dict


        if debug:
            print left_list
            print right_list
            print lemma_list
            print pos_tags_list
            print sense_context_words_mapping_dict
            print sense_pos_tags_mapping_dict
            

        instance_counter = instance_counter + 1

    return sense_context_words_mapping_dict, sense_pos_tags_mapping_dict

###############################################################################
# End of get_coll_features function
###############################################################################

###############################################################################
# Function      : get_coll_feature_vector(context_sent, window_size)
# Description   : This function extracts the collocation feature vector for
#				  a test sentence.
# Arguments     : context_sent - the sentence containing an instance of
#							     ambiguous word for which collocational
#							     feature vector needs to be extracted
#                 window-size - size of window to be considered to find 
#                               context words i.e. value for N1
#                 query_obj - a MontyLingua object
# Returns       : 1) A list that has lemmas of the words occurring 
#	                 within the window size on both side of ambiguous word.
#				  2) A list that has POS_tags of the words occurring 
#	                 within the window size on both side of ambiguous word.
###############################################################################

def get_coll_feature_vector(context_sent, window_size, query_obj):

    # convert the sent into lower case
    context_sent =  context_sent.lower()

    # replace <head> tag into an identifier @
    context_sent = context_sent.replace(" <head>", " @").\
                                replace("<head> ", "@ ")
    
    # remove all xml tags from the sent
    context_sent =  re.sub(r'<[/]?[\w\s\d=@]+>', r'', context_sent)
    
    # get the lemmas and pos tags for all words in sentence
    tagged_sent = query_obj.tag_tokenized(query_obj.tokenize(context_sent))
    lemmatized_sent = query_obj.lemmatise_tagged(tagged_sent)

    '''
    Get the context words and their tags which fall within the window size 
    '''
    # split the lemmatized sentence by spaces
    tags_list = lemmatized_sent.split() 

    # get the position of @ tag into tags_list
    identifier_pos = tags_list.index("@/IN/@")
    
    # split the tags_list to get left and right window list
    left_list = tags_list[0:identifier_pos]
    right_list =  tags_list[identifier_pos+2:]

    '''
    Add dummy elements to the left and right lists if their length is
    less than the window size
    '''

    if len(left_list) < window_size:
        temp_list  = left_list
        left_list = []
        dummy_element_count = window_size - len(temp_list)
        for i in range(0, dummy_element_count):
            left_list.append("dummyWord/DUMMY/dummyLemma")

        for tag in temp_list:
            left_list.append(tag)

        
    if len(right_list) < window_size:
        dummy_element_count = window_size - len(right_list)
        for i in range(0, dummy_element_count):
            right_list.append("dummyWord/DUMMY/dummyLemma")

   
    '''
    Extract the context words and their POS tags which fall inside
    the window size on both sides of target word. 
    
    If we consider the window size to be N1, then we need to extract
    last N1 elements from left list and first N1 elements from the 
    right list.

    These extracted elements will then be split to get the lemmas and 
    POS tags and these lemmas and tags will represent collocational feature
    vector for test data.
    '''

    lemma_list = []
    pos_tags_list = []

    loop_counter = 0

    for tag in reversed(left_list):
        
        if loop_counter == window_size:
            break

        tag_elements = tag.split("/")
        
        if tag_elements[2] != '':
            lemma_list.append(tag_elements[2])
        else:
            lemma_list.append(tag_elements[0])

        pos_tags_list.append(tag_elements[1])
    
        loop_counter = loop_counter + 1 
    
    lemma_list = list(reversed(lemma_list))
    pos_tags_list = list(reversed(pos_tags_list))
  
    

    right_tags_list  = right_list[0 : window_size]

    for tag in right_tags_list:

        tag_elements = tag.split("/")
 
        if tag_elements[2] != '':
            lemma_list.append(tag_elements[2])
        else:
            lemma_list.append(tag_elements[0])

        pos_tags_list.append(tag_elements[1])


    return lemma_list, pos_tags_list

###############################################################################
# End of get_coll_feature_vector function
###############################################################################

###############################################################################
# Function      : get_coll_feature_prob(lemma_list, pos_tags_list, 
#                                    sense_context_words_mapping_dict,
#                                    sense_pos_tags_mapping_dict, sense_list)
# Description   : This function calculates the collocation feature Probabilities
#                 (likelihood Probabilities) for each word sense.
# Arguments     : lemma_list - list of lemmas of context words
#                 pos_tags_list - list of pos tags of context words
#                 sense_context_words_mapping_dict - dict storing mapping
#                 of senses to the context words
#                 sense_pos_tags_mapping_dict - dict storing mapping of
#                 senses to pos tags of context words
#                 senses_list - list of senses
# Returns       : 1) A dict object storing mapping of senses to their 
#                    likelihood Probabilities
###############################################################################

def get_coll_feature_prob(lemma_list, pos_tags_list, \
                          sense_context_words_mapping_dict,\
                          sense_pos_tags_mapping_dict, sense_list) : 

    '''
    This function will calculate the likelihood Probabilities for naive Bayes 
    classifier using the collocation feature vectors. Here we have extracted
    two types of collocational vectors: One that stores lemmas of context
    words and other that stores POS-tags of context words. I was planning 
    to use both of these feature vectors to get likelihood Probabilities
    but I noticed that usage of POS-tag based feature significantly reduced
    the accuracy of classifier. So, I used the lemmas of context words only
    for getting likelihood Probabilities.
    '''
    # initialize a dict object to store likelihood Probabilities
    sense_to_lkhd_mapping_dict = {}

    # iterate over the senses_list to get likelihood prob for each sense
    for sense in sense_list:
        if debug:
            print sense
            print lemma_list

        sense_context_words_lists = sense_context_words_mapping_dict[sense] 
        sense_context_tags_lists = sense_pos_tags_mapping_dict[sense]
        
        total_count_for_sense = len(sense_context_words_lists)
            
        if debug:
            print sense_context_words_lists
        
        feature_prob_list = []

        # get every feature from the lemma list
        for i in range(0,len(lemma_list)):
            
            feature_count = 0
    
            for context_word_list in sense_context_words_lists:
                '''
                Count the number of times a lemma occurs in training data at 
                specific position
                '''
                if lemma_list[i] == context_word_list[i]:
                    feature_count = feature_count + 1
            
            if feature_count != 0:
                feature_prob = float(feature_count) / \
                               float(total_count_for_sense)
            else:
                '''
                If a feature is unseen in training data, then do the smoothing 
                for it. Assign a very small value 10^-9 for unseen feature's
                likelihood prob
                '''
                feature_prob = pow(10,-9)

            feature_prob_list.append(feature_prob)

        lkhd_prob = 0

        # multiply all feature Probabilities in log space to avoid underflow
        for feature_prob in feature_prob_list:
            lkhd_prob = lkhd_prob + math.log10(feature_prob)
        
        # set the likelihood prob for each sense 
        sense_to_lkhd_mapping_dict[sense] = pow(10,lkhd_prob)
    
    return sense_to_lkhd_mapping_dict

###############################################################################
# End of get_coll_feature_prob function
###############################################################################

###############################################################################
# Function      : main()
# Description   : Entry point for the project.
# Arguments     : None. Command Line Arguments in Python are retrieved from
#                 sys.argv variable of sys module.
# Returns       : None.
###############################################################################
def main():
    
    '''
    Check if any command line argument is passed to program. If not 
    throw error showing proper sample usage. 
    '''

    if (len(sys.argv) > 1):
        if debug:
            print "At least one parameter passed to program !"

        '''
        Get the values for test, training and gold std. files
        from sys.argv command line arguments and store them into 
        different variables.
        '''

        train_file_name = sys.argv[2]
        test_file_name = sys.argv[4]
        gold_std_file_name = sys.argv[6]

        if debug:
            print train_file_name
            print test_file_name
        
        '''
        create the output file. It will have the name as op_file
        '''
        op_file_handle  = open("op_file", 'w')

        
        '''
        Retrieve the training data from the training file. Training file for 
        this application is an xml file, which has specific tags for various
        training data items used in WSD. The details of tags and corresponding 
        training data items are as follows:

        ----------------------------------------------------------------------
        |  Tag  |  Attribute  |     Usage / Data items represented by tag    |
        ----------------------------------------------------------------------
        |lexet  |  item       |  Ambiguous word to be tagged by WSD          |
        ----------------------------------------------------------------------
        |instance|  id        |  Instance id for an instance of word         |
        ----------------------------------------------------------------------
        |answer |  senseid    |  Word sense associated with word instance    |
        ----------------------------------------------------------------------
        |context| ----------- |  Context usage of word sense                 |
        ----------------------------------------------------------------------

        To retrieve above mentioned training data from training file, call
        get_WSD_data() function.

        This function takes name of file from which WSD data is to be retrieved
        as an input and returns following data items for the training file:

        1) The ambiguous word to be tagged, 
        2) A list containing all instance ids from training file
        3) A list containing all tagged senses for each instance
        4) A list containing all context sentences for each instance  
        '''
        ambiguous_word, instance_id_list, sense_id_list, context_sent_list = \
                                                get_WSD_data(train_file_name)


        '''
        Get the list of unique senses possible for an ambiguous word 
        from WSD data points retrieved above. 
        
        For this, in built 'set' function can be used. set
        function takes a list and converts it to a set eliminating duplicate 
        elements of the list. This set is required to be converted into list
        again for creating list out of this set. Usage of set function to 
        find distinct elements from a list was borrowed from a blog entry 
        present online at :
        
        http://mattdickenson.com/2011/12/31/find-unique-values-in-list-python/


        Also get the freq counts for each unique sense. These freq counts will
        be stored into a dict object sense_feq_dict, which has word senses as
        its keys and freq counts for each key sense as the values.
        '''

        # get the distinct senses and store them into a list
        sense_list = list(set(sense_id_list))

        if debug:
            print sense_list
        
        # initialize a dict obj to store the freq count for each sense
        sense_freq_dict = {}

        # iterate over the senses_list to get freq for each sense 
        for word_sense in sense_list:
            sense_freq_dict[word_sense] =  sense_id_list.count(word_sense) 
    
        if debug:
            print sense_freq_dict 
        
        '''
        Calculate the prior Probabilities for naive Bayesian classifier by using
        the freq counts from sense_freq_dict. The prior probability for a word
        sense is nothing but it's freq count in training data divided by total
        instances present in the training data.  The prior Probabilities 
        calculated here will be used later in WSD task and will be stored in a 
        dict object sense_to_prior_mapping_dict which has mapping of each sense
        to its prior probability.
        '''

        # initialize sense_to_prior_mapping_dict object
        sense_to_prior_mapping_dict = {}
        
        # calculate total number of ambiguous word instance
        total_count =  len(instance_id_list)

        # iterate over the sense_list to get the prior Probabilities
        for sense in sense_list:
            sense_to_prior_mapping_dict[sense] =\
            float(sense_freq_dict[sense]) / float(total_count)            


        '''
        Start building features for naive Bayesian classifier.

        The features extracted here follow the approaches given in text book:
        "Speech and Language processing" by Jurafsky-Martin (section 20.2.1)

         Features are extracted by this application are:
        
        1) Collocational features: The words and POS-tags of the words on right
        and left side of the word tagged with the sense within a specific 
        word window size (say 'N1').

        To retrieve collocational features for naive Bayesian classifier, call
        a function get_coll_features(). This function takes following inputs:

        a) list containing all tagged senses for each instance
        b) list containing all context sentences for each instance
        c) size of window to be considered to find context words i.e. value for
        N1

        And it returns two dict object
        
        First dict object has :

        i) The word-senses as the keys of dict  and 
        ii) The values for these keys are the lists of lists showing lemmas of 
        'N1' context words on both right and left sides of the 
        ambiguous target word

        Second dict object has:

        i) The word-senses as the keys of dict  and 
        ii) The values for these keys are the lists of lists showing POS tags 
        of 'N1' context words on both right and left sides of the ambiguous 
        target word
  

        e.g. If the training data for an ambiguous word 'interest ' has 
        two senses like 'interest_6' and 'interest_4' and the context sentences
        for these two senses are as follows:
        
        1) For 'interest_6':   <s> the firm has been racing to complete the 
                                transaction by its Oct. 15 deadline to avoid a 
                                bankruptcy filing , after having failed to make
                                <head>interest</head> payments in June on 
                                nearly $ 1 billion of debt . </s> 

        
        2) For 'interest_4':   <s> they say they represent the `` public 
                               <head>interest</head> '' but they do n't do so 
                               badly for their own *interests , either . </s> 
                                
                               <s> when international business machines hit a 
                               52-week low on tuesday , it stirred some 
                               <head>interest</head> among 
                               bottom-fishers . </s> 

                              

        And suppose we decide to have word-window size N1 as 2, then the 
        contents of two dict objects returned by get_coll_features() might 
        look something like this:

        First dict object:

        ----------------------------------------------------------------
        |  key          |       value                                  |
        ----------------------------------------------------------------
        | interest_6    |   [[to, make, payment, in]]                  |
        ----------------------------------------------------------------
        | interest_4    |   [[the,public,but,they],[stir,some,among,   |
        |               |   bottom-fishers]]                           |
        ----------------------------------------------------------------   

        Second dict object:

        ----------------------------------------------------------------
        |  key          |       value                                  |
        ----------------------------------------------------------------
        | interest_6    |   [[DT, VB, NNP, IN]]                        |
        ----------------------------------------------------------------
        | interest_4    |   [[DT,JJ,CC,PRP],[VB,DT,IN,                 |
        |               |   NNP]]                                      |
        ---------------------------------------------------------------- 
        '''

        # initialize variable for window size
        window_size = 2

        # call get_coll_features() function
        sense_context_words_mapping_dict, sense_pos_tags_mapping_dict = \
        get_coll_features(sense_id_list, context_sent_list, window_size)
 
        '''
        Get the WSD data items from test file by calling get_WSD_data() 
        function
        '''
        test_ambiguous_word, test_instance_id_list, test_sense_id_list, \
        test_context_sent_list = get_WSD_data(test_file_name)

        if debug:
            print test_ambiguous_word 
            print test_instance_id_list 
            print test_sense_id_list
            print test_context_sent_list

        '''
        Start finding word sense for each ambiguous word instance from the test 
        file. For this, first we need to get the feature vectors for each 
        context sentence in the test file. Using the feature vector, likelihood
        Probabilities for naive Bayes classifier are then calculated for each
        sense present in training file. These feature likelihood probability is 
        then multiplied by prior probability of each sense to get 
        final probability. 
        And the word-sense with highest final probability is selected for that
        given word sequence.
        '''
        instance_counter = 0 

        '''
        Create a MontyLingua object which is used in getting collocational
        feature vector in later processing.
        '''
        query_obj = MontyLingua()

        '''
        First iterate over the test_context_sent_list to get individual test 
        context sentences.
        '''
        
        for test_context_sent in test_context_sent_list:
            
            '''
            Extract the feature vector for each context sentence.
            Here two types of feature vectors are extracted for each sentence.
            
            1) Collocational feature vector : This vector will have two types 
            of features:

                a) First feature will be the lemmas of the words occurring 
                within the window size (which will have same value as N1 for 
                training value) on both side of ambiguous word.

                b) Second feature will be POS-tags for the words occurring 
                within the window size (which will have same value as N1 for 
                training value) on both side of ambiguous word.

            '''

            '''
            Call get_coll_feature_vector() function to get collocational 
            feature vector. This function takes context sentence window_size 
            and a MontyLingua object as the inputs and 
            returns the two collocational feature vectors for that 
            sentence.

            e.g. If a context sentence in the test data is 

            when we arrived in st. paul , the local office of the american 
            automobile association had a hard time directing us to bethel 
            college .  

            , where hard is the target word, and if window size is 9 then

            first collocation feature vector will be the list containing lemmas
            of 9 words present on right and left side of word 'hard'.
            
            
            ['local', 'office', 'of', 'the', 'american', 'automobile', 
            'association', 'have', 'a', 'time', 'direct', 'us', 'to', 'bethel',
            'college', '.', 'dummyLemma', 'dummyLemma']
            
            second collocation feature vector will be the list containing 
            pos-tags of 9 words present on right and left side of word 'hard'.

                    
            ['JJ', 'NN', 'IN', 'DT', 'JJ', 'NN', 'NN', 'VBD', 'DT', 'NN', 'VBG'
            , 'PRP', 'TO', 'NN', 'NN', '.', 'DUMMY', 'DUMMY']

            '''

            lemma_list, pos_tags_list = \
            get_coll_feature_vector(test_context_sent, window_size, query_obj)
            
            '''
            Get the likelihood Probabilities for each word sense for a given 
            instance of ambiguous word. 
            
            To get likelihood prob for collocational features, call 
            get_coll_feature_prob() function. 
            This function takes following inputs:
            
            1) List lemmas of context words retrieved above
            2) List of POS-tags of context words retrieved above
            3) Dict object mapping each sense to the list of context words
               (for all context sentences in training data), retrieved above
            4) Dict object mapping each sense to the list of POS-tags of 
               context words  (for all context sentences in training data),  
               retrieved above
            5) List of valid senses

            And it returns a dict object which maps each sense to its
            likelihood Probabilities
            '''

            sense_to_lkhd_mapping_dict =\
                            get_coll_feature_prob(lemma_list, pos_tags_list, \
                                  sense_context_words_mapping_dict,\
                                  sense_pos_tags_mapping_dict, sense_list)
        
            '''
            Multiply the prior and likelihood prob for each sense to get final
            Probabilities. Select sense with the maximum final prob as the sense
            for ambiguous word.
            '''


            # initialize a dict object to store final Probabilities
            final_prob_dict = collections.OrderedDict()
            final_prob_list = []
            
            for sense in sense_list:
                final_prob =  math.log10(sense_to_prior_mapping_dict[sense]) +\
                              math.log10(sense_to_lkhd_mapping_dict[sense])
                
                final_prob_dict[sense] = pow(10,final_prob)
                final_prob_list.append(pow(10,final_prob))


            max_prob_sense = final_prob_dict.keys()\
                             [final_prob_list.index(max(final_prob_list))]
    
        
            # write the max prob sense as the final sense into op file
            op_file_handle.write(test_ambiguous_word + " " +  \
                                 test_instance_id_list[instance_counter] +\
                                  " " + max_prob_sense + "\n")
            
            instance_counter = instance_counter + 1
            
        op_file_handle.close()


        '''
        Now that we have our final o/p file "op_file", compare it against
        the gold std file to assess overall accuracy of classifier.
        For this call evaluate_tagging function. 
        It takes following parameters:
        1) Name of the output file
        2) Name of gold std key file

        This function calculates overall accuracy of the classifier and also
        outputs the confusion matrix, which shows the percentage of times
        a word sense is wrongly tagged with other word sense.
        '''

        evaluate_tagging("op_file",  gold_std_file_name)

    else:
        if debug:
            print "No parameter passed to the program !"
    
        print "\n\tPlease provide proper inputs to the program !"
        print "\tSample usage: "
        print "\tpython WSD_naive_bayes.py -tr hat.xml -ts ha.xml -tk ha.key\n"
###############################################################################
# End of main function
###############################################################################

'''
Boilerplate syntax to specify that main() method is the entry point for 
this program.
'''

if __name__ == '__main__':
 
    main()

##############################################################################
# End of pos_tagging.py program
#############################################################################
