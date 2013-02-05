Word_Sense_Disambiguation
=========================

Word Sense Disambiguation using Naive Bayesian Classifier using Python

 Problem 
 Description       :  Word Sense Disambiguation (WSD) is a technique to 
                      used in finding the meaning of a word in a sentence. 
                      A word can have multiple meanings and the exact meaning
                      of word is decided based upon context by humans. 
                      Computers also can follow similar technique i.e. decide 
                      meaning of an ambiguous word in sentence using context 
                      information. WSD is a technique for the same. In this 
                      technique, we build certain characteristic (called as 
                      features) from a set of data (training set) 
                      which has sense of ambiguous words decided beforehand.
                      Using this feature a naive Bayesian classifier is 
                      trained. This classifier can take be used on other set
                      of data (test set), which contains ambiguous words, to 
                      determine their senses.

 Usage             : This program takes following inputs:
                     1) -tr = the name of training file used for WSD. 
                     2) -ts = the name of test file which needs to be tagged 
                              with senses.
                     3) -tk = the name of gold std file/ manually tagged file.
                     e.g. to run this program for a training file 
                     "hard-a.xml", test file "hard-a1.xml" and gold std,
                      file "hard-a.key." use 
                      following command
 
 python WSD_naive_bayes.py -tr hard-a.xml -ts hard-a1.xml -tk hard-a.key 

                     Please note that sequence of the inputs SHOULD be same as
                     shown above. i.e. -tr <training file name> 
                     -ts <test file name> -tk <key file name> 
                    
                     Also, this program used MontyLingua NLP toolkit developed
                     by Hugo Liu at MIT Media Lab. This program must be 
                     present in python directory of MontyLingua installation
                     for proper working.
                                
                     MontyLingua can be downloaded from this link:
                     http://web.media.mit.edu/~hugo/montylingua/

                     This program creates an output file with name 
                     "op_file", which contains the tagged senses for 
                     words in test file. 
                     It also creates a csv file containing confusion
                     matrix of WSD, which denotes inaccuracies happened
                     in the WSD i.e. percentage of times a sense is 
                     incorrectly assigned to a word instead of other.
                   
