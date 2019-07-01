# ExpandRank Algo Implementation with String Subsequence Kernel

## Table of Contents
- [Aim](#Aim)
- [Idea](#Idea)
- [Data](#Data)
- [Description of the experiment](#Description-of-the-experiment)

## Aim
To extract keywords from a document

## Idea
The idea here is to construct a knowledge context, that is to select k documents nearest to the the document from which key phrases have to be extracted. The graph based ranking algorithm is then applied on the expanded document set to make use of both the local information in the specified document and the global information in the neighbour documents.The [ExpandRank paper](references/Single_Document_Keyphrase_Extraction_Using_Neighborhood_Knowledge.pdf) uses cosine similarity for finding out the k nearest documents to the document from which keywords have to be extracted. This implementation uses [String Subsequence Kernel](references/Text_Classification_using_String_Kernels_lodhi.pdf) for measuring the similarity between two documents.

## Data
The experiment was performed for 1000 documents from [Hulth2003](references/Improved_Automatic_Keyword_Extraction_Given_More_Linguistic_Knowledge_Annet_Hulth.pdf) dataset (Training), which is a collection of abstracts from different papers each with a set of associated human labelled keywords. For each document, a score was calculated and then the average of the score over those 1000 documents was used to see how well the algorithm works. The dataset can be found [here](https://github.com/snkim/AutomaticKeyphraseExtraction)

## Description of the experiment
A detailed description of the experiment can be found [here](references/Description_of_the_experiment.pdf)
