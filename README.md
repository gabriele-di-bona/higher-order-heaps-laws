# Higher order Heaps' laws

Here we provide all the necessary code and instructions to analyse both datasets and model simulations. 
References to data repositories and code to simulate the models are provided.

In this analysis, we analyse the number $D_n(t)$ of novel combinations of $n$ = 1, 2, 3, or 4 consecutive elements in a ordered sequence of items/concepts/songs/words/events.
This way one can define the n-th order Heaps' laws as the power-law behavior of $D_n(t)$ as a function of the number of combinations $t$ in the sequence.

One can also study the distribution of the appearance of different combinations throughout the sequence through the calculation of the Shannon entropy of each label.

More thorough details are shown in the paper **_TODO_**.

In the rest of this repository, we refer to the root folder `~/` as the main folder where all the content of this repository is contained.

## Get and prepare data
All data and analysis must be saved into the subfolder `~/data/`

### Last.fm
The Last.fm dataset can be downloaded from http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html. In order to get the data in the right folders, use the following commands.
```
mkdir data
cd data
mkdir ocelma-dataset
cd ocelma-dataset
wget http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
tar -xzvf lastfm-dataset-1K.tar.gz 
```
Now, the path of the file to analyse should be `~/data/ocelma-dataset/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv`.

This file is processed in a related section of the jupyter notebook `~/notebooks/prep_data.ipynb`, in which a dataframe is created including the timestamp, track and artist name and MBID, for a total of 19150866 records.

The df is hence time ordered and cleaned, excluding outliers (two records are outside the date ranges), and only users with at least 1000 records have been retained (890 out of 992).
Moreover, tracks and artists are remapped to integer indices.
The final df contains data related to 890 users from 2005-02-14 to 2009-06-19.

Finally, in the same notebook, the sequences of each user are analyzed, calculating $n$-th order Heaps' laws fits, with $n$ = 1, 2, 3, and 4, as well as shannon entropies of the artists in the sequences.

### Gutenberg
The Gutenberg dataset can be downloaded using the github repository https://github.com/gabriele-di-bona/gutenberg. 
Clone this repository into `~/data/`. This will create a new subfolder `~/data/gutenberg/`.
Use the README.md of that repository to download all the books from Gutenberg.

All texts to be analyzed should now be in `~/data/gutenberg/data/text/` as `.txt` files. 
Similarly to Last.fm, these file are processed in a related section of the jupyter notebook `~/notebooks/prep_data.ipynb`. 
Here, the book is first lemmatized, discarding all books with less than 1000 words.
The language of the book is assessed using Google's cld3 package, and then words are also stemmed using the SnowballStemmer. 
In this case, we retain books with at least 1000 words, for a total of 19637 books.
```
mkdir data
cd data
mkdir ocelma-dataset
cd ocelma-dataset
wget http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
tar -xzvf lastfm-dataset-1K.tar.gz 
```
Now, the path of the file to analyse should be `~/data/ocelma-dataset/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv`.

This file is processed in a related section of the jupyter notebook, in which a dataframe is created including the timestamp, track and artist name and MBID.

The df is hence time ordered and cleaned, excluding outliers (two records are outside the date ranges), and only users with at least 1000 records have been retained.
Moreover, tracks and artists are remapped to integer indices.
The final df contains data related to 890 users from 2005-02-14 to 2009-06-19.

Finally, in the same notebook, the sequences of each user are analyzed, calculating $n$-th order Heaps' laws fits, with $n$ = 1, 2, 3, and 4, as well as shannon entropies of the artists in the sequences.

### Semantic Scholar
WARNING: the guide in this subsection has been written in early January 2022. Since then, the Semantic Scholar website has changed. The full corpus can still be downloaded, but it requires you to submit a form to obtain an API key.
As of February 2023, it seems like the present guide still works for the 2022-01-01 release.

#### Download data
For this project, we use one of the releases of the Semantic Scholar Academic Graph dataset (S2AG), which can be downloaded from https://api.semanticscholar.org/corpus/download/.

Release used in the paper: 2022-01-01 release.

In order to download the selected release, from the root folder of our repository, run the following commands:
```
mkdir -p ~/data
mkdir -p ~/data/semanticscholar
mkdir -p ~/data/semanticscholar/2022-01-01
mkdir -p ~/data/semanticscholar/2022-01-01/corpus
cd data/semanticscholar/2022-01-01/corpus/
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-01-01/manifest.txt
wget -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2022-01-01/ -i manifest.txt
```
WARNING: The downloaded corpus (made of 6000 zip files) weighs about 192GB.

#### Preprocess data
In the notebook `~/notebooks/semanticscholar_dataset_creation.ipynb` we create the sequences of words to be examined in the paper.
We run through the corpus looking at the journals with the highest number of english papers in each field. 
The language of the book is assessed using Google's cld3 package.
For each field of study (total of 19), we obtain a list of 1000 journals.
For each journal, we create a sequence containing, consecutively, the words in the titles of the papers, temporally ordered, of that journal.
Each title is indeed first lemmatized, and then words are also stemmed using the SnowballStemmer. .
Finally, for this dataset of 19000 sequences, we index all words and stems.

All this preprocessed data can be found in the created folder `~/data/semanticscholar/2022-01-01/data`. Here you can find the list of fields of study (`all_fieldsOfStudy.tsv`), the dictionaries to convert indices to words and words to their setms, as well as the folder `journals_fieldsOfStudy` containing 19 subfolders, one for each field, with the sequences of indexed words. A similar folder exists for the stems.

#### Analyse data
Finally, similarly to the previous data sets, the preprocessed files are analysed in a related section of the jupyter notebook `~/notebooks/prep_data.ipynb`. 

Each sequence of words is loaded and analyzed, calculating $n$-th order Heaps' laws fits, with $n$ = 1, 2, 3, and 4, as well as shannon entropies of the stems in the sequences.


## Models

### UMST model
This model has been proposed in Tria et al. (2014), "The dynamics of correlated novelties" (Scientific Reports). Other details about this model can be found in the paper.

Here we create UMST simulations through the python script `~/python_scripts/launch_UMST.py` with related bash script `~/bash_scripts/launch_UMST.sh`. These run simulations through the command of the type:
```
python launch_UMST.py -ID 1 -rho 20 -eta 1 -starting_nu 1 -ending_nu 20 -Tmax 100000 -putTogether False -save_all True
```
The script not only runs the simulation, but also calculates pairs and higher-order Heaps' laws, and entropies.
The results are saved into a file depending on the parameters chosen.
The complete file of a simulation with all the information can be found for example in `~/data/UMST/simulations/raw_sequences/rho_20.00000/nu_1/Tmax_100000/eta_1.00000/UMT_run_0.pkl`, where `run_0` in the file name is the run of the simulation, depending on the ID and the parameters.
In the same folder, one can also found smaller files, one with entropy calculations (`UMT_entropy_run_0.pkl`) and one with the main results in a lighter version (`UMT_light_run_0.pkl`).
Moreover, the original sequences of colors drawn and related labels can be found in other directories, for which a separate analysis can be done in the future, without need to store all the analysis results. For example, the related raw sequence can be found in `~/data/UMST/simulations/analysis/rho_20.00000/nu_1/Tmax_100000/eta_1.00000/UMT_run_0.pkl`

In order to calculate averages over multiple runs across the same set of parameters, launch the bash script `~/bash_scripts/UMST_put_together.sh`. The output is saved into the related folder (same as the light folder of the individual simulations), with name of the file `average_UMT_light_results.pkl`.

#### Analytical results of the UMT
The numerical integration of the differential equations found in the main paper has been done using the function `NIntegrate` of Mathematica 12, using a fine logarithmically spaced grid of N=1601 points $1 = t_0 < t_1 < \cdots < t_N = 10^{16}$. 
Such analysis is done in `~/data/UMST/Analytic_UMT/AnalyticExp_final.nb` and `~/data/UMST/Analytic_UMT/AnalyticExp_triplet.nb`.
The fit of the integrated points has instead been done in the Jupyter notebook `~/notebooks/figures.ipynb`.

### ERRW model
This model has been proposed in Iacopini et al. (2018), "Network Dynamics of Innovation Processes" (Physical Review Letters). Other details about this model can be found in the paper.

You can use `~/jupyter_notebooks/run_ERRW_on_SW_nets.ipynb` to generate some sequences of this model on a small world network (Watts Strogatz model).
You can do the same by using the python script `~/python_scripts/ERRW_SW.py` and its related bash script `~/bash_scripts/launch_ERRW.sh` to run multiple runs of the model separately.
You can also modify the network, either using the same graph model with different parameter $p$ or $K$, or changing the model. 

In this paper, we use sequences obtained from ERRWs on Small World networks with average degree $k=4$.
Raw sequences are saved into `~/data/ERRW/SW/raw_sequences/`, with different parameters of $p$ (for the small world network) and for dw (the reinforcement parameter in the ERRW).
These sequences are analysed through the python script `~/python_scripts/analyse_sequences.py` and related bash script `~/bash_scripts/analyse_sequences.sh` to launch them separately on each different sequence for all datasets and models through the cluster. The syntax is at follows.
```
python ~/python_scripts/analyse_sequences.py -ID N -folder "SW"
```
Adding the argument `-order False`, it considers the pairs BA and AB as the same, so order is not considered.
More info can be found in the python script.
The results of such analysis are saved into `~/data/ERRW/SW/analysis/` with the same file name. 


## Other scripts to analyse data

### Compute entropies
Although this is included in the main calculation of the analysis of a sequence, we have built a python script that computes the Shannon entropies in the sequences of labels. 
The computation is repeated on a number of reshuffles of the sequence for comparison with a null model of the same length, as well as on the sequences of pairs.

The python script is `~/python_scripts/compute_entropies.py` and gets some arguments, as can be seen in the related bash script `~/bash_scripts/compute_entropies.sh` to launch them separately on each different sequence for all datasets and models through the cluster.

To run the code on the N-th sequence of a specific dataset D (1 -> Last.fm, 2 -> Project Gutenberg, 3 -> Semantic Scholar, 4 -> UMST, 5 -> ERRW) with 10 reshuffles, just run
```
python ~/python_scripts/compute_entropies.py -ID N -number_reshuffles 10 -dataset_ID D
```
Adding the argument `-order False`, it considers the pairs BA and AB as the same, so order is not considered.
More info can be found in the python script.

ACHTUNG: the longer the sequence, the longer the computation of entropies! This does not scale quite well, so be careful.

### Analyse sequences
We provide another script to run the analysis of a sequence either of a dataset or of a model. 
Notice that, for the dataset, this doesn't do anything more that it is not already done in the related notebook to prepare the data. 
However, if one prefers to run the analysis in parallel, one can use this script instead.

The python script is `~/python_scripts/analyse_sequence.py` and gets some arguments, as can be seen in the related bash script `~/bash_scripts/analyse_sequence.sh` to launch them separately on each different sequence for all datasets and models through the cluster.

To run the code on the N-th sequence of a specific dataset D (1 -> Last.fm, 2 -> Project Gutenberg, 3 -> Semantic Scholar, 4 -> UMST, 5 -> ERRW) with 10 reshuffles, just run
```
python ~/python_scripts/analyse_sequences.py -ID N -number_reshuffles 10 -dataset_ID D \
    --consider_temporal_order_in_tuples True --analyse_sequence_labels False \
    -save_all True \
    -rho 4 -starting_nu 1 -ending_nu 40 -eta .1 -Tmax 100000 \
    --folder_name_ERRW "SW"
```
Adding the argument `--consider_temporal_order_in_tuples False`, it considers the pairs BA and AB as the same, so order is not considered.
If the data saved occupies too much space, one can consider to save only the light version of the results, with `-save_all False`.
The parameters `-rho`, ... , `Tmax` refer only to UMST, while `--folder_name_ERRW` refers only to ERRW.
More info can be found in the python script.

## Figures
All data and models are loaded and further analysed in the Jupyter notebook `~/notebooks/figures.ipynb`.
All the figures, tables and measures found in the paper related to this repository has been made through this notebook.