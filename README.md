# Song lyric classification

## General description

Engineer thesis project which aim is to create song emotions classification model basing on lyrics. It uses NLP methods and machine learning methods to classify a lyrics to one from below classes:
- angry
- happy
- sad
- relaxed

Each lyrics goes through the preprocessing process. To extract features word embedding method is applied.

Used machine learning models for classification:
 - multilayer perceptron (MLP)
 - long short-term memory (LSTM)
 - gate recurrent unit (GRU)
 - convolutional net (CNN)

There are also variants of these models (apart from GRU) which work on fragments (verses, choruses)
of lyrics. 

The project includes the web application which allows to predict a song emotion basing on its lyrics, using the created model.

Almost all code is written in Python. All needed packages are listed in `requirements.txt`. They can be downloaded:
```
pip install -r requiremtns.txt
```
The frontend of the web application is written using React.js.

## Getting dataset

The `.csv` files with dataset are located in `datasets` directory. It consists of training, validation and test datasets. 

The process of getting the datasets can be repeated by running scripts from `scripts/datasets` in the follwoing order:
  1. `download_raw_dataset` - it downloads raw data about songs and their emotions from three sources: [PMemo2019](https://github.com/HuiZhangDB/PMEmo), [emoMusic](http://cvml.unige.ch/databases/emoMusic/),
  [MoodyLyrics4Q](http://softeng.polito.it/erion/). All needed files are saved in `/datasets` folder.
  2. `merge_raw_datasets` - it merges three datasets into one. It drops redundant data, standarize and normalize data, extract classes (happy, relaxed, sad, angry) for those data which are in nummerical format of valance and  arousal values. The result is saved as `merged_datasets.csv` file.
  3. `get_lyrics_to_dataset` - it downloads the lyrics for the songs using  LyricsGenius API. It also download music genres for songs form `theaudiodb.com`. By means of `langdetect` library non-english songs are filtered out. The result is saved as `merged_datasets.csv` file. Beacaus of a lot of API requests an execution of the script can take long time.
  4. `filter_data_with_lyrics` - it filters out records from `merged_datasets.csv`:
     - deleting manually chosen data whoch consist incorrect or junk lyrics,
     - deleting records with too long lyrics,
     - deleting duplicated song data,
     - generalizing  music genres into 7 categories
    The result is saved as `filtered_dataset_with_lyrics.csv`.
  5. `split_dataset` - script splitting the dataset into training, validation and testing datasets.

## Text preprocessing

The project consist text preprocessing functions. They are located in file `text_preprocessing.py` in `preprocessing` folder. It consist of following functions:
- a basic preprocessing function which lowers words, removes punctuation and digits, removes redundant white spaces
- removing stop words function
- lemmatizating text function

During preprocessing some steps can be ommitted, e.g. removing stop words and/or lemmatization.

## Feature extraction

To extract features word embedding methodology is used. The project utilizes fastText method.

The class of fastText model `WordEmbedder` is located in `models/word_embedding/word_embedder.py`. It loads fastText model in `.bin` format from`models/word_embedding/saved_models`. The exact name of model can be passed in constant `_MODEL_PATH`. 

Getting embedding for a given word:
```
word = 'cat'
embedder = WordEmbedder()
embedding =  embedder[word]
```
There is a script `create_fasttext_model` which creates a fastText model in `bin` format which can be afterward used. It has following paramters:
 - `dim` - dimmension of a being created model, default: `200`.
 - `large_dataset` - boolean value indicating whether use a large dataset with lyrics or a training dataset `train_dataset.csv`, default: `True`.
    
    A large dataset is not included in the repository due to its large size. There are two ways of obtaining it:
    1. it can be download from [Song lyrics from 6 musical genres](https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres) and should be extracted in `datasets/lyrics-data` directory
    2. running `download_large_lyrics_dataset` from `scripts/datasets`. It demands to have `.json` file `kaggle.json` with your kaggle account token in your HOME directory
 - `remove_stopwords` - boolean value indicating whether to remove stopwords from dataset befor creating a fastext model
 - `lemmatization` - boolean value indicating whether to remove stopwords from dataset befor creating a fasText model
 A saved model is located in `models/word_embedding/saved_models`.
   
  ## Classification models
  
  ## Web application
    
 



