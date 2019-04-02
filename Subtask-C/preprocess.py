##################################PREPROCESSING THE TWEETS###############################################
###WORKFLOW####
'''

1. CLEAN HTML ENTITIES AND OTHER UNICODE SYMBOLS.
2. REMOVE EMOJIS AND EMOTICONS.
3. REMOVE URLS.
4. REMOVE MENTIONS.
5. SEGMENT HASHTAGS.
6. EXPAND CONTRACTIONS.
7. LEMMATIZE THE WORDS.
8. REMOVE STOPWORDS.
9. HANDLE CENSORED WORDS.
10. REMOVING PUNCTUATORS.

########## INPUT ###############
# 
# LIST OF SENTENCES 
# EXAMPLE = ['Hi am hts is goign really great', 'Wow... that is so coool... When x=can i see him???!!']
# 
######## OUTPUT ##############
#
# LIST OF SENTENCES  
# 
'''




#################### CLEANING HTML STUFF ###################

import html
import unidecode

def clean_HTML(samples):
    filtered_samples = [ unidecode.unidecode(html.unescape(sample)) for sample in samples ]
    return filtered_samples

################### REMOCING EMOJIS AND EMTICONS ##############

# REPLACED BY APPROPRIATE TAGS AND ANNOTATIONS.

from ekphrasis.dicts import emoticons

emoticons_dict = emoticons.emoticons

def clean_emoticons(samples):
    filtered_samples = []
    for sample in samples:
        filtered_sample = ' '.join([re.sub(r"[<>]",'',emoticons_dict[word]) if word in emoticons_dict.keys() else word for word in sample.split()])
        filtered_samples.append(filtered_sample)
    return filtered_samples


import emoji
import re

emoji_re = emoji.get_emoji_regexp()

def clean_emojis(samples):
    # re.sub(r'(?::)(\w+){1,}_(\w+)(?::)', r' \1_\2 ') 
    # print(samples)
    filtered_samples = [emoji.demojize(sample, delimiters=[" "," "]) if emoji_re.search(sample) is not None else sample for sample in samples ]
    # print("After removing","\n",filtered_samples)
    return filtered_samples


################### REMOVING URLS ########################

import re

time_re = re.compile(r"(?:(?:\\d+)?\\.?\\d+(?:AM|PM|am|pm|a\\.m\\.|p\\.m\\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\\.m\\.|p\\.m\\.))?)")
url_re = re.compile(r"(?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})")
date_re = re.compile(r"(?:(?:(?:(?:(?<!:)\b\'?\d{1,4},? ?)?\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b(?:(?:,? ?\'?)?\d{1,4}(?:st|nd|rd|n?th)?\b(?:[,\/]? ?\'?\d{2,4}[a-zA-Z]*)?(?: ?- ?\d{2,4}[a-zA-Z]*)?(?!:\d{1,4})\b))|(?:(?:(?<!:)\b\'?\d{1,4},? ?)\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\b(?:(?:,? ?\'?)?\d{1,4}(?:st|nd|rd|n?th)?\b(?:[,\/]? ?\'?\d{2,4}[a-zA-Z]*)?(?: ?- ?\d{2,4}[a-zA-Z]*)?(?!:\d{1,4})\b)?))|(?:\b(?<!\d\.)(?:(?:(?:[0123]?[0-9][\.\-\/])?[0123]?[0-9][\.\-\/][12][0-9]{3})|(?:[0123]?[0-9][\.\-\/][0123]?[0-9][\.\-\/][12]?[0-9]{2,3}))(?!\.\d)\b))")
number_re = re.compile(r"\b\d+(?:[\.,']\d+)?\b")



def clean_tokens(samples):
    filtered_samples = []
    for sample in samples:
        filtered_sample = date_re.sub('', sample)
        filtered_sample = time_re.sub('', filtered_sample)
        filtered_sample = url_re.sub('', filtered_sample)
        
        #############################################################
        filtered_sample = re.sub(r'\b[uU][rR][lL]\b','',filtered_sample)
        #############################################################
        # This extra step is because our dataset is stupid.
        
        filtered_sample = number_re.sub('', filtered_sample)
        filtered_samples.append(filtered_sample)
    return filtered_samples


###################### SEGMENTING HASHTAGS ##########################

from ekphrasis.classes.segmenter import Segmenter

def clean_hashtags(samples):
    filtered_samples = []
    segmenter = Segmenter(corpus="twitter")
    for sample in samples:
        filtered_sample = ' '.join([segmenter.segment(word[1:]) if word.startswith('#') else word for word in sample.split()])
        filtered_samples.append(filtered_sample)
    return filtered_samples


##################### REMOVING METNIONS #############################

def clean_mentions(samples):
    filtered_samples = [re.sub(r'\@\w+','',sample) for sample in samples]
    return filtered_samples


#################### EXPANDING CONTRACTIONS ########################

def expand_contractions(samples):
    filtered_samples = []
    for text in samples:

        text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
        text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
        text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are",text)
        text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve",r"\1\2 have", text)

        text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
        text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
        text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
        text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
        text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
        text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
        text = re.sub(r"(\b)([Aa])(?:in't)", r"\1\2re not", text)
        filtered_samples.append(text)
 
    return filtered_samples


##################### LEMMATIZING STEMMING ###################


import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV
               }
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(samples):
    lemmatizer = WordNetLemmatizer()
    filtered_samples = []
    for sample in samples:
        filtered_samples.append(" ".join([lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in sample.split()]) )
    return filtered_samples



######################### REMOVING STOPWORDS ############################

import nltk
from nltk.corpus import stopwords

def remove_stop_words(samples):
    stopword_list = nltk.corpus.stopwords.words('english') 
    stopword_list.remove('no')
    stopword_list.remove('not')
    filtered_samples = []
    for sample in samples:
        filtered_samples.append(' '.join([word for word in sample.split() if word not in stopword_list]))
    return filtered_samples



################### HANDLING CENSORED DATA ############################


censored_re = re.compile(r'(?:\b\w+\*+\w+\b)', re.IGNORECASE)

def clean_censored_words(samples):
    filtered_samples = [ censored_re.sub('swear_word', sample) for sample in samples ]
    return filtered_samples

########################## REMOVING PUCNTUATORS ##########################

    
import string
def remove_punctuators(samples):
    # translator = str.maketrans('','', string.punctuation)
    # filtered_samples = [ sample.translate(translator) for sample in samples ]
    filtered_samples = [ re.sub(r'[^\w\s]','',sample).lower() for sample in samples ] 
    return filtered_samples








############################ PREPROCESS.PY ENDS HERE##########################
##############################################################################