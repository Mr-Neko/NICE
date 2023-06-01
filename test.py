from models.encoder_bert import convert_sents_to_features
from models.tokenization import BertTokenizer
import xml.etree.ElementTree as ET

from data.panoptic_narrative_grounding import Flickr30KDataset
from torch.utils.data import DataLoader
from utils.collate_fn import default_collate
import numpy as np
# def get_sentence_data(fn):
#     """
#     Parses a sentence file from the Flickr30K Entities dataset

#     input:
#       fn - full file path to the sentence file to parse
    
#     output:
#       a list of dictionaries for each sentence with the following fields:
#           sentence - the original sentence
#           phrases - a list of dictionaries for each phrase with the
#                     following fields:
#                       phrase - the text of the annotated phrase
#                       first_word_index - the position of the first word of
#                                          the phrase in the sentence
#                       phrase_id - an identifier for this phrase
#                       phrase_type - a list of the coarse categories this 
#                                     phrase belongs to

#     """
#     with open(fn, 'r') as f:
#         sentences = f.read().split('\n')

#     annotations = []
#     for sentence in sentences:
#         if not sentence:
#             continue

#         first_word = []
#         phrases = []
#         phrase_id = []
#         phrase_type = []
#         words = []
#         current_phrase = []
#         add_to_phrase = False
#         for token in sentence.split():
#             if add_to_phrase:
#                 if token[-1] == ']':
#                     add_to_phrase = False
#                     token = token[:-1]
#                     current_phrase.append(token)
#                     phrases.append(' '.join(current_phrase))
#                     current_phrase = []
#                 else:
#                     current_phrase.append(token)

#                 words.append(token)
#             else:
#                 if token[0] == '[':
#                     add_to_phrase = True
#                     first_word.append(len(words))
#                     parts = token.split('/')
#                     phrase_id.append(parts[1][3:])
#                     phrase_type.append(parts[2:])
#                 else:
#                     words.append(token)

#         sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
#         for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
#             sentence_data['phrases'].append({'first_word_index' : index,
#                                              'phrase' : phrase,
#                                              'phrase_id' : p_id,
#                                              'phrase_type' : p_type})

#         annotations.append(sentence_data)

#     return annotations

# def get_annotations(fn):
#     """
#     Parses the xml files in the Flickr30K Entities dataset

#     input:
#       fn - full file path to the annotations file to parse

#     output:
#       dictionary with the following fields:
#           scene - list of identifiers which were annotated as
#                   pertaining to the whole scene
#           nobox - list of identifiers which were annotated as
#                   not being visible in the image
#           boxes - a dictionary where the fields are identifiers
#                   and the values are its list of boxes in the 
#                   [xmin ymin xmax ymax] format
#     """
#     tree = ET.parse(fn)
#     root = tree.getroot()
#     size_container = root.findall('size')[0]
#     anno_info = {'boxes' : {}, 'scene' : [], 'nobox' : []}
#     for size_element in size_container:
#         anno_info[size_element.tag] = int(size_element.text)

#     for object_container in root.findall('object'):
#         for names in object_container.findall('name'):
#             box_id = names.text
#             box_container = object_container.findall('bndbox')
#             if len(box_container) > 0:
#                 if box_id not in anno_info['boxes']:
#                     anno_info['boxes'][box_id] = []
#                 xmin = int(box_container[0].findall('xmin')[0].text) - 1
#                 ymin = int(box_container[0].findall('ymin')[0].text) - 1
#                 xmax = int(box_container[0].findall('xmax')[0].text) - 1
#                 ymax = int(box_container[0].findall('ymax')[0].text) - 1
#                 anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
#             else:
#                 nobndbox = int(object_container.findall('nobndbox')[0].text)
#                 if nobndbox > 0:
#                     anno_info['nobox'].append(box_id)

#                 scene = int(object_container.findall('scene')[0].text)
#                 if scene > 0:
#                     anno_info['scene'].append(box_id)

#     return anno_info

# def strStr(haystack, needle) -> int:
#     def KMP(s, p):
#         nex = getNext(p)
#         i = 0
#         j = 0   # 分别是s和p的指针
#         while i < len(s) and j < len(p):
#             if j == -1 or s[i] == p[j]: # j==-1是由于j=next[j]产生
#                 i += 1
#                 j += 1
#             else:
#                 j = nex[j]

#         if j == len(p): # j走到了末尾，说明匹配到了
#             return i - j
#         else:
#             return -1

#     def getNext(p):
#         nex = [0] * (len(p) + 1)
#         nex[0] = -1
#         i = 0
#         j = -1
#         while i < len(p):
#             if j == -1 or p[i] == p[j]:
#                 i += 1
#                 j += 1
#                 nex[i] = j     # 这是最大的不同：记录next[i]
#             else:
#                 j = nex[j]

#         return nex
    
#     return KMP(haystack, needle)

# def convert_sents_to_features(sent, max_seq_length, tokenizer):
#     """Loads a data file into a list of `InputBatch`s."""

#     tokens_a = tokenizer.tokenize(sent.strip()) #list of words
#     # Account for [CLS] and [SEP] with "- 2"
#     if len(tokens_a) > max_seq_length - 2: #max_seq_length=230
#         tokens_a = tokens_a[:(max_seq_length - 2)] 
    
#     # Keep segment id which allows loading BERT-weights.
#     tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
#     segment_ids = [0] * len(tokens)

#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_mask = [1] * len(input_ids)

#     # Zero-pad up to the sequence length.
#     if max_seq_length == 231:
#         return input_ids

#     padding = [0] * (max_seq_length - len(input_ids))
#     input_ids += padding

#     assert len(input_ids) == max_seq_length

#     return input_ids

# tokenzier = BertTokenizer.from_pretrained('/root/NICE/pretrained_models/bert/bert-base-uncased.txt', do_lower_case=True)

# annotations_path = "/data/flickr30k/flickr30k_entities/annotations/Annotations/486263903.xml"
# sentence_path = "/data/flickr30k/flickr30k_entities/annotations/Sentences/486263903.txt"

# annotations = get_annotations(annotations_path)
# sentences = get_sentence_data(sentence_path)

# print(sentences)

# for sentence in sentences:

#     caption = sentence['sentence']
#     caption_token = convert_sents_to_features(caption, 230, tokenzier)

#     for i, phrase in enumerate(sentence['phrases']):

#         if phrase['phrase_id'] in annotations['boxes']:

#             noun_vector = [0] * len(caption_token)
#             phrase_token = convert_sents_to_features(phrase['phrase'], 231, tokenzier)[1: -1]
#             first_index = strStr(caption_token, phrase_token)
#             noun_vector[first_index: first_index+len(phrase_token)] = [i+1] * len(phrase_token)

# for boxes in annotations['boxes']:
#     box = [0] * 4
#     box[0] = annotations['width']
#     box[1] = annotations['height']
#     for b in annotations['boxes'][boxes]:
#         if box[0] > b[0]:
#             box[0] = b[0]
#         if box[1] > b[1]:
#             box[1] = b[1]
#         if box[2] < b[2]:
#             box[2] = b[2]
#         if box[3] < b[3]:
#             box[3] = b[3]
    
#     box[0] = box[0] / annotations['width']
#     box[1] = box[1] / annotations['height']
#     box[2] = box[2] / annotations['width']
#     box[3] = box[3] / annotations['height']
#     annotations['boxes'][boxes] = box



# A = convert_sents_to_features(sentenceA, 230, tokenzier)
# noun_vector = [0] * len(A)
# B = convert_sents_to_features(sentenceB, 231, tokenzier)[1: -1]

# first_index = strStr(A, B)
# noun_vector[first_index: first_index+len(B)] = [1] * len(B)
# print(A[noun_vector])

a = np.array([[1, 2], [3, 4]])

print(list(np.max(a, axis=0)))