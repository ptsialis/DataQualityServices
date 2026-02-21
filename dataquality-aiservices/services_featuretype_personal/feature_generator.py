#Copyright 2024 Mücahit Sahin
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import joblib
import warnings
warnings.filterwarnings("ignore")
import os

# Get current working directory
current_path = os.getcwd()


vectorizerName = joblib.load("./data/Dictionary/dictionaryName.pkl")
vectorizerSample = joblib.load("./data/Dictionary/dictionarySample.pkl")


del_pattern = r'([^,;\|]+[,;\|]{1}[^,;\|]+){1,}'
del_reg = re.compile(del_pattern)

delimeters = r"(,|;|\|)"
delimeters = re.compile(delimeters)

url_pat = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
url_reg = re.compile(url_pat)

email_pat = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b"
email_reg = re.compile(email_pat)

stop_words = set(stopwords.words('english'))


# def get_sample(dat, key_s):
#     rand = []
#     for name in key_s:
#         rand_sample = list(pd.unique(dat[name]))
#         rand_sample = rand_sample[:5]
#         while len(rand_sample) < 5:
#             rand_sample.append(list(pd.unique(dat[name]))[np.random.randint(len(list(pd.unique(dat[name]))))])
#         rand.append(rand_sample[:5])
#     return rand





def get_sample(dat, key_s):
    rand = []
    for name in key_s:
        rand_sample = list(pd.unique(dat[name]))
        k = []
        while len(k) < 5:
            k.append(list(pd.unique(dat[name]))[np.random.randint(len(list(pd.unique(dat[name]))))])

        rand.append(k)

    return rand




# def get_sample(dat, key_s, n=5):
#     rand = []
#     for name in key_s:
#         # Unique values for this column (ohne NaNs)
#         vals = pd.unique(dat[name].dropna())

#         if len(vals) == 0:
#             # Falls die Spalte nur NaNs hat
#             rand.append([None] * n)
#             continue

#         if len(vals) >= n:
#             # Es gibt genug eindeutige Werte -> zieh n ohne Replacement
#             k = np.random.choice(vals, size=n, replace=False)
#             rand.append(k.tolist())
#         else:
#             # Weniger Unique-Werte als n:
#             # 1) Nimm alle Unique-Werte
#             base = vals.tolist()
#             # 2) Fülle zufällig mit Wiederholung auf bis n
#             extra = np.random.choice(vals, size=n - len(vals), replace=True)
#             rand.append(base + extra.tolist())

#     return rand





def summary_stats(dat, key_s):
    b_data = []
    for col in key_s:
        nans = np.count_nonzero(pd.isnull(dat[col]))
        dist_val = len(pd.unique(dat[col].dropna()))
        Total_val = len(dat[col])
        mean = np.NaN
        std_dev = np.NaN
        var = np.NaN
        min_val = np.NaN
        max_val = np.NaN
        if is_numeric_dtype(dat[col]):
            mean = np.mean(dat[col])
            
            if pd.isnull(mean):
                mean = 0
                std_dev = 0
                #var = 0
                min_val = 0
                max_val = 0           
            else:    
                std_dev = np.std(dat[col])
                var = np.var(dat[col])
                min_val = float(np.min(dat[col]))
                max_val = float(np.max(dat[col]))
        b_data.append([Total_val, nans, dist_val, mean, std_dev, min_val, max_val])
    return b_data     

def get_ratio_dist_val(summary_stat_result):
    ratio_dist_val = []
    for r in summary_stat_result:
        ratio_dist_val.append(r[2]*100.0 / r[0])
    return ratio_dist_val

def get_ratio_nans(summary_stat_result):
    ratio_nans = []
    for r in summary_stat_result:
        ratio_nans.append(r[1]*100.0 / r[0])
    return ratio_nans    


# def FeaturizeFile(df):
# 	# df = pd.read_csv(CSVfile,encoding = 'latin1')

# 	stats = []
# 	attribute_name = []
# 	sample = []
# 	id_value = []
# 	i = 0

# 	castability = []
# 	number_extraction = []

# 	avg_tokens = []
# 	ratio_dist_val = []
# 	ratio_nans = []

# 	keys = list(df.keys())

# 	attribute_name.extend(keys)
# 	summary_stat_result = summary_stats(df, keys)
# 	stats.extend(summary_stat_result)
# 	samples = get_sample(df,keys)
# 	sample.extend(samples)


# 	# castability.extend(castability_feature(df, keys))
# 	# number_extraction.extend(numeric_extraction(df, keys))

# 	# avg_tokens.extend(get_avg_tokens(samples))
# 	ratio_dist_val.extend(get_ratio_dist_val(summary_stat_result))
# 	ratio_nans.extend(get_ratio_nans(summary_stat_result))


# 	csv_names = ['Attribute_name', 'total_vals', 'num_nans', 'num_of_dist_val', 'mean', 'std_dev', 'min_val',
# 	             'max_val', '%_dist_val', '%_nans', 'sample_1', 'sample_2', 'sample_3','sample_4','sample_5'
# 	            ]
# 	golden_data = pd.DataFrame(columns = csv_names)

# 	for i in range(len(attribute_name)):
#         # print(attribute_name[i])
#         val_append = []
#         val_append.append(attribute_name[i])
#         val_append.extend(stats[i])
        
#         val_append.append(ratio_dist_val[i])
#         val_append.append(ratio_nans[i])    
        
#         val_append.extend(sample[i])
#     #     val_append.append(castability[i])
#     #     val_append.append(number_extraction[i])
#     #     val_append.append(avg_tokens[i])

#         golden_data.loc[i] = val_append
#     #     print(golden_data)


# 	curdf = golden_data

# 	for row in curdf.itertuples():

# 	    # print(row[11])
# 	    is_list = False
# 	    curlst = [row[11],row[12],row[13],row[14],row[15]]
	    
# 	    delim_cnt,url_cnt,email_cnt,date_cnt =0,0,0,0
# 	    chars_totals,word_totals,stopwords,whitespaces,delims_count = [],[],[],[],[]
	    
# 	    for value in curlst: 
# 	        word_totals.append(len(str(value).split(' ')))
# 	        chars_totals.append(len(str(value)))
# 	        whitespaces.append(str(value).count(' '))
	        
# 	        if del_reg.match(str(vadef FeaturizeFile(df):
# 	# df = pd.read_csv(CSVfile,encoding = 'latin1')

# 	stats = []
# 	attribute_name = []
# 	sample = []
# 	id_value = []
# 	i = 0

# 	castability = []
# 	number_extraction = []

# 	avg_tokens = []
# 	ratio_dist_val = []
# 	ratio_nans = []

# 	keys = list(df.keys())

# 	attribute_name.extend(keys)
# 	summary_stat_result = summary_stats(df, keys)
# 	stats.extend(summary_stat_result)
# 	samples = get_sample(df,keys)
# 	sample.extend(samples)


# 	# castability.extend(castability_feature(df, keys))
# 	# number_extraction.extend(numeric_extraction(df, keys))

# 	# avg_tokens.extend(get_avg_tokens(samples))
# 	ratio_dist_val.extend(get_ratio_dist_val(summary_stat_result))
# 	ratio_nans.extend(get_ratio_nans(summary_stat_result))


# 	csv_names = ['Attribute_name', 'total_vals', 'num_nans', 'num_of_dist_val', 'mean', 'std_dev', 'min_val',
# 	             'max_val', '%_dist_val', '%_nans', 'sample_1', 'sample_2', 'sample_3','sample_4','sample_5'
# 	            ]
# 	golden_data = pd.DataFrame(columns = csv_names)

# 	for i in range(len(attribute_name)):
#         # print(attribute_name[i])
#         val_append = []
#         val_append.append(attribute_name[i])
#         val_append.extend(stats[i])
        
#         val_append.append(ratio_dist_val[i])
#         val_append.append(ratio_nans[i])    
        
#         val_append.extend(sample[i])
#     #     val_append.append(castability[i])
#     #     val_append.append(number_extraction[i])
#     #     val_append.append(avg_tokens[i])

#         golden_data.loc[i] = val_append
#     #     print(golden_data)


# 	curdf = golden_data

# 	for row in curdf.itertuples():

# 	    # print(row[11])
# 	    is_list = False
# 	    curlst = [row[11],row[12],row[13],row[14],row[15]]
	    
# 	    delim_cnt,url_cnt,email_cnt,date_cnt =0,0,0,0
# 	    chars_totals,word_totals,stopwords,whitespaces,delims_count = [],[],[],[],[]
	    
# 	    for value in curlst: lue)):  delim_cnt += 1    
# 	        if url_reg.match(str(value)):  url_cnt += 1
# 	        if email_reg.match(str(value)):  email_cnt += 1
	        
# 	        delims_count.append(len(delimeters.findall(str(value))))        
	    
# 	        tokenized = word_tokenize(str(value))
# 	        # print(tokenized)
# 	        stopwords.append(len([w for w in tokenized if w in stop_words]))    
	    
# 	        try:
# 	            _ = pd.Timestamp(value)
# 	            date_cnt += 1
# 	        except ValueError: date_cnt += 0    
	    
# 	    # print(delim_cnt,url_cnt,email_cnt)
# 	    if delim_cnt > 2:  curdf.at[row.Index, 'has_delimiters'] = True
# 	    else: curdf.at[row.Index, 'has_delimiters'] = False

# 	    if url_cnt > 2:  curdf.at[row.Index, 'has_url'] = True
# 	    else: curdf.at[row.Index, 'has_url'] = False
	        
# 	    if email_cnt > 2:  curdf.at[row.Index, 'has_email'] = True
# 	    else: curdf.at[row.Index, 'has_email'] = False   
	        
# 	    if date_cnt > 2:  curdf.at[row.Index, 'has_date'] = True
# 	    else: curdf.at[row.Index, 'has_date'] = False           
	        
# 	    curdf.at[row.Index, 'mean_word_count'] = np.mean(word_totals)
# 	    curdf.at[row.Index, 'std_dev_word_count'] = np.std(word_totals)
	    
# 	    curdf.at[row.Index, 'mean_stopword_total'] = np.mean(stopwords)
# 	    curdf.at[row.Index, 'stdev_stopword_total'] = np.std(stopwords)
	    
# 	    curdf.at[row.Index, 'mean_char_count'] = np.mean(chars_totals)    
# 	    curdf.at[row.Index, 'stdev_char_count'] = np.std(chars_totals)
	    
# 	    curdf.at[row.Index, 'mean_whitespace_count'] = np.mean(whitespaces)
# 	    curdf.at[row.Index, 'stdev_whitespace_count'] = np.std(whitespaces)    
	    
# 	    curdf.at[row.Index, 'mean_delim_count'] = np.mean(whitespaces)
# 	    curdf.at[row.Index, 'stdev_delim_count'] = np.std(whitespaces)      
	    
# 	    if curdf.at[row.Index, 'has_delimiters'] and curdf.at[row.Index, 'mean_char_count'] < 100: curdf.at[row.Index, 'is_list'] = True    
# 	    else: curdf.at[row.Index, 'is_list'] = False
	    
# 	    if curdf.at[row.Index, 'mean_word_count'] > 10: curdf.at[row.Index, 'is_long_sentence'] = True    
# 	    else: curdf.at[row.Index, 'is_long_sentence'] = False    
	    
# 	    # print(np.mean(stopwords))
	    
# 	    # print('\n\n\n')

# 	golden_data = curdf


# 	return golden_data	

def FeaturizeFile(df):
    # df = pd.read_csv(CSVfile,encoding = 'latin1')

    stats = []
    attribute_name = []
    sample = []
    id_value = []
    i = 0

    castability = []
    number_extraction = []

    avg_tokens = []
    ratio_dist_val = []
    ratio_nans = []

    keys = list(df.keys())

    attribute_name.extend(keys)
    summary_stat_result = summary_stats(df, keys)
    stats.extend(summary_stat_result)
    samples = get_sample(df, keys)
    sample.extend(samples)


    # castability.extend(castability_feature(df, keys))
    # number_extraction.extend(numeric_extraction(df, keys))

    # avg_tokens.extend(get_avg_tokens(samples))
    ratio_dist_val.extend(get_ratio_dist_val(summary_stat_result))
    ratio_nans.extend(get_ratio_nans(summary_stat_result))

    csv_names = [
        'Attribute_name', 'total_vals', 'num_nans', 'num_of_dist_val',
        'mean', 'std_dev', 'min_val', 'max_val', '%_dist_val', '%_nans',
        'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5'
    ]
    golden_data = pd.DataFrame(columns=csv_names)

    for i in range(len(attribute_name)):
        # print(attribute_name[i])
        val_append = []
        val_append.append(attribute_name[i])
        val_append.extend(stats[i])

        val_append.append(ratio_dist_val[i])
        val_append.append(ratio_nans[i])

        val_append.extend(sample[i])
        # val_append.append(castability[i])
        # val_append.append(number_extraction[i])
        # val_append.append(avg_tokens[i])

        golden_data.loc[i] = val_append
        # print(golden_data)

    curdf = golden_data

    for row in curdf.itertuples():

        # print(row[11])
        is_list = False
        curlst = [row[11], row[12], row[13], row[14], row[15]]

        delim_cnt, url_cnt, email_cnt, date_cnt = 0, 0, 0, 0
        chars_totals, word_totals, stopwords, whitespaces, delims_count = [], [], [], [], []

        for value in curlst:
            word_totals.append(len(str(value).split(' ')))
            chars_totals.append(len(str(value)))
            whitespaces.append(str(value).count(' '))

            if del_reg.match(str(value)):
                delim_cnt += 1
            if url_reg.match(str(value)):
                url_cnt += 1
            if email_reg.match(str(value)):
                email_cnt += 1

            delims_count.append(len(delimeters.findall(str(value))))

            tokenized = word_tokenize(str(value))
            # print(tokenized)
            stopwords.append(len([w for w in tokenized if w in stop_words]))

            try:
                _ = pd.Timestamp(value)
                date_cnt += 1
            except ValueError:
                date_cnt += 0

        # print(delim_cnt,url_cnt,email_cnt)
        if delim_cnt > 2:
            curdf.at[row.Index, 'has_delimiters'] = True
        else:
            curdf.at[row.Index, 'has_delimiters'] = False

        if url_cnt > 2:
            curdf.at[row.Index, 'has_url'] = True
        else:
            curdf.at[row.Index, 'has_url'] = False

        if email_cnt > 2:
            curdf.at[row.Index, 'has_email'] = True
        else:
            curdf.at[row.Index, 'has_email'] = False

        if date_cnt > 2:
            curdf.at[row.Index, 'has_date'] = True
        else:
            curdf.at[row.Index, 'has_date'] = False

        curdf.at[row.Index, 'mean_word_count'] = np.mean(word_totals)
        curdf.at[row.Index, 'std_dev_word_count'] = np.std(word_totals)

        curdf.at[row.Index, 'mean_stopword_total'] = np.mean(stopwords)
        curdf.at[row.Index, 'stdev_stopword_total'] = np.std(stopwords)

        curdf.at[row.Index, 'mean_char_count'] = np.mean(chars_totals)
        curdf.at[row.Index, 'stdev_char_count'] = np.std(chars_totals)

        curdf.at[row.Index, 'mean_whitespace_count'] = np.mean(whitespaces)
        curdf.at[row.Index, 'stdev_whitespace_count'] = np.std(whitespaces)

        curdf.at[row.Index, 'mean_delim_count'] = np.mean(whitespaces)
        curdf.at[row.Index, 'stdev_delim_count'] = np.std(whitespaces)

        if curdf.at[row.Index, 'has_delimiters'] and curdf.at[row.Index, 'mean_char_count'] < 100:
            curdf.at[row.Index, 'is_list'] = True
        else:
            curdf.at[row.Index, 'is_list'] = False

        if curdf.at[row.Index, 'mean_word_count'] > 10:
            curdf.at[row.Index, 'is_long_sentence'] = True
        else:
            curdf.at[row.Index, 'is_long_sentence'] = False

        # print(np.mean(stopwords))

        # print('\n\n\n')

    golden_data = curdf

    return golden_data





def FeatureExtraction(data, useSamples=0):

    data1 = data[['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean', 'std_dev', 'min_val', 'max_val','has_delimiters', 'has_url', 'has_email', 'has_date', 'mean_word_count',
       'std_dev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
       'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
       'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count',
       'is_list', 'is_long_sentence']]
    data1 = data1.reset_index(drop=True)
    data1 = data1.fillna(0)

    arr = data['Attribute_name'].values
    arr = [str(x) for x in arr]
    
    X = vectorizerName.transform(arr)    
    attr_df = pd.DataFrame(X.toarray())

    
    data2 = pd.concat([data1, attr_df], axis=1, sort=False)
        
    return data2


def get_feature_results(df, feature, pipe=None, stopped_index=0, save_path=None):
    
    df_feature= FeaturizeFile(df)
    featurized_df = FeatureExtraction(df_feature)

    if feature == None:
        return 'Please specify the feature you want to retrieve results from.'
    for idx in range(stopped_index, len(df_feature)):
        row = df_feature[idx:idx+1].reset_index(drop=True)
        
        if not pipe:
            pred = None
            for i in range(6):
                if not pred == None:
                    break                    
                elif i == 5:
                    #print(f'////////// Stopped at index {idx} //////////')
                    return featurized_df
                else:
                    pred = feature(row, pipe=None)
        else:
            pred = None
            for i in range(10):
                if not pred == None:
                    break
                elif i == 9:
                    #print(f'////////// Stopped at index {idx} //////////')
                    return featurized_df
                else:
                    pred = feature(row, pipe)
            
            if pred == True:
                pred = 1
            elif pred == False:
                pred = 0

        featurized_df.loc[idx, "prediction"] = pred
        # print(pred)
        # print(list(df.columns)[idx])
        
        if save_path:
            featurized_df.to_csv(save_path, index=False)
            
       # print('------------------------------------------------------------------\n')
    
    result_df = pd.DataFrame({
        "Attribute_name": df.columns,
        "prediction": featurized_df["prediction"].values[:len(df.columns)]
    })

    return result_df
    



# def get_feature_results(df, feature, pipe=None, stopped_index=0, save_path=None):
#     featurized_df = FeatureExtraction(df)
#     if save_path:
#         if os.path.isfile(save_path):
#             featurized_df = pd.read_csv(save_path, low_memory=False)
#     if feature == None:
#         return 'Please specify the feature you want to retrieve results from.'
#     for idx in range(stopped_index, len(df)):
#         row = df[idx:idx+1].reset_index(drop=True)
#         print(f"Index {idx}, Doing Feature '{row.loc[0, 'Attribute_name']}'\n")
#         if not pipe:
#             return 'Failed to make predictions! Model was not loaded / Pipeline was not set.'
#         pred = None
#         for i in range(10):
#             if not pred == None:
#                 break
#             elif i == 9:
#                 print(f'////////// Stopped at index {idx} //////////')
#                 return featurized_df
#             else:
#                 print(row,pipe)
#                 pred = feature(row)
#         else:
#             pred = feature(row)
#         if pred == True:
#             pred = 1
#         elif pred == False:
#             pred = 0

#         featurized_df.loc[idx, feature.__name__] = pred
        
#         if save_path:
#             featurized_df.to_csv(save_path, index=False)
            
#         print('------------------------------------------------------------------\n')
            
#     return featurized_df