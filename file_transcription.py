import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import random

import requests
import re
import os
import sys

import pickle
import ffmpeg
import whisper
import logging


from multiprocess import Pool

from tqdm import tqdm

#implment logger
logging.basicConfig(level=logging.INFO, filename='logs/transcription.log', format='%(asctime)s - %(levelname)s - %(message)s')

#not storing the full dataset, as I only need certain columns of it
SHAPED_DATA = None
STOP_REQUESTED = False


#interrupt program by adding the term <stop transcription> into the command.txt file
def check_command_file(filepath):
    
    global STOP_REQUESTED
    
    try:
        with open(f"command/{filepath}", 'r') as file:
            content = file.read().strip()
            if content == "stop transcription":
                STOP_REQUESTED = True
                
    except FileNotFoundError:
        STOP_REQUESTED = False
    
        


#load datatypes of dataset to avoid dtype warning message
def get_dtypes():

    with open('dictionaries/dtypes_full_dataset.pkl', 'rb') as f:
        read_dtypes = pickle.load(f)
        
    return read_dtypes
    
    
    
    
#load groupname dictionary
def groupname_dictionary():
    with open('dictionaries/group_name_dictionary.pkl', 'rb') as f:
        res_dict = pickle.load(f)
    return res_dict




#function to get filetypes 
def get_filetypes(df):
    
    file_types = df['media_file_type'].combine_first(df['fwd_media_file_type']).unique()
    file_types = file_types[~pd.isnull(file_types)]
    #delete photo, as we do not use it to derive any information yet
    file_types = np.delete(file_types, np.where(file_types == 'photo'))
    
    return file_types




#server location where files are stored
def get_pathlist():
    return ['/nas-slot3/schwurbel1', '/nas-slot4/schwurbel2']




#get full filepath for every group from the deepest storage position
def file_server_dictionary(pathlist):
    server_dict = {}
    for path in pathlist:
        for file in os.listdir(path):
            server_dict.update({file: f"{path}/{file}"})
            
    return server_dict




def load_whisper_model():
    
    transcription_model = whisper.load_model("small")
    logging.info("Whisper model loaded successfully")
    
    return transcription_model




def get_file_tuple(groupname):
    
    global SHAPED_DATA
    
    SHAPED_DATA['fwd_plus_orig_media_files'] = SHAPED_DATA['media_file'].combine_first(SHAPED_DATA['fwd_media_file'])

    
    data = SHAPED_DATA
    
    ftypes = get_filetypes(data)
    name_dict = groupname_dictionary()
    server_dict = file_server_dictionary(get_pathlist())
    
    link_tuple_list = []
    
    group_df = data[data["group_name"] == groupname]
    media_df = group_df[(group_df["media_file_type"] == 'voice message') | (group_df["fwd_media_file_type"] == "voice message")]
    files = list(media_df["fwd_plus_orig_media_files"])
    
    logging.info(f"Groupname? {groupname}")
    
    try:
        for groupcode in name_dict[groupname]:
            for file in files:
                #logging.info(f"Path? {server_dict[name_dict[groupname]]}")
                link = f"{server_dict[groupcode]}/{file}"
                if os.path.isfile(link):

                    #logging.info(f"Link? {link}")
                    link_tuple_list.append((link, groupname, file))
                else:
                    continue
    except:
        logging.info(f"This group does not appear in the name_dict: {groupname} searchcode: faultygroupname")
        pass
        
    unique_link_tuple_list = list(pd.Series(link_tuple_list).unique())

    return unique_link_tuple_list




def pool_filepaths(n_cores, groupname_list):
    
    link_tuple_list = []
    
    iter_list = groupname_list
    
    
    pool = Pool(n_cores)
    
    for result in tqdm(
        pool.imap_unordered(func=get_file_tuple, iterable=iter_list),
        total=len(iter_list)
        ):
            link_tuple_list.append(result)
    pool.close()
    

    
    
    return link_tuple_list



##################duration###########
def get_file_duration(filepath_tuple):
    
    i = 0
    
    filepath = filepath_tuple[0]
    try:
        duration = float(ffmpeg.probe(filepath)["streams"][0]['duration'])
        i = i+1
        logging.info(f"{i} {(duration, filepath_tuple[0], filepath_tuple[1], filepath_tuple[2])}")
    except:
        duration = float("NaN")
    
    return (duration, filepath_tuple[0], filepath_tuple[1], filepath_tuple[2]) 




def pool_file_duration(n_cores, file_tuple_list):
    
    file_metadata = []
    
    pool = Pool(n_cores)
    
    for result in tqdm(
        pool.imap_unordered(func=get_file_duration, iterable=file_tuple_list),
        total=len(file_tuple_list)
        ):
            file_metadata.append(result)
    pool.close()
    
    #return duration_df
    return file_metadata

##################duration#############


def create_path_tuples():
    server_dict = file_server_dictionary(get_pathlist())
    name_dict = groupname_dictionary()
    
    groupname_list = SHAPED_DATA['group_name'].unique().tolist()
    
    filtered_list = [i for i in groupname_list if i == i]

    tmp = pool_filepaths(80, filtered_list)

    path_tuples = [ele for sub in tmp for ele in sub]
    
    logging.info(f"List length? {len(path_tuples)}")

    
    return path_tuples
    
    
    

def create_transcription_df(path_tuples):
    
    
    global SHAPED_DATA
    
    path_tuples = path_tuples
     
    
    
        
    SHAPED_DATA['fwd_plus_orig_media_files'] = SHAPED_DATA['media_file'].combine_first(SHAPED_DATA['fwd_media_file'])

    
    #file_duration_tuples = pool_file_duration(120, path_tuples)
    #duration_df = pd.DataFrame(data = file_duration_tuples, columns = ['duration', 'filepath', 'group_name', 'filename'])
    filepath_df = pd.DataFrame(data = path_tuples, columns = ['filepath', 'group_name', 'filename'])
    
    logging.info(f"Filepath dataframe size {filepath_df.shape}")

    
    joined_filepath_df = pd.merge(SHAPED_DATA[['UID_key', 'fwd_plus_orig_media_files', 'media_file', 'fwd_media_file', 'group_name', 'transcribed_message']],
                            filepath_df,
                            how = 'left',
                            left_on = ['fwd_plus_orig_media_files', 'group_name'],
                            right_on = ['filename', 'group_name']
                           )
    logging.info(f"Dataframe size {joined_filepath_df[joined_filepath_df.filepath.notna()].shape}")
    
       
    
    #dur_sample = duration_df.query(
    #    'duration > 30 and duration < 80').groupby(
    #    ["group_name"]).sample(n = 1000)
    
    
    return joined_filepath_df#[['UID_key', 'duration', 'filepath', 'group_name', 'filename']], #dur_sample



# when running for the first time where filename and filepath are not given yet, have (dataframe_containing_filepath) as parameter
# else, no parameter is needed and only SHAPED_DATA is requested

def transcription(dataframe_containing_filepath):
    
    global STOP_REQUESTED
    global SHAPED_DATA
    
    #commented out because the current data does already have all necessary columns to be run in the transcription code and does not require more preparation
    #data = dataframe_containing_filepath.copy()
    #data["transcribed_message"] = "NaN"
    
    data = SHAPED_DATA.copy()
    
    model = load_whisper_model()
    
    for count, (index, row) in enumerate(data.iterrows()):
        try:
            link = row['filepath']
            #transcribed = transcribe_audio_api(link)

            logging.info(f"Trying to load {link}")

            start = datetime.datetime.now()
            transcribed = model.transcribe(link, fp16=False, language='German')
            end = datetime.datetime.now()
            logging.info(f"File with path {link} was successfully transcribed. Transcription duration: {end-start}.")

            #logging.info(f"transcription_result:{transcribed['text']}.")
            #transcribed = '123456789'
            data.loc[index, 'transcribed_message'] = transcribed['text']
            #data.loc[index, 'transcribed_message'] = transcribed
            #logging.info(f"Test if entry in df works{data.loc[index, 'transcribed_message'] }.")

            n_transcribed = data[data['transcribed_message'] != 'NaN'].shape[0]
            #logging.info(f"File with path {link} was successfully transcribed. {n_transcribed} of {data.shape[0]} files transcribed. Transcription duration: {end-start}. Dataframe row index: {index}")
            logging.info(f"dataframe size: {n_transcribed}")

            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y_%m_%d_%H")

            if (n_transcribed > 0) & (n_transcribed % 300 == 0):
                #logging.info(f"in the right if")
                data.to_csv(f'transcripted/transcribed_messages_{formatted_time}.csv.gzip', compression = 'gzip')
                logging.info(f"Storage milestone")

            elif STOP_REQUESTED == True:
                data.to_csv(f'transcripted/transcribed_messages_{formatted_time}.csv.gzip', compression = 'gzip')
                logging.info(f"Transcription interrupted")
                break

            else:
                check_command_file("command.txt")
                continue

        except:
            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y_%m_%d_%H")
            logging.info(f"I broke")


            if STOP_REQUESTED:
                transcribed_df.to_csv(f'transcripted/transcribed_messages_{formatted_time}.csv.gzip', compression = 'gzip')
                logging.info(f"Transcription interrupted")
                break

            else:
                check_command_file("command.txt")
                continue


    logging.info(f"Transcription ended")

    #return data




def storing_transcripted_data():
    
    global SHAPED_DATA
    
    #transcribed_df = transcription(create_transcription_df(create_path_tuples()))
    #logging.info(f"Finished dataset has {len(transcribed_df)} rows")
    #transcribed_df = create_transcription_df(create_path_tuples())
    #transcribed_df_tm = pd.merge(transcribed_df,
    #                         SHAPED_DATA[['UID_key', 'transcribed_message']],
    #                         on='UID_key',
    #                         how='left')
    
    #transcribed_df.to_csv('data/voice_messages_to_transcribe.csv.gzip', compression = 'gzip')
    #logging.info(f"All data was successfully stored")

    
          
def get_latest_transcribed_df():
    directory = 'transcripted'

    # Initialize variables to store the details of the most recently modified file
    latest_modified = 0
    latest_file = None

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Get the full file path
        filepath = os.path.join(directory, filename)
        # Check if it's a file
        if os.path.isfile(filepath):
            # Get the last modification time and compare it to the last stored time
            
            file_modified = os.path.getmtime(filepath)
            if file_modified > latest_modified:
                latest_modified = file_modified
                latest_file = filename
                #print(latest_file)
                
    return f'{directory}/{latest_file}'
        
          
def load_data(path):
          
    global SHAPED_DATA

    full_data = pd.read_csv(path, compression = 'gzip', usecols = ['UID_key', 'group_name', 'media_file_type', 'fwd_media_file_type', 'media_file', 'fwd_media_file'])
    #, nrows = 10000000, usecols = ['UID_key', 'group_name', 'media_file_type', 'fwd_media_file_type', 'media_file', 'fwd_media_file'])
    transcribed = pd.read_csv(get_latest_transcribed_df(), compression = 'gzip', usecols = ['UID_key', 'group_name', 'media_file', 'fwd_media_file','transcribed_message'])
    
    data = pd.merge(full_data,
                    transcribed[['UID_key', 'transcribed_message']],
                    on='UID_key',
                    how='left')
    
    
    logging.info("Data loaded successfully")
    #rel_nam_data = data[(data['media_file'].notna()) & (data['media_file_type'] == 'voice message')]
    
    
    SHAPED_DATA = data #rel_nam_data
    
          
        
        
        
        
        
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
          print('Usage: python file_transcription.py <full_path_to_file>')
    else:
        filepath = sys.argv[1]
        load_data(filepath)
        
        #not needed when 
        transcription(create_transcription_df(create_path_tuples()))
        transcription()
        #create_transcription_df(create_path_tuples())
        
        