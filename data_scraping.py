import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from multiprocess import Pool
from tqdm import tqdm
import datetime
import cProfile
import pstats
import logging

logging.basicConfig(
    filename='logs/scrapping_log.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Define the path list
pathlist = ['/nas-slot3/schwurbel1', '/nas-slot4/schwurbel2']


# Function to get file list
def get_filelist(pathlist):
    path_ = []
    channel_ = []

    for server in pathlist:
        if os.path.isdir(server):
            for channel in os.listdir(server):
                channel_path = os.path.join(server, channel)
                if os.path.isdir(channel_path):
                    channel_.append(channel_path)
                    try:
                        for file in os.listdir(channel_path):
                            file_path = os.path.join(channel_path, file)
                            if file[:7] == 'message':
                                path_.append(file_path)
                    except Exception:
                        continue

    return path_


# Function to find first and last messages
def find_first_message(content_file):
    lines = str(content_file).split("\n")
    mess_text = "message default clearfix"
    mess_id = "id="
    not_wanted = "message-"
    for line in lines:
        if mess_text in line and mess_id in line and not_wanted not in line:
            return int(re.search(r'message(\d+)', line).group(1))


def find_last_message(content_file):
    lines = str(content_file).split("\n")
    mess_text = "message default clearfix"
    mess_id = "id="
    not_wanted = "message-"
    for line in reversed(lines):
        if mess_text in line and mess_id in line and not_wanted not in line:
            return int(re.search(r'message(\d+)', line).group(1))


# Website detection function
def website_detection(ser):
    site = []
    for url in ser:
        tmp = np.nan
        if url == url:
            match = re.search(r'(?:https?:\/\/)?([\w.-]+)', str(url))
            if 'go_to' not in match.group(1) and 'message' not in match.group(1):
                tmp = match.group(1).replace('www.', '')
                site.append(tmp)
            else:
                site.append(tmp)
        else:
            site.append(tmp)
    return pd.Series(site)


# Replied-to detection
def replied_to(ser, group_name):
    message_id = []
    for rep in ser:
        tmp = np.nan
        try:
            match = re.search(r'#go_to_message(\d+)', rep)
            tmp = match.group(1)
            message_id.append(str(tmp) + str(hash(group_name)))
        except Exception:
            message_id.append(np.nan)
    return pd.Series(message_id)


# Link detection
def link_detection(ser):
    links = []
    for string in ser:
        pattern = r'<a href="([^"]+)">[^<]+</a>'
        match = re.search(pattern, str(string))
        if match:
            links.append(match.group(1))
        else:
            links.append(np.nan)
    return pd.Series(links)


# Storage filename helper
def storage_filename(filepath):
    filepath = filepath[22:]
    match = re.search(r'messages(\d+)\.html', filepath)
    num = match.group(1) if match else '1'
    filepath = filepath.split('/')[0]
    return f'{filepath}_{num}'


# Granulated function to process a file
def granulated(file):
    import pandas as pd
    import numpy as np
    from pathlib import Path

    import cProfile
    #import pstats
    import io

    import requests
    import re
    import os
    import sys

    from multiprocess import Pool
    import psutil
    from tqdm import tqdm

    from bs4 import BeautifulSoup


    pr = cProfile.Profile()
    pr.enable
    
    dst = "/data/schwurbelarchiv/extracted_information/unhashed_files" 
    
    
    with open(file, encoding='utf-8') as f:
        parsed = BeautifulSoup(f, "html.parser")
        content = (parsed, find_first_message(parsed), find_last_message(parsed))

        
    def get_author(content):
        #create dataframe containing all authors
        df_author = pd.DataFrame(columns=['uid', 'group_name', 'posting_date_author', 'author', 'fwd_author' ])

        pattern = r'title="(.*?)"'
                       

        group_name = content[0].find('div', class_= "text bold").string.strip()

        for i in range(content[1], content[2]):
            try:
                if "forwarded body" in str(content[0].find('div', id = "message%s"%i)):
                    df_author = pd.concat([df_author,
                                           pd.DataFrame([[str(i)+str(hash(group_name)),
                                                          group_name, 
                                                          re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                          content[0].find('div', id = "message%s"%i).find('div', class_="from_name").string.strip(),
                                                          re.search(r'(?<=from_name">\n).*(?= <span)', str(content[0].find('div', id = "message%s"%i).find('div', class_="forwarded body").find('div', class_="from_name")).strip()).group()]],
                                                        columns=['uid', 'group_name', 'posting_date_author', 'author', 'fwd_author'])])

                else: 
                    df_author = pd.concat([df_author, 
                                           pd.DataFrame([[str(i)+str(hash(group_name)),
                                                          group_name,
                                                          re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                          content[0].find('div', id = "message%s"%i).find('div', class_="from_name").string.strip()]],
                                                        columns=['uid', 'group_name', 'posting_date_author', 'author'])])

            except:
                continue

                
        return df_author
    
    
    def get_message(content):
        #create dataframe with all messages
        df_message = pd.DataFrame(columns=['uid', 'mid_message', 'posting_date', 'message', 'fwd_message', 'fwd_posting_date_message'])
        pattern = r'title="(.*?)"'
        
            
        group_name = str(hash(content[0].find('div', class_= "text bold").string.strip()))

        #pattern to detect text
        text_pattern = re.compile(r'<div class="text">(.*?)</div>', re.DOTALL)


        for i in range(content[1], content[2]):
            try:
                if "forwarded body" in str(content[0].find('div', id = "message%s"%i)):
                    text_content = re.findall(text_pattern, str(content[0].find('div', id = "message%s"%i)))[0]
                    df_message = pd.concat([df_message, pd.DataFrame([[str(i)+group_name, str(i), re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1), text_content.replace('<br/>', ' ').replace('\n', ''), re.search(r'<span class="details">(.*?)</span>', str(content[0].find('div', id = "message%s"%i).find('div', class_="forwarded body").find('div', class_="from_name")).strip()).group(1)]], columns=['uid', 'mid_message', 'posting_date', 'fwd_message', 'fwd_posting_date_message'])])
                else: 
                    text_content = re.findall(text_pattern, str(content[0].find('div', id = "message%s"%i)))[0]
                    df_message = pd.concat([df_message, pd.DataFrame([[str(i)+group_name, str(i), re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1), text_content.replace('<br/>', ' ').replace('\n', '')]], columns=['uid', 'mid_message', 'posting_date', 'message'])])
            except:
                continue
            
        return df_message
    
    def get_file(content):
        #create dataframe containing all media files
        df_file = pd.DataFrame(columns=['uid', 'mid_file', 'posting_date_file', 'link_url', 'media_file', 'media_file_type', 'fwd_link_url', 'fwd_media_file', 'fwd_media_file_type', 'fwd_posting_date_file'])

        pattern = r'title="(.*?)"'

        group_name = str(hash(content[0].find('div', class_= "text bold").string.strip()))

        for i in range(content[1], content[2]):
            try:
                if "forwarded body" in str(content[0].find('div', id = "message%s"%i)):
                    if "video" in str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix")):
                        df_file = pd.concat([df_file, pd.DataFrame([[str(i)+group_name,
                                                                     str(i),
                                                                     re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                                     re.search('href="(.*)"', str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix"))).group(1),
                                                                     'video',
                                                                     re.search(r'<span class="details">(.*?)</span>', str(content[0].find('div', id = "message%s"%i).find('div', class_="forwarded body").find('div', class_="from_name")).strip()).group(1)]],
                                                                   columns=['uid', 'mid_file', 'posting_date_file', 'fwd_media_file', 'fwd_media_file_type', 'fwd_posting_date_file'])])

                    elif "photo" in str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix")):
                        df_file = pd.concat([df_file, pd.DataFrame([[str(i)+group_name,
                                                                     str(i),
                                                                     re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                                     re.search('href="(.*)"', str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix"))).group(1),
                                                                     'photo',
                                                                     re.search(r'<span class="details">(.*?)</span>', str(content[0].find('div', id = "message%s"%i).find('div', class_="forwarded body").find('div', class_="from_name")).strip()).group(1)]],
                                                                   columns=['uid', 'mid_file', 'posting_date_file', 'fwd_media_file', 'fwd_media_file_type', 'fwd_posting_date_file'])])

                    elif "Voice message" in str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix")):
                        df_file = pd.concat([df_file, pd.DataFrame([[str(i)+group_name,
                                                                     str(i),
                                                                     re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                                     re.search('href="(.*)"', str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix"))).group(1),
                                                                     'voice message',
                                                                     re.search(r'<span class="details">(.*?)</span>', str(content[0].find('div', id = "message%s"%i).find('div', class_="forwarded body").find('div', class_="from_name")).strip()).group(1)]],
                                                                   columns=['uid', 'mid_file', 'posting_date_file', 'fwd_media_file', 'fwd_media_file_type', 'fwd_posting_date_file'])])

                    elif content[0].find('div', id = "message%s"%i).find('a', class_=False).get('href'):
                        df_file = pd.concat([df_file, pd.DataFrame([[str(i)+group_name,
                                                                     str(i),
                                                                     re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                                     content[0].find('div', id = "message%s"%i).find('a', class_=False).get('href')]],
                                                                   columns=['uid', 'mid_file', 'fwd_posting_date_file', 'fwd_link_url'])])

                    else: 
                        continue 

                else:
                    if "video" in str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix")):
                        df_file = pd.concat([df_file, pd.DataFrame([[str(i)+group_name,
                                                                     str(i),
                                                                     re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                                     re.search('href="(.*)"', str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix"))).group(1),
                                                                     'video']],
                                                                   columns=['uid', 'mid_file', 'posting_date_file', 'media_file', 'media_file_type'])])

                    elif "photo" in str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix")):
                        df_file = pd.concat([df_file, pd.DataFrame([[str(i)+group_name,
                                                                     str(i),
                                                                     re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                                     re.search('href="(.*)"', str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix"))).group(1),
                                                                     'photo']],
                                                                   columns=['uid', 'mid_file', 'posting_date_file', 'media_file', 'media_file_type'])])

                    elif "Voice message" in str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix")):
                        df_file = pd.concat([df_file, pd.DataFrame([[str(i)+group_name,
                                                                     str(i),
                                                                     re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                                     re.search('href="(.*)"', str(content[0].find('div', id = "message%s"%i).find('div', class_="media_wrap clearfix"))).group(1),
                                                                     'voice message']],
                                                                   columns=['uid', 'mid_file', 'posting_date_file', 'media_file', 'media_file_type'])])

                    elif content[0].find('div', id = "message%s"%i).find('a', class_=False).get('href'):
                        df_file = pd.concat([df_file, pd.DataFrame([[str(i)+group_name,
                                                                     str(i),
                                                                     re.search(pattern, str(content[0].find('div', id = "message%s"%i).find('div', class_="body"))).group(1),
                                                                     content[0].find('div', id = "message%s"%i).find('a', class_=False).get('href')]],
                                                                   columns=['uid', 'mid_file', 'posting_date_file', 'link_url'])])

                    else:
                        continue
            except:
                continue
            
        return df_file
                                                                         
                                                                            
                                                                        
                                                                        
    def get_dataset(content):
        #joining step by step with different kind of joins to ensure data reliability 
        try:
            df_content = pd.merge(get_message(content), get_file(content), on = 'uid', how = 'outer')
            df_merge = pd.merge(df_content, get_author(content), on = 'uid', how = 'outer')
            df_merge['group_name'], df_merge['author'] = df_merge['group_name'].ffill(), df_merge['author'].ffill()
            df_merge['posting_date'].fillna(df_merge['posting_date_file'], inplace = True) 
            #df_merge['posting_date'] = pd.to_datetime(df_merge['posting_date'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%d-%m %H:%M:%S')
            df_merge['posting_date'] = pd.to_datetime(df_merge['posting_date'], dayfirst = True, utc=True)
            df_merge['posting_date_file'] = pd.to_datetime(df_merge['posting_date_file'], dayfirst = True, utc=True)
            df_merge['posting_date_author'] = pd.to_datetime(df_merge['posting_date_author'], dayfirst = True, utc=True)
            
            df_merge["day"] = df_merge["posting_date"].dt.floor('D') + pd.Timedelta(12, unit='h')
            df_merge["week"] = df_merge["posting_date"].dt.isocalendar().week
            df_merge["weekday"] = df_merge["posting_date"].dt.isocalendar().day
            df_merge['message_hash'] = df_merge['message'][df_merge['message'].notna()].apply(hash).apply(str)
            df_merge['fwd_message_hash'] = df_merge['fwd_message'][df_merge['fwd_message'].notna()].apply(hash).apply(str)
            df_merge['website'] = website_detection(df_merge['link_url']).combine_first(website_detection(df_merge['fwd_link_url']))
            df_merge['message_id'] = df_merge['mid_message'].combine_first(df_merge['mid_file'])
            df_merge['replied_to'] = replied_to(df_merge['link_url'], df_merge['group_name'][1])

            #df_merge['transcribed_message'] = np.nan
            
            return df_merge

        except:
            return pd.DataFrame()

    

    


    get_dataset(content).to_csv(
		Path(dst, storage_filename(file) + ".csv.gzip"),
		 index=False,
		  compression="gzip"
	)

    #pr.disable()
    #s = io.StringIO()
    #pr.print_stats(stream=s)

    # Save the profiling results to a text file
    #with open(f'profile_granulated_{os.getpid()}.txt', 'w') as f:
    #    f.write(s.getvalue())

def safe_granulated(file):
    try:
        granulated(file)

    except PermissionError as e: 
        logging.error(f"PermissionError for file {file}: {e}")
        
    except Exception as e:
        logging.error(f"ProcessingError for file {file}: {e}")


# Function to process files in parallel
def pool_parse_rescue(number_of_cores, files):
    pool = Pool(number_of_cores)
    for _ in tqdm(pool.imap_unordered(safe_granulated, files), total=len(files)):
        pass
    pool.close()


# Main Execution
if __name__ == "__main__":
    n_cores = int(input("Enter the number of cores to use for parallel processing: "))
    filelist = get_filelist(pathlist)
    print(len(filelist))


    #pool_parse_rescue(n_cores, filelist)
    pool_parse_rescue(n_cores, filelist[18000:40000])

