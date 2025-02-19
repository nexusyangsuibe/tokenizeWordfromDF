# 分词筛选模块

from functools import partial
from hashlib import sha256
from io import StringIO
import multiprocessing as mp
import pickle
import os
import re

import pandas as pd
import numpy as np
import thulac

# common tool functions are as follows
def outputAsXlsx(df,output_filename,output_pathname,thereshold_rows=1000000,thereshold_GB=4):
    # output the dataframe as xlsx file with divsion within the thereshold_rows and thereshold_GB
    # 搜寻正确的分块数
    def findBestBulkNum(df,thereshold_GB,best_bulk_num=1):
        for idx in range(best_bulk_num):
            memory_usage_GB=df.iloc[int(len(df)*(idx/best_bulk_num)):int(len(df)*((idx+1)/best_bulk_num))].memory_usage(deep=True).sum()/(1024**3)
            if memory_usage_GB>thereshold_GB:
                new_bulk_num=max(int(df.memory_usage(deep=True).sum()/(1024**3))//thereshold_GB+1,best_bulk_num+1)
                return findBestBulkNum(df,thereshold_GB,best_bulk_num=new_bulk_num)
        else:
            return best_bulk_num
    # 按分块数输出
    def outputAccording2BestBulkNum(df_bulk,fileName,thereshold_GB):
        bulk_num=findBestBulkNum(df_bulk,thereshold_GB)
        if bulk_num==1:
            df_bulk.to_excel(fileName)
        else:
            print(f"文件{fileName}所需的存储空间超过阙值{thereshold_GB}GB，再分为{bulk_num}个文件输出")
            for iidx in range(bulk_num):
                fileName_=f"{''.join(fileName.split('.')[:-1])}_{iidx+1}.xlsx"
                print(f"正在写入{fileName_}")
                df_bulk.iloc[int(file_rows*(iidx/bulk_num)):int(file_rows*((iidx+1)/bulk_num))].to_excel(fileName_)
        return None
    # 先按照行数阙值分为file_num+1个文件输出，对每个输出文件检查存储空间大小并根据最优文件数输出
    file_num=int(df.shape[0]//thereshold_rows)
    print(f"共{df.shape[0]}行，文件名为{output_filename}，分为{file_num+1}个文件输出")
    if file_num==0:
        outputAccording2BestBulkNum(df,fileName=f"{output_pathname}{'' if output_pathname.endswith('/') else '/'}{''.join(output_filename.split('.')[:-1])}.xlsx",thereshold_GB=thereshold_GB)
    else:
        file_rows,last_rows=divmod(df.shape[0],file_num+1)
        last_rows=file_rows+last_rows
        print(f"前{file_num}个文件{file_rows}行，最后1个文件{last_rows}行")
        for idx in range(file_num):
            df_bulk=df.iloc[idx*file_rows:(idx+1)*file_rows]
            fileName=f"{output_pathname}{'' if output_pathname.endswith('/') else '/'}{''.join(output_filename.split('.')[:-1])}_{idx+1}.xlsx"
            print(f"正在写入{fileName}")
            outputAccording2BestBulkNum(df_bulk,fileName,thereshold_GB)
        if last_rows:
            df_bulk=df.iloc[file_num*file_rows:]
            fileName=f"{output_pathname}{'' if output_pathname.endswith('/') else '/'}{''.join(output_filename.split('.')[:-1])}_{file_num+1}.xlsx"
            print(f"正在写入{fileName}")
            outputAccording2BestBulkNum(df_bulk,fileName,thereshold_GB)
    return None

def saveConcatedDataAsFinalResult(runtime_code,concatedDF,output_filename,clear_respawnpoint_upon_conplete):
    # the end process of the concatDF, including writing the final result to the disk and clear the respawnpoint folder
    if not clear_respawnpoint_upon_conplete or not output_filename:
        pickle.dump(concatedDF,open(f"respawnpoint/{runtime_code}_word_tokenized.pkl","wb"))
    if output_filename:
        print("开始将最终结果写入硬盘")
        if output_filename.endswith(".pkl"):
            pickle.dump(concatedDF,open(f"finalresults/{output_filename}","wb"))
        elif output_filename.endswith(".xlsx"):
            outputAsXlsx(concatedDF,output_filename,"finalresults")
        elif output_filename.endswith(".csv"):
            concatedDF.to_csv(f"finalresults/{output_filename}")
        else:
            raise ValueError(f"不支持的文件格式{output_filename}，请核查")
    if clear_respawnpoint_upon_conplete:
        if not output_filename and input("由于未指定output_filename，finalresults文件夹中不会产生任何结果文件，又clear_respawnpoint_upon_conplete参数为True，将清空respawnpoint文件夹中的所有临时文件，因此您的本次运行不会产生任何可观测的结果，输入y继续，输入其他任意字符取消清空respawnpoint文件夹：").lower()!="y":
            print("用户取消清空respawnpoint文件夹")
            return False # return False to indicate that nothing emerge in either finalresults or respawnpoint
        print("开始清空respawnpoint文件夹")
        for file in os.listdir("respawnpoint/"):
            os.remove("respawnpoint/" + file)
    return None

# work horses are as followed
def doTokenize(tokenizer,user_added_critic_words,user_added_other_words,user_added_stop_words,only_retain_meaningful_words,omit_content_in_parentheses,delete_single_character,minor_retain_words_thereshold,other_preprocessing_injection,single_text_mit_idx):
    # tokenize the text and return the needed statistics nach the option only_retain_meaningful_words
    idx,raw_text=single_text_mit_idx
    # preprocessing
    text=raw_text.lower() # In most cases lowerization is a good idea, except when processing some language in which capizalization has special meanings, e.g. In Deutsch Capitalization represents noun
    text=re.sub(r"(\d+)\.?(\d+)%?","24",text) # thulac is very bad at recognizing numbers, especially with punkt oder %
    text=re.sub(r"丨|—","，",text) # replace the Chinese character 丨 and — with comma else the tokenizer will recognize them wrongly
    text=re.sub(r"\u3000","",text) # replace \u3000 with empty string, in most cases it will be detected and deleted automatically by thulac, but it is not always the case
    if omit_content_in_parentheses:
        len_text_before_delete_content_in_parentheses=len(text)
        text=text.replace("（","(") # some authors are so stupid and irresponsible that they use Chinese parenthesis to match English parenthesis or wise versa, it is time-consuming to detect these faults, so I just change all the parentheses to the English style
        text=text.replace("）",")")
        forward_lookup_text=re.sub(r"\(.*?\)|（.*?）|\[.*?\]|【.*?】","",text)
        if len(forward_lookup_text)/len_text_before_delete_content_in_parentheses<0.24: # it is a commen sense that content in the parentheses should not account for a major part of the paragraph, if it does so, there may be a mistake in matching
            backward_lookup_text=text[::-1] # reverse the string because so authors are so fool that the write a left parenthesis without a right parenthesis matching it. if the regexp do not match from right to left, a left parenthesis that occur early in the paragraph may match a right parenthesis at the end, thus the whole paragraph is deleted as content within the parentheses
            backward_lookup_text=re.sub(r"\).*?\(|）.*?（|\].*?\[|】.*?【","",backward_lookup_text) # if needed, delete all the content within parentheses, the original form if reverse is not applied at the former step is text=re.sub(r"\(.*?\)|（.*?）|\[.*?\]|【.*?】","",text)
            backward_lookup_text=backward_lookup_text[::-1]
            if len(backward_lookup_text)==len(forward_lookup_text): # it is faster than backward_loopup_text==foreard_lookup_text
                text=forward_lookup_text
            elif len(backward_lookup_text)/len_text_before_delete_content_in_parentheses<0.24:
                print(f"索引{idx}对应的文本信息'{raw_text[:max(24,len(raw_text))]}'疑似有未能正确闭合的括号，该文本不删去括号内内容")
            else:
                text=backward_lookup_text
        else:
            text=forward_lookup_text
    if other_preprocessing_injection:
        if type(other_preprocessing_injection)!=function:
            raise TypeError("'other_preprocessing_injection'参数只能传入单变量的function")
        text=other_preprocessing_injection(text)
    text_tuple=text.split("。") # thulac seems to apply memory with a fixed size, so that when the paragraph is too long, it cannot handle it, so we split it into sentences roughly
    text_tuple=tuple(re.sub(r"(\s)+"," ",text).strip() for text in text_tuple) # replace continual vacuum to a single one and remove the vacuum at the start or end point of each sentence
    text_tuple=tuple(text for text in text_tuple if text) # delete all empty sentences
    if len(text_tuple)==0:
        print(f"索引{idx}对应的文本信息'{raw_text}'经过预处理后为空，删去该条记录")
        return None
    text_tuple=tuple((*(text_tuple[idx]+"。" for idx in range(len(text_tuple)-1)),text_tuple[-1])) # the punctuation per se is important for grammtic analysis, so we add it back
    tokenized_text_with_pos=[]
    for text in text_tuple:
        tokenized_text_with_pos.extend(tokenizer.cut(text))
    if only_retain_meaningful_words:
        tokenized_text=[tmp[0] for tmp in tokenized_text_with_pos if tmp[0] not in user_added_stop_words] # drop the POS and user added stop words since they are not needed for lexical analysis
        if delete_single_character:
            tokenized_text=[tmp for tmp in tokenized_text if len(tmp)>1 or tmp in user_added_critic_words or tmp in user_added_other_words] # if needed delete the single characters
        all_words_count=len(tokenized_text)
        if minor_retain_words_thereshold and all_words_count<minor_retain_words_thereshold:
            print(f"在索引位置{idx}的文本'{raw_text[:max(24,len(raw_text))]}'，实行分词筛选后得到的token为{tokenized_text}，其未达到要求的最低可接受文本长度{minor_retain_words_thereshold}，因此被删去")
            return None
        critic_words_count=len(tuple(tmp for tmp in tokenized_text if tmp in user_added_critic_words))
        critic_words_frequency=(critic_words_count/all_words_count) if all_words_count else 0
        return (idx," ".join(tokenized_text),critic_words_count,all_words_count,critic_words_frequency)
    else:
        return (idx,tokenized_text_with_pos)

# main function
def tokenizeWordfromDF(runtime_code,input_file,tokenize_column_name,tokenized_column_name,user_added_stop_words_filename,user_added_critic_words_filename,user_added_other_words_filename,size_per_respawn_chunk_GB,only_retain_meaningful_words,omit_content_in_parentheses,delete_single_character,minor_retain_words_thereshold,other_preprocessing_injection,output_filename,clear_respawnpoint_before_run,clear_respawnpoint_upon_conplete):
    print("分词模块开始运行")
    # create the folders if not exists
    if "respawnpoint" not in os.listdir():
        print(f"在工作目录{os.getcwd()}下未找到用于存储临时文件的respawnpoint文件夹，将自动创建")
        os.mkdir("respawnpoint")
    if "finalresults" not in os.listdir():
        print(f"在工作目录{os.getcwd()}下未找到用于存储最终结果的finalresults文件夹，将自动创建")
        os.mkdir("finalresults")
    # clear the respawnpoint folder before running if needed
    if clear_respawnpoint_before_run:
        for file in os.listdir("respawnpoint/"):
            os.remove("respawnpoint/"+file)
    # read and check the input file
    if type(input_file)==str and input_file.endswith(".pkl"):
        input_file=pickle.load(open(input_file,"rb"))
    if type(input_file)!=pd.DataFrame:
        raise ValueError("输入文件必须是DataFrame类型或包含DataFrame的pickle文件路径名")
    if input_file.shape[0]==0:
        raise ValueError("输入的DataFrame为空")
    tmp_buf=StringIO()
    input_file.info(buf=tmp_buf)
    df_identity_code=sha256(tmp_buf.getvalue().encode(encoding="utf-8")).hexdigest()[:24] # used to check the whether the file in the respawnpoint is the results of handling the same DataFrame
    index_name=input_file.index.name # save the index name to restore it after reset the index
    if not index_name:
        while True: # if the existing index does not have a name, we need to give it one name else we cannnot recover it
            new_index_name=f"neue_idx_name_{np.random.randint(10000,100000)}"
            if new_index_name not in input_file.columns:
                break
        input_file.index.name=new_index_name
        index_name=new_index_name
    input_file=input_file.reset_index() # reset the index to default increasing primary key to ensure that the index is unique
    pickle.dump(input_file,open(f"respawnpoint/{runtime_code}_input_dataframe_backup.pkl","wb")) # save the input file to the respawnpoint folder so that the input Dataframe can be released from the memory, thus columns that do not need to be tokenized will not occupy the precious memory
    if type(tokenize_column_name)!=str or tokenize_column_name not in input_file.columns:
        raise ValueError(f"输入的tokenize_column必须是字符串类型且存在于输入文件的列名中，当前指定的{tokenize_column_name=}，而输入文件的列名为{input_file.columns=}")
    input_file=input_file[tokenize_column_name] # only keep the column to be tokenized to save memory
    len_input_file=len(input_file)
    input_file=None # manually collect the garbage to save memory
    # read and check the user-defined word dictionary
    user_added_stop_words=tuple()
    user_added_critic_words=tuple()
    user_added_other_words=tuple()
    if user_added_stop_words_filename:
        user_added_stop_words=tuple(set(word.lower() for word in open(user_added_stop_words_filename,"r",encoding="utf-8").read().split("\n") if word)) # for words that should be removed but do not remove automatically by thulac
    if user_added_critic_words_filename:
        user_added_critic_words=tuple(set(word.lower() for word in open(user_added_critic_words_filename,"r",encoding="utf-8").read().split("\n") if word)) # for words that are indended to calculate the frequency of appearance
    if user_added_other_words_filename:
        user_added_other_words=tuple(set(word.lower() for word in open(user_added_other_words_filename,"r",encoding="utf-8").read().split("\n") if word)) # for words that should be detected automatically by thulac but do not
    user_dict=tuple({*user_added_stop_words,*user_added_critic_words,*user_added_other_words})
    with open(f"respawnpoint/{runtime_code}_user_dict_for_tokenizer.txt","w",encoding="utf-8") as f:
        f.write("\n".join(user_dict)) # the thulac package requires the user-defined dictionary to be written into a txt file so we follow the rule
    # parameter check
    if minor_retain_words_thereshold:
        if (type(minor_retain_words_thereshold) not in [int,float]) or minor_retain_words_thereshold<1:
            raise TypeError(f"参数'minor_retain_words_thereshold'应该是正整数，您输入的{minor_retain_words_thereshold}不符合要求")
    if not only_retain_meaningful_words:
        if user_added_stop_words:
            print("在'only_retain_meaningful_words=False'隐含的语法分析模式下'user_added_stop_words'参数不生效")
        if minor_retain_words_thereshold:
            print("在'only_retain_meaningful_words=False'隐含的语法分析模式下'minor_retain_words_thereshold'参数不生效")
        if delete_single_character:
            print("在'only_retain_meaningful_words=False'隐含的语法分析模式下'delete_single_character'参数不生效")
    # load and initialize the thulac tokenizer
    thul=thulac.thulac(user_dict=f"respawnpoint/{runtime_code}_user_dict_for_tokenizer.txt",filt=only_retain_meaningful_words) # the parameter filt should be defined as True when we do lexcial analysis, and False when we do grammatical analysis in which POS tags are needed
    doTokenizeOhneNewsContentTuple=partial(doTokenize,thul,user_added_critic_words,user_added_other_words,user_added_stop_words,only_retain_meaningful_words,omit_content_in_parentheses,delete_single_character,minor_retain_words_thereshold,other_preprocessing_injection)
    # check the respawnpoint to find the finished and unfinished interval
    finished_intervals=[]
    result_filenames=[]
    for raw_filename in os.listdir("respawnpoint"):
        if ck_point:=re.match(f"{runtime_code}_wt_(\\d+)_(\\d+)_{df_identity_code}_{tokenize_column_name}_{'meaningful' if only_retain_meaningful_words else 'all'}.pkl",raw_filename):
            result_filenames.append("respawnpoint/"+raw_filename)
            finished_intervals.append([int(ck_point.group(1)),int(ck_point.group(2))]) # note all the intervals are defined as closed in the front and open in the end
    if finished_intervals:
        for batch_interval in finished_intervals:
            if batch_interval[0]>=batch_interval[1]:
                raise ValueError(f"在respawnpoint文件夹中找到了不合理命名的文件{runtime_code}_wt_{batch_interval[0]}_{batch_interval[1]}_{df_identity_code}_{tokenize_column_name}.pkl，其所显示的已完成区间{batch_interval}是一个空集或不是一个在实数空间可以理解的区间")
        while True: # find the union of the intervals in the finished_intervals
            if len(finished_intervals)==1:
                break
            outer_break_flag=0 # the flag to break outer for-loop
            for idx_self in range(len(finished_intervals)-1):
                if outer_break_flag==1:
                    break
                self_interval=finished_intervals[idx_self]
                for idx_other in range(idx_self+1,len(finished_intervals)):
                    other_interval=finished_intervals[idx_other]
                    if (self_interval[1]>other_interval[0]) or (other_interval[1]>self_interval[0]):
                        raise ValueError(f"检测到已完成处理的区间中存在重叠关系的区间{self_interval}与{other_interval}，这会导致在最终合并操作时对于采用哪一份运行的结果产生歧义，请检查您的respawnpoint文件夹")
                    elif self_interval[1]==other_interval[0] or self_interval[0]==other_interval[1]:
                        finished_intervals.remove(self_interval)
                        finished_intervals.remove(other_interval)
                        finished_intervals.append([min(self_interval[0],self_interval[1],other_interval[0],other_interval[1]),max(self_interval[0],self_interval[1],other_interval[0],other_interval[1])])
                        outer_break_flag=1
                        break
            else:
                break
        finished_intervals.sort() # sort nach the starting point of the interval
        unfinished_intervals=[]
        if (finished_intervals_begin:=finished_intervals[0][0])>0:
            unfinished_intervals.append([0,finished_intervals_begin])
        if (finished_intervals_end:=finished_intervals[-1][1])<len_input_file:
            unfinished_intervals.append([finished_intervals_end,len_input_file])
        if len(finished_intervals)>1:
            for idx in len(range(finished_intervals)-1):
                unfinished_intervals.append([finished_intervals[idx][1],finished_intervals[idx+1][0]])
    else:
        unfinished_intervals=[[0,len_input_file]]
    print(f"通过检查respawnpoint文件夹中的备份文件的文件名，我们推断出已完成的区间{finished_intervals=}，待完成的区间{unfinished_intervals=}")
    for unfinished_interval in unfinished_intervals: # In the first edition of the program I just concat all the intervals together, but owing to the potential incontinuity, it is hard to convey it as a Interval in the filename, so in this edition I modify to use a for-loop to tackle each continuum respectively
        input_file_interval=pickle.load(open(f"respawnpoint/{runtime_code}_input_dataframe_backup.pkl","rb"))[tokenize_column_name].iloc[unfinished_interval[0]:unfinished_interval[1]]
        mem_usage=input_file_interval.memory_usage(deep=True) # since pandas give us a method to get the exact memory usage of the dataframe, we use it to estimate the memory usage of the whole process rather than use pympler.asizeof to calculate the dict's memory usage
        input_file_interval=tuple(input_file_interval.to_dict().items()) # convert to a format esay for multiprocessing
        len_input_file_interval=len(input_file_interval)
        # if the memory usage is too large, we need to split the task into several batches and save the result of each batch upon conpleted, so that the user can continue the process without losing the previous results in such a time-consuming task, of-course memory saving is also a reason
        mem_usage_GB=round(mem_usage/(1024**3),6)
        partitions=int(mem_usage_GB/size_per_respawn_chunk_GB)+1 # define how many respawn chunks are needed, add 1 to avoid devide by zero error
        size_per_respawn_chunk_GB,last_part_add_size=divmod(len_input_file_interval,partitions) # define how many rows should be processed in each respawn chunk
        batch_intervals=[[idx*size_per_respawn_chunk_GB,(idx+1)*size_per_respawn_chunk_GB] for idx in range(partitions-1)]
        batch_intervals.append([size_per_respawn_chunk_GB*(partitions-1),size_per_respawn_chunk_GB*partitions+last_part_add_size])
        print(f"在区间{unfinished_interval}中共有{len_input_file_interval}行需要分词处理，内存占用{mem_usage_GB}GB，共分为{partitions}组运行")
        # save the tuples to be processed to the disk and explicitly collect the garbage to save memory
        raw_filenames=[]
        for batch_interval in batch_intervals:
            raw_filename=f"respawnpoint/{runtime_code}_word_tokenize_raw_tuple_{batch_interval[0]}_{batch_interval[1]}.pkl"
            pickle.dump(input_file_interval[batch_interval[0]:batch_interval[1]],open(raw_filename,"wb"))
            raw_filenames.append(raw_filename)
        input_file_interval=None # garbage collection to save memory
        # do the tokenize and generate statistics (if needed) for each respawn chunk
        for raw_filename in raw_filenames:
            # tokenize
            texts_mit_idx=pickle.load(open(raw_filename,"rb"))
            with mp.Pool() as pool:
                results=pool.map(doTokenizeOhneNewsContentTuple,texts_mit_idx)
            # results=map(doTokenizeOhneNewsContentTuple,texts_mit_idx) # test code for debug since debug is difficult in multiprocessing
            results=tuple(result for result in results if result)
            if not results:
                print(f"经过分词筛选，在{texts_mit_idx[0][0]}至{texts_mit_idx[-1][0]}区间内的所有文本都没有达到可以返回的要求，这并不是一个常见的情况，请检查您的参数设置")
            # save and return
            runtime_interval=re.match(f"respawnpoint/{runtime_code}_word_tokenize_raw_tuple_(\\d+)_(\\d+).pkl",raw_filename)
            store_path=f"respawnpoint/{runtime_code}_wt_{runtime_interval.group(1)}_{runtime_interval.group(2)}_{df_identity_code}_{tokenize_column_name}_{'meaningful' if only_retain_meaningful_words else 'all'}.pkl"
            pickle.dump(results,open(store_path,"wb"))
            print(f"区间{runtime_interval.group(1)}至{runtime_interval.group(2)}的分词任务完成")
            result_filenames.append(store_path)
    # collect and check the results
    final_results=[]
    for result_filename in result_filenames:
        try:
            final_results.extend(pickle.load(open(result_filename,"rb")))
        except Exception as e:
            print(f"pickle文件{result_filename}读取失败，原因是{e}")
            print("在这一步出错通常是由于在将pickle文件写入硬盘时出现了错误，这个错误是偶然发生的且并不常见，重新运行程序通常可以解决此问题")
            print(f"自动删除出现错误的文件{result_filename}并重新运行该部分的分词")
            os.remove(result_filename)
            clear_respawnpoint_before_run=False
            return tokenizeWordfromDF(runtime_code,input_file,tokenize_column_name,tokenized_column_name,user_added_stop_words_filename,user_added_critic_words_filename,user_added_other_words_filename,size_per_respawn_chunk_GB,only_retain_meaningful_words,omit_content_in_parentheses,delete_single_character,minor_retain_words_thereshold,other_preprocessing_injection,output_filename,clear_respawnpoint_before_run,clear_respawnpoint_upon_conplete)
    final_results.sort() # we sort it to the natural number order so that we can write the results into the Dataframe according to the order instead of the index per se
    final_results=zip(*final_results)
    # concat the tokenized result into the original DataFrame
    input_file=pickle.load(open(f"respawnpoint/{runtime_code}_input_dataframe_backup.pkl","rb"))
    input_file=input_file.loc[list(final_results.__next__())] # note that tuple is not allowed as the parameter in the loc method
    if only_retain_meaningful_words: # if the user asks to retain the meaningful words only, the word frequency is needed for lexical analysis
        input_file[tokenized_column_name]=final_results.__next__()
        input_file["critic_words_count"]=final_results.__next__()
        input_file["all_words_count"]=final_results.__next__()
        input_file["critic_words_frequency"]=final_results.__next__()
    else:
        input_file["tokenized_text_mit_pos"]=final_results.__next__()
    # save the final DataFrame
    input_file=input_file.set_index(index_name)
    saveConcatedDataAsFinalResult(runtime_code,input_file,output_filename,clear_respawnpoint_upon_conplete)
    return input_file

def tokenizeWordfromDFusingConfigMenu(config_menu):
    # a shortcut to run the code using a config menu to avoid input to many parameters
    runtime_code=config_menu["runtime_code"]
    input_file=config_menu["input_file"]
    tokenize_column_name=config_menu["tokenize_column_name"]
    tokenized_column_name=config_menu["tokenized_column_name"]
    user_added_stop_words_filename=config_menu.get("user_added_stop_words_filename",None)
    user_added_critic_words_filename=config_menu.get("user_added_critic_words_filename",None)
    user_added_other_words_filename=config_menu.get("user_added_other_words_filename",None)
    size_per_respawn_chunk_GB=config_menu.get("size_per_respawn_chunk_GB",0.6)
    only_retain_meaningful_words=config_menu.get("only_retain_meaningful_words",True)
    omit_content_in_parentheses=config_menu.get("omit_content_in_parentheses",True)
    delete_single_character=config_menu.get("delete_single_character",True)
    minor_retain_words_thereshold=config_menu.get("minor_retain_words_thereshold",6)
    other_preprocessing_injection=config_menu.get("other_preprocessing_injection",None)
    output_filename=config_menu.get("output_filename",None)
    clear_respawnpoint_before_run=config_menu.get("clear_respawnpoint_before_run",False)
    clear_respawnpoint_upon_conplete=config_menu.get("clear_respawnpoint_upon_conplete",True)
    return tokenizeWordfromDF(runtime_code,input_file,tokenize_column_name,tokenized_column_name,user_added_stop_words_filename,user_added_critic_words_filename,user_added_other_words_filename,size_per_respawn_chunk_GB,only_retain_meaningful_words,omit_content_in_parentheses,delete_single_character,minor_retain_words_thereshold,other_preprocessing_injection,output_filename,clear_respawnpoint_before_run,clear_respawnpoint_upon_conplete)
