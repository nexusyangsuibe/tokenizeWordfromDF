from tokenizeWordfromDF import tokenizeWordfromDFusingConfigMenu
import time
import os

if __name__=="__main__":

    # 分词配置文件
    word_tokenize_cfg={
        "runtime_code":"cyj", # 运行时代码，必选项，用于区分不同的运行时，用于输出文件名的开头，应该是字符串格式
        "input_file":"resources/csmar_news.pkl", # 输入文件，必选项，应该是一个pandas.DataFrame或内容是一个DataFrame的pickle文件的路径名
        "tokenize_column_name":"NewsContent", # 需要执行分词的列的列名，必选项，应该是字符串格式
        "tokenized_column_name":"NewsContentTokenized", # 分词结果的列的列名，必选项，应该是字符串格式，当且仅当only_retain_meaningful_words为True时生效
        "user_added_stop_words_filename":"resources/user_added_stop_words.txt", # 用户定义的停用词词表的路径名，可选项，默认为None即只根据词性进行筛选，，当且仅当only_retain_meaningful_words为True时生效，应该是一个以utf-8格式定义的每行一个词的文本文件的文件名
        "user_added_critic_words_filename":None, # 用户定义的关键词词表的路径名，可选项，默认为None，用于计算关键词的词频，，当且仅当only_retain_meaningful_words为True时生效，应该是一个以utf-8格式定义的每行一个词的文本文件的文件名
        "user_added_other_words_filename":"resources/user_added_other_words.txt", # 用户定义的其他词词表的路径名，可选项，默认为None，用于在分词程序未能正确识别特定词语的情况下将该词语加入词库以完成识别，应该是一个以utf-8格式定义的每行一个词的文本文件的文件名
        "size_per_respawn_chunk_GB":0.6, # 存档块的大小，可选项，默认为0.6，应该是浮点数格式，每完成固定行数的分词会自动在respawnpoint文件夹中保存以实现断点续运行，在每一次循环中只将固定行数的文本读入内存，因此也与内存使用量有关，一个经验法则是1.5*内存大小（GB）/CPU逻辑处理器处理/待处理DataFrame的内存大小（GB）
        "only_retain_meaningful_words":True, # 是否只保留有意义的词语，可选项，默认为True，应该是布尔值类型，若为True启用语义分析模式，去除没有实际意义的词语（例如语气词）和用户定义的额外停用词，并返回以空格分隔的token和关键词词频，若为False则启用语法分析模式，返回带有词性标注的分词
        "omit_content_in_parentheses":True, # 是否去除括号内的内容，可选项，默认为True，应该是布尔值类型，当且仅当only_retain_meaningful_words为True时生效，去除括号内的内容主要是为了防止同义重复（同位语）干扰词频计算
        "delete_single_character":True,  # 是否删去单字，可选项，默认为True，应该是布尔值类型，当且仅当only_retain_meaningful_words为True时生效，主要是用于去除“是”、“要”、“有”这些确实是实义动词但语义含量有限的单字
        "minor_retain_words_thereshold":6, # 为保留文本需要有意义的单词的最小数量，可选项，默认为6，应该是正整数类型，当且仅当only_retain_meaningful_words为True时生效,主要用于剔除信息含量有限的短文本，若不需要可以设为None
        "other_preprocessing_injection":None, # 用户注入的其他文本预处理程序，可选项，默认为None，应该是单变量的函数类型，请不要使用高级函数（例如闭包）以避免函数无法序列化进入多进程，通常用于掐头去尾等操作
        "output_filename":"example_output.pkl", # 输出文件名，可选项，默认为None即不需要输出，为防止文本中逗号与逗号分隔符混淆不支持csv，支持xlsx，数据量较大时会以1000000行为分割输出多个excel文件，支持pickle(.pkl)格式，输出文件会出现在finalresults文件夹中
        # 以下两项respwanpoint文件夹配置，respawnpoint文件夹用于存储临时文件，以避免在读取大量数据时占用过多内存造成程序崩溃，若不清空respawnpoint文件夹可能导致之前的临时文件被重复读入
        "clear_respawnpoint_before_run":True, # 开始运行前是否清空respawnpoint文件夹，可选项，默认为False，若无需使用上一次运行的结果或希望在运行前保持respwanpoint文件夹的清洁，可以设为True
        "clear_respawnpoint_upon_conplete":False # 完成后是否清空respawnpoint文件夹，可选项，默认为True，若在下一次运行时还需要使用本次运行的部分结果，请设置为False
    }

    print("程序运行开始\n")
    t0=time.time()
    if "respawnpoint" not in os.listdir():
        print(f"在工作目录{os.getcwd()}下未找到用于存储临时文件的respawnpoint文件夹，将自动创建")
        os.mkdir("respawnpoint")
    if "finalresults" not in os.listdir():
        print(f"在工作目录{os.getcwd()}下未找到用于存储最终结果的finalresults文件夹，将自动创建")
        os.mkdir("finalresults")
    rtn_wt=tokenizeWordfromDFusingConfigMenu(config_menu=word_tokenize_cfg)
    print(rtn_wt.head(10))
    print(f"\n程序运行完成，用时{(time.time()-t0):.4f}秒")