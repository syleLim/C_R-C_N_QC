import numpy as numpy
import codecs
import copy
import random

class process_data :
    def __init__(self) :
        # Different data sets to try.
        # Note: TREC has no development set.
        # And SUBJ and MPQA have no splits (must use cross-validation)
        self.FILE_PATHS = {"SST1": ("data/stsa.fine.phrases.train",
                           "data/stsa.fine.dev",
                           "data/stsa.fine.test"),
                      "SST2": ("data/stsa.binary.phrases.train",
                               "data/stsa.binary.dev",
                               "data/stsa.binary.test"),
                      "TREC": ("data/TREC.train.all", None,
                               "data/TREC.test.all"),
                        "SUBJ": ("data/subj.all", None, None),
                        "MPQA": ("data/mpqa.all", None, None)}
        
    def Get_data_path(self, dataset) :
        train, valid, test = self.FILE_PATHS[dataset]

        return train, valid, test


    def Get_data(self, data_path) :
        datas = []

        with codecs.open(data_path, 'r', encoding = 'latin-1') as f :
            for line in f :
                datas.append(line)

        return datas

    def data_info(self, data_list) :
        flag_1 = 0
        flag_2 = 0
        flag_3 = 0
        flag_4 = 0
        flag_5 = 0
        flag_6 = 0

        for data in data_list :
            if data[0] == '0' :
                flag_1 +=1
            elif data[0] == '1' :
                flag_2 +=1
            elif data[0] == '2' :
                flag_3 +=1
            elif data[0] == '3' :
                flag_4 +=1
            elif data[0] == '4' :
                flag_5 +=1
            elif data[0] == '5' :
                flag_6 +=1

        print(flag_1)
        print(flag_2)
        print(flag_3)
        print(flag_4)
        print(flag_5)
        print(flag_6)

    def get_param(self, datas) : 
        """
        return
        data_list = list [score, sentence]
        word_list = list [word]
        word_dict = dict {word : feature num}
        char_list = list [char]
        char_dict = list [char : feature num]

        word_max_num = longest word length
        sentence_max_num  = longest sentence length

        char_size = char_list size
        word_size = word_list size

        score_size = class size ( 0 ~ 5 ) = y size  
        """

        all_sentence = ""
        data_list= []
        all_word = []

        for data in datas :
             # maybe?

            score = data[0]        ### TODO : have to change by data
            sentence = data[2:-1]

            data_list.append([score, sentence]) ## [0, XXXXX] ....
            
            for word in sentence.split(' ') :
                all_word.append(word)

            all_sentence += sentence  ## 알파벳 추출용


        char_list = list(set(all_sentence)) # 알파벳 수
        #char_list.insert(0, "C_PAD") # padding 에 넣을 값 추가
        char_dict = {c : i+1 for i, c in enumerate(char_list)} # char마다 label 부여?
        char_size = len(char_list) # embedding size ,알파벳 등 의 개수
    
        word_list = list(set(all_word))
        #word_list.inset(0, 'W_PAD')
        word_dict = {w : i+1 for i, w in enumerate(word_list)}
        word_reverse_dict = {i+1 : w for i, w in enumerate(word_list)}
        word_size = len(word_list)

        data_list.sort(key=lambda s : len(s[1]))
        sentence_max_len = len(data_list[-1][1].split(' '))
        #print(data_list[-1][1])


        word_list.sort(key=lambda s : len(s))
        #print()
        word_max_len = len(word_list[-1])
        
        score_list = list(set(i[0] for i in data_list))
        score_size = len(score_list) # class size = 1~5 

        random.shuffle(data_list, random.random) # 최종으로 섞음

        return data_list, word_list, word_dict, word_reverse_dict, word_size, char_list, char_dict, char_size, score_list, score_size, word_max_len, sentence_max_len

    def Form_porcessing(self, data_list, word_dict, char_dict) :
        label_list = []

        for i, data in enumerate(data_list) :
            #print(data[1])
            words = data[1].split(' ')
            
            word_tags = []
            for j, word in enumerate(words) :
                word_tag = word_dict[word]
                
                char_tags = []
                chars = list(word)
                for k, char in enumerate(chars) :
                    char_tag = char_dict[char]
                    char_tags.append(char_tag)

                word_tags.append([copy.deepcopy(char_tags), word_tag])

            new_data = [data[0], word_tags]
            data_list[i] = copy.deepcopy(new_data)
            
        ### Format [([char_tags....], word_tag), ....]
        #print(data_list)
        return data_list


    def Get_word_lens(self, word_tag_list, word_reverse_dict) :
        word_lens = []
        word_max_len = 0

        for sentence in word_tag_list :
            word_lens_temp = []

            for word_tag in sentence :
                word = word_reverse_dict[word_tag]
                word_lens_temp.append(len(word))

                if len(word) > word_max_len :
                    word_max_len = len(word)

            word_lens.append(copy.deepcopy(word_lens_temp))

        return word_lens, word_max_len


    def Word_padding(self, word_max_len, sentence_max_len, char_tag_list) :
        ## TODO : word_tag -> word and cal length
        ##      : padding in char
        for sentence in char_tag_list :
            if sentence_max_len > len(sentence) :
                for j in range(sentence_max_len - len(sentence)) :
                    sentence.append([])

        for sentence in char_tag_list :
            for char_tags in sentence :
                if len(char_tags) < word_max_len :
                    for j in range(word_max_len - len(char_tags)) :
                        char_tags.append(0)

        return char_tag_list


    def Sentence_padding(self, max_len, word_tag_list, word_lens) :
        ## TODO : padding making
        ## data = word_tag_list
        sentence_lens = []
        #print(word_tag_list)
        for i, sentence in enumerate(word_tag_list) :
            sentence_lens.append(len(sentence))

            if len(sentence) < max_len :
                for j in range(max_len - len(sentence)) :
                    sentence.append(0)
                    word_lens[i].append(0)


        return word_tag_list, word_lens, sentence_lens

    def Get_sentence(self, word_reverse_dict, sentence) :
        se = []
        
        for tag in sentence :
            if tag == 0 :
                continue
            else :
                se.append(word_reverse_dict[tag])

        return se


if __name__ == "__main__" :
    pd = process_data()

    data_path = pd.Get_data_path('TREC')

    train_datas = pd.Get_data(data_path[0])
    test_datas = pd.Get_data(data_path[2])
    
    datas = train_datas + test_datas

    temp = pd.get_param(datas)

    data_list = temp[0]
    word_list = temp[1]
    word_dict = temp[2]
    word_reverse_dict = temp[3]
    word_size = temp[4]
    char_list = temp[5]
    char_dict = temp[6]
    char_size = temp[7]
    score_list = temp[8]
    score_size = temp[9]

    #data_form = pd.Form_porcessing(data_list, word_dict, char_dict)
    print(data_list[0])

    pd.data_info(data_list)

    #print(data_form)