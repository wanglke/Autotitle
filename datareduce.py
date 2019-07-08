#encoding=utf-8
import os
with open('data1/财经.txt', 'rb') as f, open('data1/财经0.txt', 'wb') as fp:
    i = 0
    for line in f:
        if i < 1000:
            fp.write(line)
        else:
            break
        i += 1

file_txt = []
for _, _, files in os.walk('data'):
    for file in files:
        if os.path.splitext(file)[1] == '.txt':
            file_txt.append(file)


# import jieba
# def tokenizer(text):
#     '''
#     Simple Parser converting each document to lower-case, then
#         removing the breaks for new lines and finally splitting on the
#         whitespace
#     '''
#     stpwrdlst = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines()]
#     stpwrdlst.append(' ')
#     words_ls = []
#
#     words = [w for w in jieba.cut(text.replace('\n', '')) if w not in stpwrdlst]
#     return words
#
# print(tokenizer('夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'))