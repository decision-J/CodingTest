# Hash part

##################################################
############## 1. 완주하지 못한 선수 ##############
##################################################

# My code 1 : 40점(정확성 40 + 효율성 0)
import numpy as np

participants = [["leo", "kiki", "eden"], ["marina", "josipa", "nikola", "vinko", "filipa"], ["mislav", "stanko", "mislav", "ana"]]
completions = [["eden", "kiki"], ["josipa", "filipa", "marina", "nikola"], ["stanko", "ana", "mislav"]]

def solution(participant, completion):
    answer = [participant for participant in participant if participant not in completion] # 중복case가 없다면 쉽게 찾을 수 있음

    if answer == [] : # 중복case가 있다면 위 코드의 answer가 blank로 나옴. dictionary를 이용해서 중복 차이 확인 후 해결

        count={}
        participant.sort()
        for i in participant:
            try: count[i] += 1
            except: count[i]=1

        count1={}
        completion.sort()
        for i in completion:
            try: count1[i] += 1
            except: count1[i]=1

        diff = [list(count.values())[i] - list(count1.values())[i] for i in range(len(count))]
        loc = np.array(diff).argmax()

        answer = participant[loc]

    else : answer = answer[0]

    return answer

[solution(participants[i], completions[i]) for i in range(len(participants))]

# My code 2 : 40점(정확성 40 + 효율성 0)
## 애초에 answer를 hash로 만들고 participant에서 +1 , completion에서 -1을 해가는 방법
import numpy as np

participants = [["leo", "kiki", "eden"], ["marina", "josipa", "nikola", "vinko", "filipa"], ["mislav", "stanko", "mislav", "ana"]]
completions = [["eden", "kiki"], ["josipa", "filipa", "marina", "nikola"], ["stanko", "ana", "mislav"]]

def solution(participant, completion):
    answer = {}
    for i in participant:
        try: answer[i] += 1
        except: answer[i]=1

    for i in completion:
        try: answer[i] -= 1
        except: answer[i] = 0

    loc = np.array(list(answer.values())).argmax()

    answer = participant[loc]

    return answer

[solution(participants[i], completions[i]) for i in range(len(participants))]

# Best code : 100점
## 위 코드와 달리 -1이 아니라 del로 이름을 제거해버림. np의 import 및 동작이 필요 없음
participants = [["leo", "kiki", "eden"], ["marina", "josipa", "nikola", "vinko", "filipa"], ["mislav", "stanko", "mislav", "ana"]]
completions = [["eden", "kiki"], ["josipa", "filipa", "marina", "nikola"], ["stanko", "ana", "mislav"]]

def solution(participant, completion):
    hash ={}
    for i in participant:
        if i in hash:
            hash[i] += 1
        else:
            hash[i] = 1

    for i in completion:
        if hash[i] == 1:
            del hash[i]
        else: hash[i] -= 1

    answer = list(hash.keys())[0]

    return answer

[solution(participants[i], completions[i]) for i in range(len(participants))]


##################################################
############## 2. 전화번호 목록 ###################
##################################################
phone_book = ["119", "97674223", "1195524421"]
phone_book = ["123", "456", "789"]
phone_book = ["12", "123", "1235", "567", "88"]

# My code 1 : 84.6점 (정확성 84.6 + 효율성 0)
from copy import copy

hash ={}
for i in phone_book:
    hash[i] = True

for i in range(len(phone_book)) :
    x = phone_book[i]
    candidate = copy(phone_book)
    candidate.pop(i)
    for y in candidate :
         if hash[y] == False :
            pass
         elif y[0:len(x)] == x :
             hash[y] = False
         else : hash[y] = True

answer = sum(list(hash.values())) == len(phone_book)
answer


# My code 2 : 점 (정확성 84.6  + 효율성 15.4)
phone_book = ["119", "97674223", "1195524421"]
phone_book = ["123", "456", "789"]
phone_book = ["12", "123", "1235", "567", "88"]

## 효율성 테스트 용
import numpy as np
phone_book = []
for i in range(1000000):
    x = i
    phone_book.append(str(int(np.random.uniform(0,20000000000))))
len(phone_book)

## Main
answer = True

phone_book.sort()

for i in range(len(phone_book)-1):
    if phone_book[i] in phone_book[i+1]:
        answer = False

answer



####################################################################################################################################################
# 스택/큐 part

#########################################
############## 1. 주식가격 ##############
########################################









####################################################################################################################################################
# 정렬 part

#########################################
############## 1. K번째 수 ##############
########################################
array = [1,5,2,6,3,7,4]
commands = [[2,5,3],[4,4,1],[1,7,3]]

# My code 1 : 100점
answer = [sorted(array[commands[i][0]-1:commands[i][1]])[commands[i][2]-1] for i in range(len(commands))]
answer





####################################################################################################################################################
# 완전탐색 part

########################################
############## 1. 모의고사 ##############
########################################
answers = [1,2,3,4,5]
answers = [1,3,2,4,2]
answers = [1,3,2,4,2] * 2000
answers = [5]
answers = []

# My code 1 : 78.6점
supo1 = [1, 2, 3, 4, 5] * int(10000/len([1, 2, 3, 4, 5]))
supo2 = [2, 1, 2, 3, 2, 4, 2, 5] * int(10000/len([2, 1, 2, 3, 2, 4, 2, 5]))
supo3 = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5] * int(10000/len([3, 3, 1, 1, 2, 2, 4, 4, 5, 5]))

ans1 = sum([int(supo1[0:len(answers)][i] == answers[i]) for i in range(len(answers))])
ans2 = sum([int(supo2[0:len(answers)][i] == answers[i]) for i in range(len(answers))])
ans3 = sum([int(supo3[0:len(answers)][i] == answers[i]) for i in range(len(answers))])

ans = [ans1, ans2, ans3]

answer = [(ans.index(max(ans),i) + 1) for i in range(ans.count(max(ans)))]
answer

# My code 2 : 100점
supo1 = [1, 2, 3, 4, 5] * int(10000/len([1, 2, 3, 4, 5]))
supo2 = [2, 1, 2, 3, 2, 4, 2, 5] * int(10000/len([2, 1, 2, 3, 2, 4, 2, 5]))
supo3 = [3, 3, 1, 1, 2, 2, 4, 4, 5, 5] * int(10000/len([3, 3, 1, 1, 2, 2, 4, 4, 5, 5]))

ans1 = sum([int(supo1[0:len(answers)][i] == answers[i]) for i in range(len(answers))])
ans2 = sum([int(supo2[0:len(answers)][i] == answers[i]) for i in range(len(answers))])
ans3 = sum([int(supo3[0:len(answers)][i] == answers[i]) for i in range(len(answers))])

ans = [ans1, ans2, ans3]

answer = []
for i in range(len(ans)):
    if ans[i] == max(ans):
        answer.append(i+1)

answer



####################################################################################################################################################
# 탐욕법 part

#######################################
############## 1. 체육복 ##############
#######################################
n = 3

lost = [2, 4]
lost = [3]

reserve = [1, 3, 5]
reserve = [3]
reserve = [1]

# My code 1 : 100점

student = [i+1 for i in range(n)]

hash = {}
for i in student:
    if i in student:
        hash[i] = 1

    if i in reserve:
        hash[i] += 1

    if i in lost:
        hash[i] -= 1

count = [0]
for i in student:
    if i == 1 :
        if hash[i] == 0:
            if hash[i+1] > 1 :
                hash[i+1] -= 1
                hash[i] += 1
            else :
                count[0] += 1

    elif i == n :
        if hash[i] == 0:
            if hash[i-1] > 1 :
                hash[i-1] -= 1
                hash[i] += 1
            else :
                count[0] += 1

    elif hash[i] == 0 :
        if hash[i-1] > 1 :
            hash[i-1] -= 1
            hash[i] += 1
        elif hash[i+1] > 1 :
            hash[i+1] -= 1
            hash[i] += 1
        else :
            count[0] += 1

answer = n - count[0]
answer
