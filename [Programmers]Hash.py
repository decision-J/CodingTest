# Hash part

##################################################
############## 1. 완주하지 못한 선수 ##############
##################################################

# My code 1
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

# My code 2
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

# Best code
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
############## 2. ##############
##################################################
