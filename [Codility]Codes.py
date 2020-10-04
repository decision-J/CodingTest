# Lesson 0: test case of coding test
A = [1,3,6,4,1,2] # 5
A = [1,2,3] # 4
A = [-1,-3] # 1
A = [3] # 4
A = [0] # 1
A = [-1] # 1
A = [-1, 1, 2, 5] # 3
A = [-1, 0] # 1


from random import *
import random
numlist = list(range(-100000, 1000000))
A = random.choices(numlist, k=100000)
len(A)

solution(A)

def solution(A):
    if (len(A) == 1) and A[0]>=0:
        return A[0] + 1

    count = 0
    dict = {}
    for i in A:
        if i < 0 :
            count += 1
        elif i >= 0:
            if i in dict:
                dict[i] += 1
            else :
                dict[i] = 1

    if count == len(A) :
        return 1

    sort_dict = sorted(dict.keys())
    compare = list(range(1,max(sort_dict)+1))

    if len(sort_dict) == len(compare):
        return max(sort_dict) + 1
    elif compare == []:
        return 1
    else :
        for i in range(len(dict)):
            if sort_dict[i] != compare[i]:
                return compare[i]
            elif i == (len(dict)-1):
                return 1





# Lesson 1: Binary Gap
def solution(N):
    binary = bin(N)[2:]

    loc = []
    for i in range(len(binary)):
        if binary[i] == '1' :
            loc.append(i)

    gap = []
    for i in range(len(loc)-1):
        gap.append(loc[i+1] - loc[i] - 1)

    if gap == []:
        answer = 0
    else :
        answer = max(gap)

    return(answer)

solution(1041)


# Lesson 2: CyclicRotation
A = [3,7,6]
K = 3

[6,3,7], [7,6,3] [3,7,6] [6,3,7]

import copy as copy

def solution(A,K):
    if (len(A) == K or len(A) == 1):
        R = A
    elif K > len(A) :
        K = K - len(A)
        R = copy.copy(A)
        for i in range(len(R)):
            R[i] = A[i-K]
    else :
        R = copy.copy(A)
        for i in range(len(R)):
            R[i] = A[i-K]

    return(R)

solution(A, K)



# Lesson 2: OddOccurrencesInArray
A = [9,3,9,3,3,3,7]

## version 1: 66 score
def solution(A) :
    A_dict = dict((x,A.count(x)) for x in set(A))
    A_value = list(A_dict.values())
    A_key = list(A_dict.keys())

    for i in range(len(A_value)) :
        if (A_value[i] % 2) != 0:
            return A_key[i]

solution(A)


## version 2: 100 score
def solution(A) :
    A_dict = {}
    for i in A:
        if i in A_dict:
            A_dict[i] += 1
        else :
            A_dict[i] = 1
    A_value = list(A_dict.values())
    A_key = list(A_dict.keys())

    for i in range(len(A_value)) :
        if (A_value[i] % 2) != 0:
            return A_key[i]

solution(A)


# Lesson 3: FrogJmp
X = 10 ; Y = 85 ; D = 30
X = 10 ; Y = 10 ; D = 30

def solution(X, Y, D):
    if Y==X :
        return 0
    else :
        return -(-(Y-X) // D)

solution(X,Y,D)


# Lesson 3: PermMissingElem
A = [2,3,4,5]
A = []

## version 1: 50 score
def solution(A):
    sort_A = sorted(A)

    for i in range(len(A)-1):
        if (sort_A[i+1] - sort_A[i]) != 1 :
            return sort_A[i]+1

solution(A)

## version 2: 50 score
A = [2,3,4,5]
A = []
A = [2]

def solution(A):
    sort_A = sorted(A)

    if len(A) == 0:
        return 1

    elif sort_A[0] == 2:
        return 1

    else :
        for i in range(len(A)-1):
            if sort_A[i+1] != (sort_A[i] + 1) :
                return sort_A[i]+1

solution(A)

## version 3: 100 score
A = [2,3,4,5]
A = []
A = [2]

def solution(A):
    N = len(A)
    return sum(range(N+2)) - sum(A)

solution(A)



# Lesson 3: TapeEquilibrium

## version 1 : 53 score
A = [3,1,2,4,3]
A = [-3,1,2,4,3]

def solution(A):
    sum_all = sum(A)
    P = 1
    diff = []

    while P < len(A):
        sum_P = sum(A[0:P])
        diff.append(abs(sum_P - (sum_all - sum_P)))

        if abs(sum_P - (sum_all - sum_P)) == 0:
            diff = [0]
            break

        P += 1

    return min(diff)

solution(A)

## version 2 :  53 score
A = [3,1,2,4,3]
A = [-3,1,2,-4,3]
A = [2,2,2,2]
A = [1,2,3,4,5]

A = [3,1,2,4,10]

from random import *
import random
numlist = list(range(-1000, 1000))
A = random.choices(numlist, k=10000)
len(A)

def solution(A):
    sum_all = sum(A)
    diff = {}
    P=1

    while P < len(A):
        sum_P = sum(A[0:P])
        diff[P] = abs(sum_P - (sum_all - sum_P))
        if diff[P] == 0:
            break
        P += 1

    return min(diff.values())

solution(A)

## version 3 :  100 score
A = [3,1,2,4,3]

from random import *
import random
numlist = list(range(-1000, 1000))
A = random.choices(numlist, k=20000)
len(A)

def solution(A):
    sum1 = 0
    sum2 = sum(A)
    diff = []

    for P in range(1,len(A)) :
        sum1 += A[P-1]
        sum2 -= A[P-1]
        sum_diff = abs(sum1 - sum2)
        diff.append(sum_diff)

    return min(diff)

solution(A)


# Lesson 4: FrogRiverOne
## version 1: 54 score
A = [1,3,1,4,2,3,4,5]
X = 5

from random import *
import random
X = 20000
numlist = list(range(1, X+1))
A = random.choices(numlist, k=100000)
len(A)

def solution(X, A):
    dict = {}
    for i in range(len(A)):
        if A[i] not in dict.values() :
            dict[i] = A[i]

    if sorted(dict.values()) == [x+1 for x in list(range(X))] :
        return max(sorted(dict.keys()))
    else :
        return -1

solution(X,A)

## version 2: 54 score
A = [1,3,1,4,2,3,5,4]
X = 5

from random import *
import random
X = 100000
numlist = list(range(1, X+1))
A = random.sample(numlist, k=100000)
len(A)
min(A)
max(A)

def solution(X, A):
    if sum(set(A)) == sum(range(X+1)):
        dict = {}
        for i in range(1,X+1):
            dict[i] = A.index(i)

        return max(dict.values())

    else :
        return -1


solution(X,A)

## version 3 from internet: 100 score
A = [1,3,1,4,2,3,5,4]
X = 5

from random import *
import random
X = 100000
numlist = list(range(1, X+1))
A = random.sample(numlist, k=100000)
len(A)
min(A)
max(A)

def solution(X, A):
    check = [0] * X
    check_sum = 0

    for i in range(len(A)):
        if check[A[i]-1] == 0:
            check[A[i]-1] = 1
            check_sum += 1
            if check_sum == X:
                return i
    return -1

solution(X, A)


# Lesson 4: MaxCounters
## version 1: 66 score
A = [3,4,4,6,1,4,4]
N = 5

from random import *
import random
N = 100000
numlist = list(range(1, N+1))
A = random.choices([100001], k=100000)
len(A)
max(A)
min(A)

def solution(N, A):
    operator = [0] * N

    for i in A:
        if i != N+1:
            operator[i-1] += 1
        else :
            operator = [max(operator)] * N

    return operator

 solution(N, A)

## version 2 : 100 score
A = [3,4,4,6,1,4,4,6,6,5,1,2,6,6,6,2,3,4,5]
N = 5

def solution(N, A):
    if (N+1) in A :
        dict1={0:0}
        for i in range(len(A)):
            if A[i] == N+1:
                dict1[i] = 1
        idx = list(dict1.keys())


        max_value = []

        for i in range(0, len(idx)):
            if i+1 != len(idx) :
                dict2 = {0:0}
                for j in A[idx[i]:idx[i+1]]:
                    if j != N+1:
                        if j in dict2.keys() :
                            dict2[j] += 1
                        else :
                            dict2[j] = 1
                max_value.append(max(dict2.values()))

        operator = [sum(max_value)] * N

        for i in A[(idx[len(idx)-1]+1):]:
            operator[i-1] += 1

    else :
        operator = [0] * N

        for i in A:
            operator[i-1] += 1

    return operator

solution(N, A)


# Lesson 5: CountDiv
## version 1: 50 score
def solution(A, B, K):
    if A == 0:
        return -(-(B+1-A) // K)
    else :
        return -(-(B-A) // K)

solution(0, 20000000, 1)
solution(10, 10, 5)


# Lesson 6: Distinct
## version 1: 100 score
A = [2,1,1,2,3,1]

from random import *
import random
N = 1000000
numlist = list(range(-N, N+1))
A = random.choices(numlist, k=100000)
len(A)
max(A)
min(A)

def solution(A):
    dict = {}
    for i in A:
        if i in dict:
            dict[i] += 1
        else :
            dict[i] = 1

    return len(dict.keys())

solution(A)

## version 2: 100 score
def solution(A):
    return len(set(A))

solution(A)


# Lesson 7: Nesting
## version 1: 12 score
S = "(()(())())"
S = "(()(())("

def solution(S):
    dict = {0:0, 1:0}
    for i in range(len(S)):
        if S[i] =="(":
            dict[0] += 1
        else :
            dict[1] += 1

    if list(dict.values())[0] == list(dict.values())[1]:
        return 1
    else :
        return 0

solution(S)

## version 2: 62 score
S = "(()(())())"
S = "()()"
S = "(()(())("
S = "))))(((("
S = "()))((()"
S = "(()())((())(())()())"

def solution(S):
    while "()" in S:
        for i in range(len(S)-1):
            if S[i:i+2] == "()":
                if i == 0:
                    S = S[i+2:len(S)]
                elif i == (len(S)-2) :
                    S = S[0:i]
                else :
                    S = S[0:i] + S[i+2:len(S)]

    if len(S) > 0 :
        return 0
    else :
        return 1

solution(S)

## version 3: 75 score
S = "(()(())())" # 1
S = "())" # 0
S = "()()" # 1
S = "(()(())(" # 0
S = "))))((((" # 0
S = "()))((()" # 0
S = "(()())((())(())()())" # 1
S = "(()"
S = "("

solution(S)

def solution(S):
    dict = {0:0, 1:0, 2:0, 3:0}
    for i in range(len(S)):
        if S[i] =="(":
            dict[0] += 1
            dict[2] += i
        else :
            dict[1] += 1
            dict[3] += i

    if list(dict.values())[0] == list(dict.values())[1] and list(dict.values())[2] < list(dict.values())[3]:
        return 1
    elif list(dict.values())[0] == list(dict.values())[1] == list(dict.values())[2] == list(dict.values())[3]:
        return 1
    else :
        return 0

## version 4 from internet : 100 score
S = "(()(())())" # 1
S = "())" # 0

def solution(S):
    # write your code in Python 3.6

    if len(S) == 0: return 1
    if len(S) == 1: return 0

    stack = []

    for char in S:
        if char=="(":
            stack.append(char)
        else:
            if len(stack) > 0:
                stack.pop()
            else:
                return 0
    if len(stack) > 0: return 0
    return 1


# Lesson 8: Dominator
A = [3,4,3,4,4,4,4,4,2,3,-1,3,3]
A = []

solution(A)

def solution(A):
    if A == [] :
        return -1

    dict = {}
    for i in A:
        if i in dict :
            dict[i] += 1
        else :
            dict[i] = 1

    if max(list(dict.values())) > (len(A)/2) :
        max_value = max(list(dict.values()))
    else :
        return -1

    idx = list(dict.values()).index(max_value)
    dominator = list(dict.keys())[idx]

    return A.index(dominator)


# Lesson 9: MaxSliceSum
## version 1: 30 score
A = [3,2,-6,4,0]
A = [3,2,-6,4,0,-3,2,3,5,-2]
A = [3]
A = [3,2,4,0]

from random import *
import random
N = 1000000
numlist = list(range(-N, N+1))
A = random.choices(numlist, k=1000000)
len(A)
max(A)
min(A)

def solution(A):
    idx=[]
    for i in range(len(A)) :
        if A[i]<0 :
            idx.append(i)

    if idx == [] :
        return sum(A)
    else :
        sums = []
        for j in range(len(idx)+1):
            if j==0:
                sums.append(sum(A[:idx[j]]))
            elif j==len(idx):
                sums.append(sum(A[idx[j-1]+1:]))
            else :
                sums.append(sum(A[idx[j-1]+1:idx[j]]))

        return max(sums)

solution(A)

## version 2: 69 score
A = [2,-6,5,-3,4]
A = [3,2,-6,4,0,-3,2,3,5,-2]
A = [3]
A = [3,2,4,0]
A = [-6,-3,-2]
A = [-1]

from random import *
import random
N = 1000000
numlist = list(range(-N, N+1))
A = random.choices(numlist, k=1000000)
len(A)
max(A)
min(A)

def solution(A):
    idx=[]
    for i in range(len(A)) :
        if A[i]<0 :
            idx.append(i)

    if idx == [] :
        return sum(A)
    elif idx == list(range(len(A))):
        return max(A)
    else :
        sums = []
        for j in range(len(idx)+1):
            if j==0:
                sums.append(sum(A[:idx[j]]))
            elif j==len(idx):
                sums.append(sum(A[idx[j-1]+1:]))
            else :
                sums.append(sum(A[idx[j-1]+1:idx[j]]))

        return max(sums)

solution(A)

## version 3 from internet
A = [1, -7,-8, 2]

def solution(A):
    max = A[0]
    acc = 0

    for e in A:
        acc += e
        if acc > max:
            max = acc

        if acc < 0:
            acc = 0

    return max


# Lesson 10: CountFactors
## version 1 : 100 score
N = 24 # 8
N = 1 # 1
N = 11 # 2
N = 49 # 3
N = 16 # 5

N = 100000000
N = 2147483647

solution(N)

def solution(N):
    if N == 1:
        return 1

    factors = 2
    i = 2
    counts = 0

    while (i <= N):
        if i > N // i:
            break
        if i**2 == N :
            factors += 1
            break
        if (N % i == 0) :
            counts += 1
        i += 1

    return factors + (counts * 2)


# Lesson 11: CountNonDivisible
## version 1: 33 score
A = [3,1,2,3,6] # [2,4,3,2,0]
A = [4,3,1,2,3,6] # [3,3,5,4,3,1]
A = [2,3,4]

solution(A)

def solution(A):
    sort_A = sorted(A)
    dict = {}

    for i in range(1,len(sort_A)+1):
        if sort_A[i-1] == 1:
            counts = len(A) - 1
        else :
            counts = 0
            counts += len(sort_A[i:])
            for j in sort_A[:i-1] :
                if sort_A[i-1] % j != 0 :
                    counts += 1

        dict[sort_A[i-1]] = counts

    return [dict[x] for x in A]


# Lesson 12: ChocolatesByNumbers
## version 1: 37 score
N, M = 10, 4
solution(N, M)

N, M = 10000, 6
solution(N, M)

N, M = 24, 18
solution(N, M)

def solution(N, M):
    if N % M == 0 :
        return N // M
    else :
        chocolates = [(x * M) for x in list(range((N // M)+1))]
        while 1==1 :
            if len(chocolates) != len(set(chocolates)) :
                break
            else :
                chocolates += [((x * M + abs((chocolates[len(chocolates)-1] + M - N))) % N) for x in list(range((N // M)+1))]
        return len(chocolates) - 1


# Lesson 15: AbsDistinct
A = [-5, -3, -1, 0, 3, 6] # 5

from random import *
import random
A = random.choices(range(-2147483648, 2147483648), k=100000)
len(A)

def solution(A):
    return len(set([abs(x) for x in A]))


# Lesson 16: MaxNonoverlappingSegments
## version 1: 30 score
A, B = [1,3,7,9,9], [5,6,8,9,10]
solution(A, B)
A, B = [1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]

def solution(A, B):
    seg = list(range(A[0], B[0]+1))
    counts = 0
    for i in range(1,len(A)):
        if A[i] not in seg :
            counts += 1
            seg += list(range(A[i], B[i]+1))
    return (counts + 1)
