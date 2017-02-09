import string as s

print("this is test")
word_list = {}
count = 0
entry = "I miss my little Batty, she was only 3 months old when she got sick, she was always so fun and full of life &lt;3 Even at the very end..."
entry = str(entry.lower())


for c in s.punctuation:
  entry = entry.replace(c, ' ' + c + ' ')
  entry_list = entry.split()
  for w in entry_list:
    if word_list.get(w, False) is False:
      word_list[w] = count
      count = count + 1

import numpy as np

A = np.array([ [1,2,3], [4,5,6], [7,8,9]]);

b = A[:2]

C_range = [10**c for c in range(-3,4)]
print(C_range)

C_range = np.random.uniform(-3, 3, 25)


print(range(25))
for i in range(25):
  print(i)

A=np.zeros((3,4))
print(A)