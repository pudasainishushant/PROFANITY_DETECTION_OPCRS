from time import sleep
import random
from collections import Counter

first_name_list = []
with open('nepali_names.txt','r') as c:
    for line in c:
        for word in line.split():
            first_name_list.append(word)

middle_name_list = []
with open('middle_names.txt','r') as c:
    for line in c:
        for word in line.split():
            middle_name_list.append(word)


surnamess_list = []
with open('surnamess.txt','r') as f:
    for line in f:
        for word in line.split():
                surnamess_list.append(word)

print(len(first_name_list),len(middle_name_list),len(surnamess_list))

new_first_name_list = random.sample(first_name_list,20)
new_middle_name_list = middle_name_list
new_surname_list = surnamess_list

		
with open('new.txt', 'w+') as z:
	for i in new_first_name_list:
		counter=0
		for j in new_middle_name_list:	
			for k in surnamess_list:				
				num_1 = random.randint(0,len(first_name_list)-1)
				word_1 = first_name_list[num_1]
				num_2 = random.randint(0,len(middle_name_list)-1)
				word_2 = middle_name_list[num_2]
				num_3 = random.randint(0,len(surnamess_list)-1)
				word_3 = surnamess_list[num_3]
				if counter<5:
					required_comb = word_1 + ' ' + word_3
					print(required_comb)
					counter=counter+1
					print(counter)
				else:
					required_comb = word_1+' '+word_2+' '+word_3
					#print(required_comb)
					print(required_comb, file=z)
					#z.write(required_comb)
