# s="hellofuckerkchaterochala"

# def partitions(s):
#   yield [s]
#   if(len(s)>2):
#     for i in range(len(s)-1, 0, -1):
#       for g in partitions(s[i:]):
#         out = [s[:i]] + g
#         if not any([len(out[i]) == len(out[i+1]) and len(out[i])==1 for i in range(len(out)-1)]):
#           yield out

# a = list(partitions(s))
# for i in a:
#   for w in i:
#     if w=='fucker':
#       print("True")

str = "haina aja k vako khamcha lai tesai hero huncha k garna khojya ho kunni udaidinchu ani"

def subset(text):
  n = len(text)
    
  #For holding all the formed substrings  
  arr = []
  #This loop maintains the starting character  
  for i in range(0,n):  
    #This loop will add a character to start character one by one till the end is reached  
    for j in range(i,n):  
        arr.append(str[i:(j+1)])
  return arr

a = subset(str)
print(a)

# for i in a:
#   if i=='bau':
#     print("True")
   
# #Prints all the subset  
# print("All subsets for given string are: ");  
# for i in arr:  
#     print(i);  