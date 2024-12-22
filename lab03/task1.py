def OR(x1, x2):
  if x1 >= 1 or x2 >= 1:
   return 1
  else:
   return 0
def AND(x1, x2):
  if x1 == 1 and x2 == 1:
   return 1
  else:
   return 0
def XOR(x1, x2):
 if AND(x1, x2) != OR(x1, x2):
  return 1
 else:
  return 0

print(f'OR(1,1) = {OR(1, 1)} OR(0,1) = {OR(0, 1)} OR(0,0) = {OR(0, 0)}')
print(f'AND(0,0) = {AND(0, 0)} AND(1,0) = {AND(1, 0)} AND(1,1) = {AND(1, 1)}')