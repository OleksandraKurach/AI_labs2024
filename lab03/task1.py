def xor(x1, x2):
    #XOR через OR та AND
    or_result = x1 or x2
    and_result = x1 and x2
    xor_result = or_result and not and_result
    return int(xor_result)

print("xor(0, 0) =", xor(0, 0))  #0
print("xor(0, 1) =", xor(0, 1))  #1
print("xor(1, 0) =", xor(1, 0))  #1
print("xor(1, 1) =", xor(1, 1))  #0
