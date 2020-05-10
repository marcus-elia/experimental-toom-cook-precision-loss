# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:15:41 2020

@author: Marcus

This file is for experimentally determining the precision loss, for all three
(matrix, natural, efficient), hopefully up to at least Toom-10.
"""

import numpy as np

# ========================================
#
#             Math Functions
#
# ========================================
def EEA(a, b):
    """ Performs the extended Euclidean algorithm on a and b, returns
        (gcd, u, v) such that au + bv = gcd"""
    u = 1
    g = a
    x = 0
    y = b
    while y != 0:
        q = g // y
        t = g % y
        s = u - q*x
        u = x
        g = y
        x = s
        y = t
    v = (g - a*u) // b    
    return (g, u, v)

def inverse_mod(a, m):
    """ Returns the inverse of a mod m if it exists"""
    g,u,v = EEA(a, m)
    if g != 1:
        raise ValueError("({}, {}) != 1, can't invert".format(a,m))
    else:
        return ((u % m) + m) % m

def split_powers_of_two(n):
    """ Returns (a,b) where a*b = n and a is odd and b is a power of 2"""
    even_part = 1
    odd_part = n
    while odd_part % 2 == 0:
        odd_part //= 2
        even_part *= 2
    return (odd_part, even_part)

# ========================================
#
#        Toom-Cook Helper Functions
#
# ========================================
def schoolbook_mod(f, g, m):
    """ Uses schoolbook multiplication to multiply f and g mod 
        m. Returns the product as a list"""
    d = len(f) + len(g) - 1
    
    # initialize a list of zeros
    product = [0]*d
    
    # distribute through all possible combinations of coefficients
    for i in range(len(f)):
        for j in range(len(g)):
            product[i + j] = (product[i+j] + f[i]*g[j]) % m
    return product

def split(f, num_blocks):
    """ Splits the list f into num_blocks different blocks of equal size
        If it doesn't divide evenly, we put zeros on the end of the last
        block."""
    blocks = []
    copy_f = list(f)  # copy f so we don't ruin it!!!!!!!!
    while len(copy_f) % num_blocks != 0:
        copy_f.append(0)
    block_length = len(copy_f) // num_blocks
    index = 0
    while index + block_length < len(copy_f):
        blocks.append(copy_f[index:index+block_length])
        index += block_length
    blocks.append(copy_f[index:])
    return blocks    

def make_eval_list(n):
    """ In Toom-n, this makes the list of numbers to plug in"""
    eval_list = [0]
    for a in range(1, n-1):
        eval_list.append(a)
        eval_list.append(-a)
    eval_list.append(n-1)
    eval_list.append('infinity')
    return eval_list

def evaluate_blocks_mod(blocks, value, m):
    """ blocks is a list of lists, each list is the coefficients of a
        polynomial. But each list a coefficient. For example, if blocks is
        [[1,2],[3,4],[5,6]] and value is -2, we return
        [1,2] + [-6,-8] + [20,24] = [15, 18].  If the value is infinity,
        we return the leading coefficient.
        This does it mod m"""
        
    if value == 'infinity':
        return blocks[-1]
    
    # initialize an empty list of the right length
    answer = [0]*len(blocks[0])
    
    coefficient = 1
    for i in range(len(blocks)):
        for j in range(len(blocks[0])):
            answer[j] = (answer[j] + coefficient*blocks[i][j]) % m
        coefficient = (coefficient*value) % m
    return answer

def evaluate_blocks_list_mod(blocks, values, m):
    """ Evaluates the blocks on a list of values, and returns a list"""
    answer = []
    for value in values:
        answer.append(evaluate_blocks_mod(blocks, value, m))
    return answer

# ========================================
#
#             Interpolation
#
# ========================================

def solve_for_coefficients_mod(n, r, m, formulas="efficient"):
    """ This function handles the long formulas needed to explicitly find
        the Toom formulas for Toom-2 up to Toom-6.
        Enter 'matrix', 'natural', or 'efficient' for formulas"""
    if formulas == "natural":
        if n == 4:
            r0 = r[0]

            r6 = r['infinity']

            L = len(r0)

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6)
        
        if n == 5:
            r0 = r[0]

            r8 = r['infinity']

            L = len(r0)

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8)
        
        if n == 6:
            r0 = r[0]

            r10 = r['infinity']

            L = len(r0)

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10)
        
        if n == 7:
            r0 = r[0]

            r12 = r['infinity']

            L = len(r0)

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12)
        
        if n == 8:
            r0 = r[0]

            r14 = r['infinity']

            L = len(r0)

            r12 = [(-792*(r[1][i] + r[-1][i])
             + 495*(r[2][i] + r[-2][i])
             - 220*(r[3][i] + r[-3][i])
             + 66*(r[4][i] + r[-4][i])
             - 12*(r[5][i] + r[-5][i])
             + (r[6][i] + r[-6][i])
             + 924*r0[i]
             - 43589145600*r14[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(479001600)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
             - 7264857600*r14[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
             - 461260800*r14[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
             - 9369360*r14[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
             - 32760*r14[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
             - 2*r14[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r13 = [(429*r[1][i] 
             - 572*r[2][i]
             + 429*r[3][i]
             - 208*r[4][i]
             + 65*r[5][i]
             - 12*r[6][i]
             + r[7][i]
             - 132*r0[i]
             + 84*r2[i]
             - 252*r4[i]
             + 2004*r6[i]
             - 37212*r8[i]
             + 1710324*r10[i]
             - 325024572*r12[i]
             - 80789566956*r14[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6227020800)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
             - 3632428800*r13[i]
             - 28616744548*r14[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
             - 726485760*r13[i]
             - 4084385750*r14[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
             - 57657600*r13[i]
             - 239967004*r14[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
             - 1561560*r13[i]
             - 4717438*r14[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
             - 8190*r13[i]
             - 16382*r14[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            - r13[i]
            - r14[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14)
        
        if n == 9:
            r0 = r[0]

            r16 = r['infinity']

            L = len(r0)

            r14 = [(3003*(r[1][i] + r[-1][i])
             - 2002*(r[2][i] + r[-2][i])
             + 1001*(r[3][i] + r[-3][i])
             - 364*(r[4][i] + r[-4][i])
             + 91*(r[5][i] + r[-5][i])
             - 14*(r[6][i] + r[-6][i])
             + (r[7][i] + r[-7][i])
             - 3432*r0[i]
             - 12204960768000*r16[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(87178291200)
                denom_inv = inverse_mod(odd_part, m)
                r14[i] = (r14[i] * denom_inv) % m
                r14[i] = r14[i] // even_part

            r12 = [(-792*(r[1][i] + r[-1][i])
             + 495*(r[2][i] + r[-2][i])
             - 220*(r[3][i] + r[-3][i])
             + 66*(r[4][i] + r[-4][i])
             - 12*(r[5][i] + r[-5][i])
             + (r[6][i] + r[-6][i])
             + 924*r0[i]
             - 43589145600*r14[i]
             - 2528170444800*r16[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(479001600)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
             - 7264857600*r14[i]
             - 223134912000*r16[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
             - 461260800*r14[i]
             - 7904856960*r16[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
             - 9369360*r14[i]
             - 85307040*r16[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
             - 32760*r14[i]
             - 131064*r16[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
             - 2*r14[i]
             - 2*r16[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r15 = [(-1430*r[1][i] 
             + 2002*r[2][i]
             - 1638*r[3][i]
             + 910*r[4][i]
             - 350*r[5][i]
             + 90*r[6][i]
             - 14*r[7][i]
             + r[8][i]
             + 429*r0[i]
             - 264*r2[i]
             + 744*r4[i]
             - 5304*r6[i]
             + 81384*r8[i]
             - 2605944*r10[i]
             + 192387624*r12[i]
             - 55942352184*r14[i]
             - 20546119600536*r16[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(1307674368000)
                denom_inv = inverse_mod(odd_part, m)
                r15[i] = (r15[i] * denom_inv) % m
                r15[i] = r15[i] // even_part

            r13 = [(429*r[1][i] 
             - 572*r[2][i]
             + 429*r[3][i]
             - 208*r[4][i]
             + 65*r[5][i]
             - 12*r[6][i]
             + r[7][i]
             - 132*r0[i]
             + 84*r2[i]
             - 252*r4[i]
             + 2004*r6[i]
             - 37212*r8[i]
             + 1710324*r10[i]
             - 325024572*r12[i]
             - 80789566956*r14[i]
             - 871782912000*r15[i]
             - 8422900930332*r16[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6227020800)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
             - 3632428800*r13[i]
             - 28616744548*r14[i]
             - 210680870400*r15[i]
             - 1479485236228*r16[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
             - 726485760*r13[i]
             - 4084385750*r14[i]
             - 22313491200*r15[i]
             - 119387268038*r16[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
             - 57657600*r13[i]
             - 239967004*r14[i]
             - 988107120*r15[i]
             - 4037604460*r16[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
             - 1561560*r13[i]
             - 4717438*r14[i]
             - 14217840*r15[i]
             - 42784582*r16[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
             - 8190*r13[i]
             - 16382*r14[i]
             - 32766*r15[i]
             - 65534*r16[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            - r13[i]
            - r14[i]
            - r15[i]
            - r16[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16)
        
        if n == 10:
            r0 = r[0]

            r18 = r['infinity']

            L = len(r0)

            r16 = [(-11440*(r[1][i] + r[-1][i])
             + 8008*(r[2][i] + r[-2][i])
             - 4368*(r[3][i] + r[-3][i])
             + 1820*(r[4][i] + r[-4][i])
             - 560*(r[5][i] + r[-5][i])
             + 120*(r[6][i] + r[-6][i])
             - 16*(r[7][i] + r[-7][i])
             + (r[8][i] + r[-8][i])
             + 12870*r0[i]
             - 4268249137152000*r18[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20922789888000)
                denom_inv = inverse_mod(odd_part, m)
                r16[i] = (r16[i] * denom_inv) % m
                r16[i] = r16[i] // even_part

            r14 = [(3003*(r[1][i] + r[-1][i])
             - 2002*(r[2][i] + r[-2][i])
             + 1001*(r[3][i] + r[-3][i])
             - 364*(r[4][i] + r[-4][i])
             + 91*(r[5][i] + r[-5][i])
             - 14*(r[6][i] + r[-6][i])
             + (r[7][i] + r[-7][i])
             - 3432*r0[i]
             - 12204960768000*r16[i]
             - 1058170098585600*r18[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(87178291200)
                denom_inv = inverse_mod(odd_part, m)
                r14[i] = (r14[i] * denom_inv) % m
                r14[i] = r14[i] // even_part

            r12 = [(-792*(r[1][i] + r[-1][i])
             + 495*(r[2][i] + r[-2][i])
             - 220*(r[3][i] + r[-3][i])
             + 66*(r[4][i] + r[-4][i])
             - 12*(r[5][i] + r[-5][i])
             + (r[6][i] + r[-6][i])
             + 924*r0[i]
             - 43589145600*r14[i]
             - 2528170444800*r16[i]
             - 120467944396800*r18[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(479001600)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
             - 7264857600*r14[i]
             - 223134912000*r16[i]
             - 6289809926400*r18[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
             - 461260800*r14[i]
             - 7904856960*r16[i]
             - 131254905600*r18[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
             - 9369360*r14[i]
             - 85307040*r16[i]
             - 771695280*r18[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
             - 32760*r14[i]
             - 131064*r16[i]
             - 524280*r18[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
             - 2*r14[i]
             - 2*r16[i]
             - 2*r18[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r17 = [(4862*r[1][i] 
             - 7072*r[2][i]
             + 6188*r[3][i]
             - 3808*r[4][i]
             + 1700*r[5][i]
             - 544*r[6][i]
             + 119*r[7][i]
             - 16*r[8][i]
             + r[9][i]
             - 1430*r0[i]
             + 858*r2[i]
             - 2310*r4[i]
             + 15258*r6[i]
             - 206790*r8[i]
             + 5386458*r10[i]
             - 272513670*r12[i]
             + 30255826458*r14[i]
             - 12765597850950*r16[i]
             - 6622557957272742*r18[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(355687428096000)
                denom_inv = inverse_mod(odd_part, m)
                r17[i] = (r17[i] * denom_inv) % m
                r17[i] = r17[i] // even_part

            r15 = [(-1430*r[1][i] 
             + 2002*r[2][i]
             - 1638*r[3][i]
             + 910*r[4][i]
             - 350*r[5][i]
             + 90*r[6][i]
             - 14*r[7][i]
             + r[8][i]
             + 429*r0[i]
             - 264*r2[i]
             + 744*r4[i]
             - 5304*r6[i]
             + 81384*r8[i]
             - 2605944*r10[i]
             + 192387624*r12[i]
             - 55942352184*r14[i]
             - 20546119600536*r16[i]
             - 266765571072000*r17[i]
             - 3083760849804024*r18[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(1307674368000)
                denom_inv = inverse_mod(odd_part, m)
                r15[i] = (r15[i] * denom_inv) % m
                r15[i] = r15[i] // even_part

            r13 = [(429*r[1][i] 
             - 572*r[2][i]
             + 429*r[3][i]
             - 208*r[4][i]
             + 65*r[5][i]
             - 12*r[6][i]
             + r[7][i]
             - 132*r0[i]
             + 84*r2[i]
             - 252*r4[i]
             + 2004*r6[i]
             - 37212*r8[i]
             + 1710324*r10[i]
             - 325024572*r12[i]
             - 80789566956*r14[i]
             - 871782912000*r15[i]
             - 8422900930332*r16[i]
             - 75583578470400*r17[i]
             - 643521842437836*r18[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6227020800)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
             - 3632428800*r13[i]
             - 28616744548*r14[i]
             - 210680870400*r15[i]
             - 1479485236228*r16[i]
             - 10038995366400*r17[i]
             - 66394067988388*r18[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
             - 726485760*r13[i]
             - 4084385750*r14[i]
             - 22313491200*r15[i]
             - 119387268038*r16[i]
             - 628980992640*r17[i]
             - 3275389222070*r18[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
             - 57657600*r13[i]
             - 239967004*r14[i]
             - 988107120*r15[i]
             - 4037604460*r16[i]
             - 16406863200*r17[i]
             - 66398623804*r18[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
             - 1561560*r13[i]
             - 4717438*r14[i]
             - 14217840*r15[i]
             - 42784582*r16[i]
             - 128615880*r17[i]
             - 386371918*r18[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
             - 8190*r13[i]
             - 16382*r14[i]
             - 32766*r15[i]
             - 65534*r16[i]
             - 131070*r17[i]
             - 262142*r18[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            - r13[i]
            - r14[i]
            - r15[i]
            - r16[i]
            - r17[i]
            - r18[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18)
        if n == 11:
            r0 = r[0]

            r20 = r['infinity']

            L = len(r0)

            r18 = [(43758*(r[1][i] + r[-1][i])
             - 31824*(r[2][i] + r[-2][i])
             + 18564*(r[3][i] + r[-3][i])
             - 8568*(r[4][i] + r[-4][i])
             + 3060*(r[5][i] + r[-5][i])
             - 816*(r[6][i] + r[-6][i])
             + 153*(r[7][i] + r[-7][i])
             - 18*(r[8][i] + r[-8][i])
             + (r[9][i] + r[-9][i])
             - 48620*r0[i]
             - 1824676506132480000*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6402373705728000)
                denom_inv = inverse_mod(odd_part, m)
                r18[i] = (r18[i] * denom_inv) % m
                r18[i] = r18[i] // even_part

            r16 = [(-11440*(r[1][i] + r[-1][i])
             + 8008*(r[2][i] + r[-2][i])
             - 4368*(r[3][i] + r[-3][i])
             + 1820*(r[4][i] + r[-4][i])
             - 560*(r[5][i] + r[-5][i])
             + 120*(r[6][i] + r[-6][i])
             - 16*(r[7][i] + r[-7][i])
             + (r[8][i] + r[-8][i])
             + 12870*r0[i]
             - 4268249137152000*r18[i]
             - 527128768438272000*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20922789888000)
                denom_inv = inverse_mod(odd_part, m)
                r16[i] = (r16[i] * denom_inv) % m
                r16[i] = r16[i] // even_part

            r14 = [(3003*(r[1][i] + r[-1][i])
             - 2002*(r[2][i] + r[-2][i])
             + 1001*(r[3][i] + r[-3][i])
             - 364*(r[4][i] + r[-4][i])
             + 91*(r[5][i] + r[-5][i])
             - 14*(r[6][i] + r[-6][i])
             + (r[7][i] + r[-7][i])
             - 3432*r0[i]
             - 12204960768000*r16[i]
             - 1058170098585600*r18[i]
             - 73775500710912000*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(87178291200)
                denom_inv = inverse_mod(odd_part, m)
                r14[i] = (r14[i] * denom_inv) % m
                r14[i] = r14[i] // even_part

            r12 = [(-792*(r[1][i] + r[-1][i])
             + 495*(r[2][i] + r[-2][i])
             - 220*(r[3][i] + r[-3][i])
             + 66*(r[4][i] + r[-4][i])
             - 12*(r[5][i] + r[-5][i])
             + (r[6][i] + r[-6][i])
             + 924*r0[i]
             - 43589145600*r14[i]
             - 2528170444800*r16[i]
             - 120467944396800*r18[i]
             - 5167100908569600*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(479001600)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
             - 7264857600*r14[i]
             - 223134912000*r16[i]
             - 6289809926400*r18[i]
             - 169058189664000*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
             - 461260800*r14[i]
             - 7904856960*r16[i]
             - 131254905600*r18[i]
             - 2143293425280*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
             - 9369360*r14[i]
             - 85307040*r16[i]
             - 771695280*r18[i]
             - 6960985920*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
             - 32760*r14[i]
             - 131064*r16[i]
             - 524280*r18[i]
             - 2097144*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
             - 2*r14[i]
             - 2*r16[i]
             - 2*r18[i]
             - 2*r20[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r19 = [(-16796*r[1][i] 
             + 25194*r[2][i]
             - 23256*r[3][i]
             + 15504*r[4][i]
             - 7752*r[5][i]
             + 2907*r[6][i]
             - 798*r[7][i]
             + 152*r[8][i]
             - 18*r[9][i]
             + r[10][i]
             + 4862*r0[i]
             - 2860*r2[i]
             + 7436*r4[i]
             - 46420*r6[i]
             + 576236*r8[i]
             - 13098580*r10[i]
             + 532310636*r12[i]
             - 39968611540*r14[i]
             + 6350631494636*r16[i]
             - 3730771315561300*r18[i]
             - 2637991952943407764*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(121645100408832000)
                denom_inv = inverse_mod(odd_part, m)
                r19[i] = (r19[i] * denom_inv) % m
                r19[i] = r19[i] // even_part

            r17 = [(4862*r[1][i] 
             - 7072*r[2][i]
             + 6188*r[3][i]
             - 3808*r[4][i]
             + 1700*r[5][i]
             - 544*r[6][i]
             + 119*r[7][i]
             - 16*r[8][i]
             + r[9][i]
             - 1430*r0[i]
             + 858*r2[i]
             - 2310*r4[i]
             + 15258*r6[i]
             - 206790*r8[i]
             + 5386458*r10[i]
             - 272513670*r12[i]
             + 30255826458*r14[i]
             - 12765597850950*r16[i]
             - 6622557957272742*r18[i]
             - 101370917007360000*r19[i]
             - 1375210145685786630*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(355687428096000)
                denom_inv = inverse_mod(odd_part, m)
                r17[i] = (r17[i] * denom_inv) % m
                r17[i] = r17[i] // even_part

            r15 = [(-1430*r[1][i] 
             + 2002*r[2][i]
             - 1638*r[3][i]
             + 910*r[4][i]
             - 350*r[5][i]
             + 90*r[6][i]
             - 14*r[7][i]
             + r[8][i]
             + 429*r0[i]
             - 264*r2[i]
             + 744*r4[i]
             - 5304*r6[i]
             + 81384*r8[i]
             - 2605944*r10[i]
             + 192387624*r12[i]
             - 55942352184*r14[i]
             - 20546119600536*r16[i]
             - 266765571072000*r17[i]
             - 3083760849804024*r18[i]
             - 32945548027392000*r19[i]
             - 332500281299403096*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(1307674368000)
                denom_inv = inverse_mod(odd_part, m)
                r15[i] = (r15[i] * denom_inv) % m
                r15[i] = r15[i] // even_part

            r13 = [(429*r[1][i] 
             - 572*r[2][i]
             + 429*r[3][i]
             - 208*r[4][i]
             + 65*r[5][i]
             - 12*r[6][i]
             + r[7][i]
             - 132*r0[i]
             + 84*r2[i]
             - 252*r4[i]
             + 2004*r6[i]
             - 37212*r8[i]
             + 1710324*r10[i]
             - 325024572*r12[i]
             - 80789566956*r14[i]
             - 871782912000*r15[i]
             - 8422900930332*r16[i]
             - 75583578470400*r17[i]
             - 643521842437836*r18[i]
             - 5269678622208000*r19[i]
             - 41890044885642492*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6227020800)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
             - 3632428800*r13[i]
             - 28616744548*r14[i]
             - 210680870400*r15[i]
             - 1479485236228*r16[i]
             - 10038995366400*r17[i]
             - 66394067988388*r18[i]
             - 430591742380800*r19[i]
             - 2750479262009668*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
             - 726485760*r13[i]
             - 4084385750*r14[i]
             - 22313491200*r15[i]
             - 119387268038*r16[i]
             - 628980992640*r17[i]
             - 3275389222070*r18[i]
             - 16905818966400*r19[i]
             - 86665431465638*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
             - 57657600*r13[i]
             - 239967004*r14[i]
             - 988107120*r15[i]
             - 4037604460*r16[i]
             - 16406863200*r17[i]
             - 66398623804*r18[i]
             - 267911678160*r19[i]
             - 1078605601420*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
             - 1561560*r13[i]
             - 4717438*r14[i]
             - 14217840*r15[i]
             - 42784582*r16[i]
             - 128615880*r17[i]
             - 386371918*r18[i]
             - 1160164320*r19[i]
             - 3482590102*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
             - 8190*r13[i]
             - 16382*r14[i]
             - 32766*r15[i]
             - 65534*r16[i]
             - 131070*r17[i]
             - 262142*r18[i]
             - 524286*r19[i]
             - 1048574*r20[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            - r13[i]
            - r14[i]
            - r15[i]
            - r16[i]
            - r17[i]
            - r18[i]
            - r19[i]
            - r20[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20)
        
        if n == 12:
            r0 = r[0]

            r22 = r['infinity']

            L = len(r0)

            r20 = [(-167960*(r[1][i] + r[-1][i])
             + 125970*(r[2][i] + r[-2][i])
             - 77520*(r[3][i] + r[-3][i])
             + 38760*(r[4][i] + r[-4][i])
             - 15504*(r[5][i] + r[-5][i])
             + 4845*(r[6][i] + r[-6][i])
             - 1140*(r[7][i] + r[-7][i])
             + 190*(r[8][i] + r[-8][i])
             - 20*(r[9][i] + r[-9][i])
             + (r[10][i] + r[-10][i])
             + 184756*r0[i]
             - 936667273148006400000*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2432902008176640000)
                denom_inv = inverse_mod(odd_part, m)
                r20[i] = (r20[i] * denom_inv) % m
                r20[i] = r20[i] // even_part

            r18 = [(43758*(r[1][i] + r[-1][i])
             - 31824*(r[2][i] + r[-2][i])
             + 18564*(r[3][i] + r[-3][i])
             - 8568*(r[4][i] + r[-4][i])
             + 3060*(r[5][i] + r[-5][i])
             - 816*(r[6][i] + r[-6][i])
             + 153*(r[7][i] + r[-7][i])
             - 18*(r[8][i] + r[-8][i])
             + (r[9][i] + r[-9][i])
             - 48620*r0[i]
             - 1824676506132480000*r20[i]
             - 309100200138842112000*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6402373705728000)
                denom_inv = inverse_mod(odd_part, m)
                r18[i] = (r18[i] * denom_inv) % m
                r18[i] = r18[i] // even_part

            r16 = [(-11440*(r[1][i] + r[-1][i])
             + 8008*(r[2][i] + r[-2][i])
             - 4368*(r[3][i] + r[-3][i])
             + 1820*(r[4][i] + r[-4][i])
             - 560*(r[5][i] + r[-5][i])
             + 120*(r[6][i] + r[-6][i])
             - 16*(r[7][i] + r[-7][i])
             + (r[8][i] + r[-8][i])
             + 12870*r0[i]
             - 4268249137152000*r18[i]
             - 527128768438272000*r20[i]
             - 51442361350668288000*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20922789888000)
                denom_inv = inverse_mod(odd_part, m)
                r16[i] = (r16[i] * denom_inv) % m
                r16[i] = r16[i] // even_part

            r14 = [(3003*(r[1][i] + r[-1][i])
             - 2002*(r[2][i] + r[-2][i])
             + 1001*(r[3][i] + r[-3][i])
             - 364*(r[4][i] + r[-4][i])
             + 91*(r[5][i] + r[-5][i])
             - 14*(r[6][i] + r[-6][i])
             + (r[7][i] + r[-7][i])
             - 3432*r0[i]
             - 12204960768000*r16[i]
             - 1058170098585600*r18[i]
             - 73775500710912000*r20[i]
             - 4555411900194355200*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(87178291200)
                denom_inv = inverse_mod(odd_part, m)
                r14[i] = (r14[i] * denom_inv) % m
                r14[i] = r14[i] // even_part

            r12 = [(-792*(r[1][i] + r[-1][i])
             + 495*(r[2][i] + r[-2][i])
             - 220*(r[3][i] + r[-3][i])
             + 66*(r[4][i] + r[-4][i])
             - 12*(r[5][i] + r[-5][i])
             + (r[6][i] + r[-6][i])
             + 924*r0[i]
             - 43589145600*r14[i]
             - 2528170444800*r16[i]
             - 120467944396800*r18[i]
             - 5167100908569600*r20[i]
             - 208331313744153600*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(479001600)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
             - 7264857600*r14[i]
             - 223134912000*r16[i]
             - 6289809926400*r18[i]
             - 169058189664000*r20[i]
             - 4419351149875200*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
             - 461260800*r14[i]
             - 7904856960*r16[i]
             - 131254905600*r18[i]
             - 2143293425280*r20[i]
             - 34682510016000*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
             - 9369360*r14[i]
             - 85307040*r16[i]
             - 771695280*r18[i]
             - 6960985920*r20[i]
             - 62711787600*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
             - 32760*r14[i]
             - 131064*r16[i]
             - 524280*r18[i]
             - 2097144*r20[i]
             - 8388600*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
             - 2*r14[i]
             - 2*r16[i]
             - 2*r18[i]
             - 2*r20[i]
             - 2*r22[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r21 = [(58786*r[1][i] 
             - 90440*r[2][i]
             + 87210*r[3][i]
             - 62016*r[4][i]
             + 33915*r[5][i]
             - 14364*r[6][i]
             + 4655*r[7][i]
             - 1120*r[8][i]
             + 189*r[9][i]
             - 20*r[10][i]
             + r[11][i]
             - 16796*r0[i]
             + 9724*r2[i]
             - 24596*r4[i]
             + 147004*r6[i]
             - 1708916*r8[i]
             + 35240284*r10[i]
             - 1237329236*r12[i]
             + 73853629564*r14[i]
             - 7850527669556*r16[i]
             + 1717351379730844*r18[i]
             - 1359124435588313876*r20[i]
             - 1272410676942417239876*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(51090942171709440000)
                denom_inv = inverse_mod(odd_part, m)
                r21[i] = (r21[i] * denom_inv) % m
                r21[i] = r21[i] // even_part

            r19 = [(-16796*r[1][i] 
             + 25194*r[2][i]
             - 23256*r[3][i]
             + 15504*r[4][i]
             - 7752*r[5][i]
             + 2907*r[6][i]
             - 798*r[7][i]
             + 152*r[8][i]
             - 18*r[9][i]
             + r[10][i]
             + 4862*r0[i]
             - 2860*r2[i]
             + 7436*r4[i]
             - 46420*r6[i]
             + 576236*r8[i]
             - 13098580*r10[i]
             + 532310636*r12[i]
             - 39968611540*r14[i]
             + 6350631494636*r16[i]
             - 3730771315561300*r18[i]
             - 2637991952943407764*r20[i]
             - 46833363657400320000*r21[i]
             - 734121065118879803860*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(121645100408832000)
                denom_inv = inverse_mod(odd_part, m)
                r19[i] = (r19[i] * denom_inv) % m
                r19[i] = r19[i] // even_part

            r17 = [(4862*r[1][i] 
             - 7072*r[2][i]
             + 6188*r[3][i]
             - 3808*r[4][i]
             + 1700*r[5][i]
             - 544*r[6][i]
             + 119*r[7][i]
             - 16*r[8][i]
             + r[9][i]
             - 1430*r0[i]
             + 858*r2[i]
             - 2310*r4[i]
             + 15258*r6[i]
             - 206790*r8[i]
             + 5386458*r10[i]
             - 272513670*r12[i]
             + 30255826458*r14[i]
             - 12765597850950*r16[i]
             - 6622557957272742*r18[i]
             - 101370917007360000*r19[i]
             - 1375210145685786630*r20[i]
             - 17172233341046784000*r21[i]
             - 201832098313986359142*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(355687428096000)
                denom_inv = inverse_mod(odd_part, m)
                r17[i] = (r17[i] * denom_inv) % m
                r17[i] = r17[i] // even_part

            r15 = [(-1430*r[1][i] 
             + 2002*r[2][i]
             - 1638*r[3][i]
             + 910*r[4][i]
             - 350*r[5][i]
             + 90*r[6][i]
             - 14*r[7][i]
             + r[8][i]
             + 429*r0[i]
             - 264*r2[i]
             + 744*r4[i]
             - 5304*r6[i]
             + 81384*r8[i]
             - 2605944*r10[i]
             + 192387624*r12[i]
             - 55942352184*r14[i]
             - 20546119600536*r16[i]
             - 266765571072000*r17[i]
             - 3083760849804024*r18[i]
             - 32945548027392000*r19[i]
             - 332500281299403096*r20[i]
             - 3215147584416768000*r21[i]
             - 30076927429146721464*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(1307674368000)
                denom_inv = inverse_mod(odd_part, m)
                r15[i] = (r15[i] * denom_inv) % m
                r15[i] = r15[i] // even_part

            r13 = [(429*r[1][i] 
             - 572*r[2][i]
             + 429*r[3][i]
             - 208*r[4][i]
             + 65*r[5][i]
             - 12*r[6][i]
             + r[7][i]
             - 132*r0[i]
             + 84*r2[i]
             - 252*r4[i]
             + 2004*r6[i]
             - 37212*r8[i]
             + 1710324*r10[i]
             - 325024572*r12[i]
             - 80789566956*r14[i]
             - 871782912000*r15[i]
             - 8422900930332*r16[i]
             - 75583578470400*r17[i]
             - 643521842437836*r18[i]
             - 5269678622208000*r19[i]
             - 41890044885642492*r20[i]
             - 325386564299596800*r21[i]
             - 2481686964269990316*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6227020800)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
             - 3632428800*r13[i]
             - 28616744548*r14[i]
             - 210680870400*r15[i]
             - 1479485236228*r16[i]
             - 10038995366400*r17[i]
             - 66394067988388*r18[i]
             - 430591742380800*r19[i]
             - 2750479262009668*r20[i]
             - 17360942812012800*r21[i]
             - 108550450893568228*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
             - 726485760*r13[i]
             - 4084385750*r14[i]
             - 22313491200*r15[i]
             - 119387268038*r16[i]
             - 628980992640*r17[i]
             - 3275389222070*r18[i]
             - 16905818966400*r19[i]
             - 86665431465638*r20[i]
             - 441935114987520*r21[i]
             - 2244295389943190*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
             - 57657600*r13[i]
             - 239967004*r14[i]
             - 988107120*r15[i]
             - 4037604460*r16[i]
             - 16406863200*r17[i]
             - 66398623804*r18[i]
             - 267911678160*r19[i]
             - 1078605601420*r20[i]
             - 4335313752000*r21[i]
             - 17403958407004*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
             - 1561560*r13[i]
             - 4717438*r14[i]
             - 14217840*r15[i]
             - 42784582*r16[i]
             - 128615880*r17[i]
             - 386371918*r18[i]
             - 1160164320*r19[i]
             - 3482590102*r20[i]
             - 10451964600*r21[i]
             - 31364282398*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
             - 8190*r13[i]
             - 16382*r14[i]
             - 32766*r15[i]
             - 65534*r16[i]
             - 131070*r17[i]
             - 262142*r18[i]
             - 524286*r19[i]
             - 1048574*r20[i]
             - 2097150*r21[i]
             - 4194302*r22[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            - r13[i]
            - r14[i]
            - r15[i]
            - r16[i]
            - r17[i]
            - r18[i]
            - r19[i]
            - r20[i]
            - r21[i]
            - r22[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22)
        
        if n == 13:
            r0 = r[0]

            r24 = r['infinity']

            L = len(r0)

            r22 = [(646646*(r[1][i] + r[-1][i])
             - 497420*(r[2][i] + r[-2][i])
             + 319770*(r[3][i] + r[-3][i])
             - 170544*(r[4][i] + r[-4][i])
             + 74613*(r[5][i] + r[-5][i])
             - 26334*(r[6][i] + r[-6][i])
             + 7315*(r[7][i] + r[-7][i])
             - 1540*(r[8][i] + r[-8][i])
             + 231*(r[9][i] + r[-9][i])
             - 22*(r[10][i] + r[-10][i])
             + (r[11][i] + r[-11][i])
             - 705432*r0[i]
             - 568744368255469486080000*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(1124000727777607680000)
                denom_inv = inverse_mod(odd_part, m)
                r22[i] = (r22[i] * denom_inv) % m
                r22[i] = r22[i] // even_part

            r20 = [(-167960*(r[1][i] + r[-1][i])
             + 125970*(r[2][i] + r[-2][i])
             - 77520*(r[3][i] + r[-3][i])
             + 38760*(r[4][i] + r[-4][i])
             - 15504*(r[5][i] + r[-5][i])
             + 4845*(r[6][i] + r[-6][i])
             - 1140*(r[7][i] + r[-7][i])
             + 190*(r[8][i] + r[-8][i])
             - 20*(r[9][i] + r[-9][i])
             + (r[10][i] + r[-10][i])
             + 184756*r0[i]
             - 936667273148006400000*r22[i]
             - 211124803367560642560000*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2432902008176640000)
                denom_inv = inverse_mod(odd_part, m)
                r20[i] = (r20[i] * denom_inv) % m
                r20[i] = r20[i] // even_part

            r18 = [(43758*(r[1][i] + r[-1][i])
             - 31824*(r[2][i] + r[-2][i])
             + 18564*(r[3][i] + r[-3][i])
             - 8568*(r[4][i] + r[-4][i])
             + 3060*(r[5][i] + r[-5][i])
             - 816*(r[6][i] + r[-6][i])
             + 153*(r[7][i] + r[-7][i])
             - 18*(r[8][i] + r[-8][i])
             + (r[9][i] + r[-9][i])
             - 48620*r0[i]
             - 1824676506132480000*r20[i]
             - 309100200138842112000*r22[i]
             - 40778478784550707200000*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6402373705728000)
                denom_inv = inverse_mod(odd_part, m)
                r18[i] = (r18[i] * denom_inv) % m
                r18[i] = r18[i] // even_part

            r16 = [(-11440*(r[1][i] + r[-1][i])
             + 8008*(r[2][i] + r[-2][i])
             - 4368*(r[3][i] + r[-3][i])
             + 1820*(r[4][i] + r[-4][i])
             - 560*(r[5][i] + r[-5][i])
             + 120*(r[6][i] + r[-6][i])
             - 16*(r[7][i] + r[-7][i])
             + (r[8][i] + r[-8][i])
             + 12870*r0[i]
             - 4268249137152000*r18[i]
             - 527128768438272000*r20[i]
             - 51442361350668288000*r22[i]
             - 4385609982489415680000*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20922789888000)
                denom_inv = inverse_mod(odd_part, m)
                r16[i] = (r16[i] * denom_inv) % m
                r16[i] = r16[i] // even_part

            r14 = [(3003*(r[1][i] + r[-1][i])
             - 2002*(r[2][i] + r[-2][i])
             + 1001*(r[3][i] + r[-3][i])
             - 364*(r[4][i] + r[-4][i])
             + 91*(r[5][i] + r[-5][i])
             - 14*(r[6][i] + r[-6][i])
             + (r[7][i] + r[-7][i])
             - 3432*r0[i]
             - 12204960768000*r16[i]
             - 1058170098585600*r18[i]
             - 73775500710912000*r20[i]
             - 4555411900194355200*r22[i]
             - 261131482210959360000*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(87178291200)
                denom_inv = inverse_mod(odd_part, m)
                r14[i] = (r14[i] * denom_inv) % m
                r14[i] = r14[i] // even_part

            r12 = [(-792*(r[1][i] + r[-1][i])
             + 495*(r[2][i] + r[-2][i])
             - 220*(r[3][i] + r[-3][i])
             + 66*(r[4][i] + r[-4][i])
             - 12*(r[5][i] + r[-5][i])
             + (r[6][i] + r[-6][i])
             + 924*r0[i]
             - 43589145600*r14[i]
             - 2528170444800*r16[i]
             - 120467944396800*r18[i]
             - 5167100908569600*r20[i]
             - 208331313744153600*r22[i]
             - 8083281646573056000*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(479001600)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
             - 7264857600*r14[i]
             - 223134912000*r16[i]
             - 6289809926400*r18[i]
             - 169058189664000*r20[i]
             - 4419351149875200*r22[i]
             - 113605204648320000*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
             - 461260800*r14[i]
             - 7904856960*r16[i]
             - 131254905600*r18[i]
             - 2143293425280*r20[i]
             - 34682510016000*r22[i]
             - 558432020361600*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
             - 9369360*r14[i]
             - 85307040*r16[i]
             - 771695280*r18[i]
             - 6960985920*r20[i]
             - 62711787600*r22[i]
             - 564657746400*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
             - 32760*r14[i]
             - 131064*r16[i]
             - 524280*r18[i]
             - 2097144*r20[i]
             - 8388600*r22[i]
             - 33554424*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
             - 2*r14[i]
             - 2*r16[i]
             - 2*r18[i]
             - 2*r20[i]
             - 2*r22[i]
             - 2*r24[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r23 = [(-208012*r[1][i] 
             + 326876*r[2][i]
             - 326876*r[3][i]
             + 245157*r[4][i]
             - 144210*r[5][i]
             + 67298*r[6][i]
             - 24794*r[7][i]
             + 7084*r[8][i]
             - 1518*r[9][i]
             + 230*r[10][i]
             - 22*r[11][i]
             + r[12][i]
             + 58786*r0[i]
             - 33592*r2[i]
             + 83096*r4[i]
             - 479752*r6[i]
             + 5299736*r8[i]
             - 101549512*r10[i]
             + 3208453976*r12[i]
             - 164071220872*r14[i]
             + 13743680753816*r16[i]
             - 1993276972245832*r18[i]
             + 581947914140407256*r20[i]
             - 603916464771468176392*r22[i]
             - 730803773459954540777704*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(25852016738884976640000)
                denom_inv = inverse_mod(odd_part, m)
                r23[i] = (r23[i] * denom_inv) % m
                r23[i] = r23[i] // even_part

            r21 = [(58786*r[1][i] 
             - 90440*r[2][i]
             + 87210*r[3][i]
             - 62016*r[4][i]
             + 33915*r[5][i]
             - 14364*r[6][i]
             + 4655*r[7][i]
             - 1120*r[8][i]
             + 189*r[9][i]
             - 20*r[10][i]
             + r[11][i]
             - 16796*r0[i]
             + 9724*r2[i]
             - 24596*r4[i]
             + 147004*r6[i]
             - 1708916*r8[i]
             + 35240284*r10[i]
             - 1237329236*r12[i]
             + 73853629564*r14[i]
             - 7850527669556*r16[i]
             + 1717351379730844*r18[i]
             - 1359124435588313876*r20[i]
             - 1272410676942417239876*r22[i]
             - 25852016738884976640000*r23[i]
             - 462292539259962003646196*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(51090942171709440000)
                denom_inv = inverse_mod(odd_part, m)
                r21[i] = (r21[i] * denom_inv) % m
                r21[i] = r21[i] // even_part

            r19 = [(-16796*r[1][i] 
             + 25194*r[2][i]
             - 23256*r[3][i]
             + 15504*r[4][i]
             - 7752*r[5][i]
             + 2907*r[6][i]
             - 798*r[7][i]
             + 152*r[8][i]
             - 18*r[9][i]
             + r[10][i]
             + 4862*r0[i]
             - 2860*r2[i]
             + 7436*r4[i]
             - 46420*r6[i]
             + 576236*r8[i]
             - 13098580*r10[i]
             + 532310636*r12[i]
             - 39968611540*r14[i]
             + 6350631494636*r16[i]
             - 3730771315561300*r18[i]
             - 2637991952943407764*r20[i]
             - 46833363657400320000*r21[i]
             - 734121065118879803860*r22[i]
             - 10556240168378032128000*r23[i]
             - 142438684135271315212564*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(121645100408832000)
                denom_inv = inverse_mod(odd_part, m)
                r19[i] = (r19[i] * denom_inv) % m
                r19[i] = r19[i] // even_part

            r17 = [(4862*r[1][i] 
             - 7072*r[2][i]
             + 6188*r[3][i]
             - 3808*r[4][i]
             + 1700*r[5][i]
             - 544*r[6][i]
             + 119*r[7][i]
             - 16*r[8][i]
             + r[9][i]
             - 1430*r0[i]
             + 858*r2[i]
             - 2310*r4[i]
             + 15258*r6[i]
             - 206790*r8[i]
             + 5386458*r10[i]
             - 272513670*r12[i]
             + 30255826458*r14[i]
             - 12765597850950*r16[i]
             - 6622557957272742*r18[i]
             - 101370917007360000*r19[i]
             - 1375210145685786630*r20[i]
             - 17172233341046784000*r21[i]
             - 201832098313986359142*r22[i]
             - 2265471043586150400000*r23[i]
             - 24529324224160803328710*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(355687428096000)
                denom_inv = inverse_mod(odd_part, m)
                r17[i] = (r17[i] * denom_inv) % m
                r17[i] = r17[i] // even_part

            r15 = [(-1430*r[1][i] 
             + 2002*r[2][i]
             - 1638*r[3][i]
             + 910*r[4][i]
             - 350*r[5][i]
             + 90*r[6][i]
             - 14*r[7][i]
             + r[8][i]
             + 429*r0[i]
             - 264*r2[i]
             + 744*r4[i]
             - 5304*r6[i]
             + 81384*r8[i]
             - 2605944*r10[i]
             + 192387624*r12[i]
             - 55942352184*r14[i]
             - 20546119600536*r16[i]
             - 266765571072000*r17[i]
             - 3083760849804024*r18[i]
             - 32945548027392000*r19[i]
             - 332500281299403096*r20[i]
             - 3215147584416768000*r21[i]
             - 30076927429146721464*r22[i]
             - 274100623905588480000*r23[i]
             - 2446077617962088140056*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(1307674368000)
                denom_inv = inverse_mod(odd_part, m)
                r15[i] = (r15[i] * denom_inv) % m
                r15[i] = r15[i] // even_part

            r13 = [(429*r[1][i] 
             - 572*r[2][i]
             + 429*r[3][i]
             - 208*r[4][i]
             + 65*r[5][i]
             - 12*r[6][i]
             + r[7][i]
             - 132*r0[i]
             + 84*r2[i]
             - 252*r4[i]
             + 2004*r6[i]
             - 37212*r8[i]
             + 1710324*r10[i]
             - 325024572*r12[i]
             - 80789566956*r14[i]
             - 871782912000*r15[i]
             - 8422900930332*r16[i]
             - 75583578470400*r17[i]
             - 643521842437836*r18[i]
             - 5269678622208000*r19[i]
             - 41890044885642492*r20[i]
             - 325386564299596800*r21[i]
             - 2481686964269990316*r22[i]
             - 18652248729354240000*r23[i]
             - 138536531588626169052*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6227020800)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
             - 3632428800*r13[i]
             - 28616744548*r14[i]
             - 210680870400*r15[i]
             - 1479485236228*r16[i]
             - 10038995366400*r17[i]
             - 66394067988388*r18[i]
             - 430591742380800*r19[i]
             - 2750479262009668*r20[i]
             - 17360942812012800*r21[i]
             - 108550450893568228*r22[i]
             - 673606803881088000*r23[i]
             - 4154688725062207108*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
             - 726485760*r13[i]
             - 4084385750*r14[i]
             - 22313491200*r15[i]
             - 119387268038*r16[i]
             - 628980992640*r17[i]
             - 3275389222070*r18[i]
             - 16905818966400*r19[i]
             - 86665431465638*r20[i]
             - 441935114987520*r21[i]
             - 2244295389943190*r22[i]
             - 11360520464832000*r23[i]
             - 57360469753884038*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
             - 57657600*r13[i]
             - 239967004*r14[i]
             - 988107120*r15[i]
             - 4037604460*r16[i]
             - 16406863200*r17[i]
             - 66398623804*r18[i]
             - 267911678160*r19[i]
             - 1078605601420*r20[i]
             - 4335313752000*r21[i]
             - 17403958407004*r22[i]
             - 69804002545200*r23[i]
             - 279780634372780*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
             - 1561560*r13[i]
             - 4717438*r14[i]
             - 14217840*r15[i]
             - 42784582*r16[i]
             - 128615880*r17[i]
             - 386371918*r18[i]
             - 1160164320*r19[i]
             - 3482590102*r20[i]
             - 10451964600*r21[i]
             - 31364282398*r22[i]
             - 94109624400*r23[i]
             - 282362427622*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
             - 8190*r13[i]
             - 16382*r14[i]
             - 32766*r15[i]
             - 65534*r16[i]
             - 131070*r17[i]
             - 262142*r18[i]
             - 524286*r19[i]
             - 1048574*r20[i]
             - 2097150*r21[i]
             - 4194302*r22[i]
             - 8388606*r23[i]
             - 16777214*r24[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            - r13[i]
            - r14[i]
            - r15[i]
            - r16[i]
            - r17[i]
            - r18[i]
            - r19[i]
            - r20[i]
            - r21[i]
            - r22[i]
            - r23[i]
            - r24[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24)
        
        if n == 14:
            r0 = r[0]

            r26 = r['infinity']

            L = len(r0)

            r24 = [(-2496144*(r[1][i] + r[-1][i])
             + 1961256*(r[2][i] + r[-2][i])
             - 1307504*(r[3][i] + r[-3][i])
             + 735471*(r[4][i] + r[-4][i])
             - 346104*(r[5][i] + r[-5][i])
             + 134596*(r[6][i] + r[-6][i])
             - 42504*(r[7][i] + r[-7][i])
             + 10626*(r[8][i] + r[-8][i])
             - 2024*(r[9][i] + r[-9][i])
             + 276*(r[10][i] + r[-10][i])
             - 24*(r[11][i] + r[-11][i])
             + (r[12][i] + r[-12][i])
             + 2704156*r0[i]
             - 403291461126605635584000000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(620448401733239439360000)
                denom_inv = inverse_mod(odd_part, m)
                r24[i] = (r24[i] * denom_inv) % m
                r24[i] = r24[i] // even_part

            r22 = [(646646*(r[1][i] + r[-1][i])
             - 497420*(r[2][i] + r[-2][i])
             + 319770*(r[3][i] + r[-3][i])
             - 170544*(r[4][i] + r[-4][i])
             + 74613*(r[5][i] + r[-5][i])
             - 26334*(r[6][i] + r[-6][i])
             + 7315*(r[7][i] + r[-7][i])
             - 1540*(r[8][i] + r[-8][i])
             + 231*(r[9][i] + r[-9][i])
             - 22*(r[10][i] + r[-10][i])
             + (r[11][i] + r[-11][i])
             - 705432*r0[i]
             - 568744368255469486080000*r24[i]
             - 166357727714724824678400000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(1124000727777607680000)
                denom_inv = inverse_mod(odd_part, m)
                r22[i] = (r22[i] * denom_inv) % m
                r22[i] = r22[i] // even_part

            r20 = [(-167960*(r[1][i] + r[-1][i])
             + 125970*(r[2][i] + r[-2][i])
             - 77520*(r[3][i] + r[-3][i])
             + 38760*(r[4][i] + r[-4][i])
             - 15504*(r[5][i] + r[-5][i])
             + 4845*(r[6][i] + r[-6][i])
             - 1140*(r[7][i] + r[-7][i])
             + 190*(r[8][i] + r[-8][i])
             - 20*(r[9][i] + r[-9][i])
             + (r[10][i] + r[-10][i])
             + 184756*r0[i]
             - 936667273148006400000*r22[i]
             - 211124803367560642560000*r24[i]
             - 36608302274885332992000000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2432902008176640000)
                denom_inv = inverse_mod(odd_part, m)
                r20[i] = (r20[i] * denom_inv) % m
                r20[i] = r20[i] // even_part

            r18 = [(43758*(r[1][i] + r[-1][i])
             - 31824*(r[2][i] + r[-2][i])
             + 18564*(r[3][i] + r[-3][i])
             - 8568*(r[4][i] + r[-4][i])
             + 3060*(r[5][i] + r[-5][i])
             - 816*(r[6][i] + r[-6][i])
             + 153*(r[7][i] + r[-7][i])
             - 18*(r[8][i] + r[-8][i])
             + (r[9][i] + r[-9][i])
             - 48620*r0[i]
             - 1824676506132480000*r20[i]
             - 309100200138842112000*r22[i]
             - 40778478784550707200000*r24[i]
             - 4645053436190368481280000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6402373705728000)
                denom_inv = inverse_mod(odd_part, m)
                r18[i] = (r18[i] * denom_inv) % m
                r18[i] = r18[i] // even_part

            r16 = [(-11440*(r[1][i] + r[-1][i])
             + 8008*(r[2][i] + r[-2][i])
             - 4368*(r[3][i] + r[-3][i])
             + 1820*(r[4][i] + r[-4][i])
             - 560*(r[5][i] + r[-5][i])
             + 120*(r[6][i] + r[-6][i])
             - 16*(r[7][i] + r[-7][i])
             + (r[8][i] + r[-8][i])
             + 12870*r0[i]
             - 4268249137152000*r18[i]
             - 527128768438272000*r20[i]
             - 51442361350668288000*r22[i]
             - 4385609982489415680000*r24[i]
             - 343350594609952849920000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20922789888000)
                denom_inv = inverse_mod(odd_part, m)
                r16[i] = (r16[i] * denom_inv) % m
                r16[i] = r16[i] // even_part

            r14 = [(3003*(r[1][i] + r[-1][i])
             - 2002*(r[2][i] + r[-2][i])
             + 1001*(r[3][i] + r[-3][i])
             - 364*(r[4][i] + r[-4][i])
             + 91*(r[5][i] + r[-5][i])
             - 14*(r[6][i] + r[-6][i])
             + (r[7][i] + r[-7][i])
             - 3432*r0[i]
             - 12204960768000*r16[i]
             - 1058170098585600*r18[i]
             - 73775500710912000*r20[i]
             - 4555411900194355200*r22[i]
             - 261131482210959360000*r24[i]
             - 14266599888013304832000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(87178291200)
                denom_inv = inverse_mod(odd_part, m)
                r14[i] = (r14[i] * denom_inv) % m
                r14[i] = r14[i] // even_part

            r12 = [(-792*(r[1][i] + r[-1][i])
             + 495*(r[2][i] + r[-2][i])
             - 220*(r[3][i] + r[-3][i])
             + 66*(r[4][i] + r[-4][i])
             - 12*(r[5][i] + r[-5][i])
             + (r[6][i] + r[-6][i])
             + 924*r0[i]
             - 43589145600*r14[i]
             - 2528170444800*r16[i]
             - 120467944396800*r18[i]
             - 5167100908569600*r20[i]
             - 208331313744153600*r22[i]
             - 8083281646573056000*r24[i]
             - 305994026290208256000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(479001600)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
             - 7264857600*r14[i]
             - 223134912000*r16[i]
             - 6289809926400*r18[i]
             - 169058189664000*r20[i]
             - 4419351149875200*r22[i]
             - 113605204648320000*r24[i]
             - 2890388998040544000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
             - 461260800*r14[i]
             - 7904856960*r16[i]
             - 131254905600*r18[i]
             - 2143293425280*r20[i]
             - 34682510016000*r22[i]
             - 558432020361600*r24[i]
             - 8966533159584000*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
             - 9369360*r14[i]
             - 85307040*r16[i]
             - 771695280*r18[i]
             - 6960985920*r20[i]
             - 62711787600*r22[i]
             - 564657746400*r24[i]
             - 5082926350320*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
             - 32760*r14[i]
             - 131064*r16[i]
             - 524280*r18[i]
             - 2097144*r20[i]
             - 8388600*r22[i]
             - 33554424*r24[i]
             - 134217720*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
             - 2*r14[i]
             - 2*r16[i]
             - 2*r18[i]
             - 2*r20[i]
             - 2*r22[i]
             - 2*r24[i]
             - 2*r26[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r25 = [(742900*r[1][i] 
             - 1188640*r[2][i]
             + 1225785*r[3][i]
             - 961400*r[4][i]
             + 600875*r[5][i]
             - 303600*r[6][i]
             + 123970*r[7][i]
             - 40480*r[8][i]
             + 10350*r[9][i]
             - 2000*r[10][i]
             + 275*r[11][i]
             - 24*r[12][i]
             + r[13][i]
             - 208012*r0[i]
             + 117572*r2[i]
             - 285532*r4[i]
             + 1602692*r6[i]
             - 16996252*r8[i]
             + 307475012*r10[i]
             - 8966430172*r12[i]
             + 409745686532*r14[i]
             - 29195711499292*r16[i]
             + 3312133208909252*r18[i]
             - 636215671041835612*r20[i]
             + 241648300078174135172*r22[i]
             - 321511316149669476991132*r24[i]
             - 492817676505266866078123708*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(15511210043330985984000000)
                denom_inv = inverse_mod(odd_part, m)
                r25[i] = (r25[i] * denom_inv) % m
                r25[i] = r25[i] // even_part

            r23 = [(-208012*r[1][i] 
             + 326876*r[2][i]
             - 326876*r[3][i]
             + 245157*r[4][i]
             - 144210*r[5][i]
             + 67298*r[6][i]
             - 24794*r[7][i]
             + 7084*r[8][i]
             - 1518*r[9][i]
             + 230*r[10][i]
             - 22*r[11][i]
             + r[12][i]
             + 58786*r0[i]
             - 33592*r2[i]
             + 83096*r4[i]
             - 479752*r6[i]
             + 5299736*r8[i]
             - 101549512*r10[i]
             + 3208453976*r12[i]
             - 164071220872*r14[i]
             + 13743680753816*r16[i]
             - 1993276972245832*r18[i]
             + 581947914140407256*r20[i]
             - 603916464771468176392*r22[i]
             - 730803773459954540777704*r24[i]
             - 16803810880275234816000000*r25[i]
             - 339155768243774227716964552*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(25852016738884976640000)
                denom_inv = inverse_mod(odd_part, m)
                r23[i] = (r23[i] * denom_inv) % m
                r23[i] = r23[i] // even_part

            r21 = [(58786*r[1][i] 
             - 90440*r[2][i]
             + 87210*r[3][i]
             - 62016*r[4][i]
             + 33915*r[5][i]
             - 14364*r[6][i]
             + 4655*r[7][i]
             - 1120*r[8][i]
             + 189*r[9][i]
             - 20*r[10][i]
             + r[11][i]
             - 16796*r0[i]
             + 9724*r2[i]
             - 24596*r4[i]
             + 147004*r6[i]
             - 1708916*r8[i]
             + 35240284*r10[i]
             - 1237329236*r12[i]
             + 73853629564*r14[i]
             - 7850527669556*r16[i]
             + 1717351379730844*r18[i]
             - 1359124435588313876*r20[i]
             - 1272410676942417239876*r22[i]
             - 25852016738884976640000*r23[i]
             - 462292539259962003646196*r24[i]
             - 7561714896123855667200000*r25[i]
             - 115761644587269354830466596*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(51090942171709440000)
                denom_inv = inverse_mod(odd_part, m)
                r21[i] = (r21[i] * denom_inv) % m
                r21[i] = r21[i] // even_part

            r19 = [(-16796*r[1][i] 
             + 25194*r[2][i]
             - 23256*r[3][i]
             + 15504*r[4][i]
             - 7752*r[5][i]
             + 2907*r[6][i]
             - 798*r[7][i]
             + 152*r[8][i]
             - 18*r[9][i]
             + r[10][i]
             + 4862*r0[i]
             - 2860*r2[i]
             + 7436*r4[i]
             - 46420*r6[i]
             + 576236*r8[i]
             - 13098580*r10[i]
             + 532310636*r12[i]
             - 39968611540*r14[i]
             + 6350631494636*r16[i]
             - 3730771315561300*r18[i]
             - 2637991952943407764*r20[i]
             - 46833363657400320000*r21[i]
             - 734121065118879803860*r22[i]
             - 10556240168378032128000*r23[i]
             - 142438684135271315212564*r24[i]
             - 1830415113744266649600000*r25[i]
             - 22632897298190126259675220*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(121645100408832000)
                denom_inv = inverse_mod(odd_part, m)
                r19[i] = (r19[i] * denom_inv) % m
                r19[i] = r19[i] // even_part

            r17 = [(4862*r[1][i] 
             - 7072*r[2][i]
             + 6188*r[3][i]
             - 3808*r[4][i]
             + 1700*r[5][i]
             - 544*r[6][i]
             + 119*r[7][i]
             - 16*r[8][i]
             + r[9][i]
             - 1430*r0[i]
             + 858*r2[i]
             - 2310*r4[i]
             + 15258*r6[i]
             - 206790*r8[i]
             + 5386458*r10[i]
             - 272513670*r12[i]
             + 30255826458*r14[i]
             - 12765597850950*r16[i]
             - 6622557957272742*r18[i]
             - 101370917007360000*r19[i]
             - 1375210145685786630*r20[i]
             - 17172233341046784000*r21[i]
             - 201832098313986359142*r22[i]
             - 2265471043586150400000*r23[i]
             - 24529324224160803328710*r24[i]
             - 258058524232798248960000*r25[i]
             - 2652208374242713043720742*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(355687428096000)
                denom_inv = inverse_mod(odd_part, m)
                r17[i] = (r17[i] * denom_inv) % m
                r17[i] = r17[i] // even_part

            r15 = [(-1430*r[1][i] 
             + 2002*r[2][i]
             - 1638*r[3][i]
             + 910*r[4][i]
             - 350*r[5][i]
             + 90*r[6][i]
             - 14*r[7][i]
             + r[8][i]
             + 429*r0[i]
             - 264*r2[i]
             + 744*r4[i]
             - 5304*r6[i]
             + 81384*r8[i]
             - 2605944*r10[i]
             + 192387624*r12[i]
             - 55942352184*r14[i]
             - 20546119600536*r16[i]
             - 266765571072000*r17[i]
             - 3083760849804024*r18[i]
             - 32945548027392000*r19[i]
             - 332500281299403096*r20[i]
             - 3215147584416768000*r21[i]
             - 30076927429146721464*r22[i]
             - 274100623905588480000*r23[i]
             - 2446077617962088140056*r24[i]
             - 21459412163122053120000*r25[i]
             - 185641639183185136464504*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(1307674368000)
                denom_inv = inverse_mod(odd_part, m)
                r15[i] = (r15[i] * denom_inv) % m
                r15[i] = r15[i] // even_part

            r13 = [(429*r[1][i] 
             - 572*r[2][i]
             + 429*r[3][i]
             - 208*r[4][i]
             + 65*r[5][i]
             - 12*r[6][i]
             + r[7][i]
             - 132*r0[i]
             + 84*r2[i]
             - 252*r4[i]
             + 2004*r6[i]
             - 37212*r8[i]
             + 1710324*r10[i]
             - 325024572*r12[i]
             - 80789566956*r14[i]
             - 871782912000*r15[i]
             - 8422900930332*r16[i]
             - 75583578470400*r17[i]
             - 643521842437836*r18[i]
             - 5269678622208000*r19[i]
             - 41890044885642492*r20[i]
             - 325386564299596800*r21[i]
             - 2481686964269990316*r22[i]
             - 18652248729354240000*r23[i]
             - 138536531588626169052*r24[i]
             - 1019042849143807488000*r25[i]
             - 7436421488952386592396*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6227020800)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
             - 3632428800*r13[i]
             - 28616744548*r14[i]
             - 210680870400*r15[i]
             - 1479485236228*r16[i]
             - 10038995366400*r17[i]
             - 66394067988388*r18[i]
             - 430591742380800*r19[i]
             - 2750479262009668*r20[i]
             - 17360942812012800*r21[i]
             - 108550450893568228*r22[i]
             - 673606803881088000*r23[i]
             - 4154688725062207108*r24[i]
             - 25499502190850688000*r25[i]
             - 155878445775166700068*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
             - 726485760*r13[i]
             - 4084385750*r14[i]
             - 22313491200*r15[i]
             - 119387268038*r16[i]
             - 628980992640*r17[i]
             - 3275389222070*r18[i]
             - 16905818966400*r19[i]
             - 86665431465638*r20[i]
             - 441935114987520*r21[i]
             - 2244295389943190*r22[i]
             - 11360520464832000*r23[i]
             - 57360469753884038*r24[i]
             - 289038899804054400*r25[i]
             - 1454155949521941110*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
             - 57657600*r13[i]
             - 239967004*r14[i]
             - 988107120*r15[i]
             - 4037604460*r16[i]
             - 16406863200*r17[i]
             - 66398623804*r18[i]
             - 267911678160*r19[i]
             - 1078605601420*r20[i]
             - 4335313752000*r21[i]
             - 17403958407004*r22[i]
             - 69804002545200*r23[i]
             - 279780634372780*r24[i]
             - 1120816644948000*r25[i]
             - 4488349371924604*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
             - 1561560*r13[i]
             - 4717438*r14[i]
             - 14217840*r15[i]
             - 42784582*r16[i]
             - 128615880*r17[i]
             - 386371918*r18[i]
             - 1160164320*r19[i]
             - 3482590102*r20[i]
             - 10451964600*r21[i]
             - 31364282398*r22[i]
             - 94109624400*r23[i]
             - 282362427622*r24[i]
             - 847154391720*r25[i]
             - 2541597392878*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
             - 8190*r13[i]
             - 16382*r14[i]
             - 32766*r15[i]
             - 65534*r16[i]
             - 131070*r17[i]
             - 262142*r18[i]
             - 524286*r19[i]
             - 1048574*r20[i]
             - 2097150*r21[i]
             - 4194302*r22[i]
             - 8388606*r23[i]
             - 16777214*r24[i]
             - 33554430*r25[i]
             - 67108862*r26[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            - r13[i]
            - r14[i]
            - r15[i]
            - r16[i]
            - r17[i]
            - r18[i]
            - r19[i]
            - r20[i]
            - r21[i]
            - r22[i]
            - r23[i]
            - r24[i]
            - r25[i]
            - r26[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26)
        
        if n == 15:
            r0 = r[0]

            r28 = r['infinity']

            L = len(r0)

            r26 = [(9657700*(r[1][i] + r[-1][i])
             - 7726160*(r[2][i] + r[-2][i])
             + 5311735*(r[3][i] + r[-3][i])
             - 3124550*(r[4][i] + r[-4][i])
             + 1562275*(r[5][i] + r[-5][i])
             - 657800*(r[6][i] + r[-6][i])
             + 230230*(r[7][i] + r[-7][i])
             - 65780*(r[8][i] + r[-8][i])
             + 14950*(r[9][i] + r[-9][i])
             - 2600*(r[10][i] + r[-10][i])
             + 325*(r[11][i] + r[-11][i])
             - 26*(r[12][i] + r[-12][i])
             + (r[13][i] + r[-13][i])
             - 10400600*r0[i]
             - 330295706662690015543296000000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(403291461126605635584000000)
                denom_inv = inverse_mod(odd_part, m)
                r26[i] = (r26[i] * denom_inv) % m
                r26[i] = r26[i] // even_part

            r24 = [(-2496144*(r[1][i] + r[-1][i])
             + 1961256*(r[2][i] + r[-2][i])
             - 1307504*(r[3][i] + r[-3][i])
             + 735471*(r[4][i] + r[-4][i])
             - 346104*(r[5][i] + r[-5][i])
             + 134596*(r[6][i] + r[-6][i])
             - 42504*(r[7][i] + r[-7][i])
             + 10626*(r[8][i] + r[-8][i])
             - 2024*(r[9][i] + r[-9][i])
             + 276*(r[10][i] + r[-10][i])
             - 24*(r[11][i] + r[-11][i])
             + (r[12][i] + r[-12][i])
             + 2704156*r0[i]
             - 403291461126605635584000000*r26[i]
             - 149903436100759314746572800000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(620448401733239439360000)
                denom_inv = inverse_mod(odd_part, m)
                r24[i] = (r24[i] * denom_inv) % m
                r24[i] = r24[i] // even_part

            r22 = [(646646*(r[1][i] + r[-1][i])
             - 497420*(r[2][i] + r[-2][i])
             + 319770*(r[3][i] + r[-3][i])
             - 170544*(r[4][i] + r[-4][i])
             + 74613*(r[5][i] + r[-5][i])
             - 26334*(r[6][i] + r[-6][i])
             + 7315*(r[7][i] + r[-7][i])
             - 1540*(r[8][i] + r[-8][i])
             + 231*(r[9][i] + r[-9][i])
             - 22*(r[10][i] + r[-10][i])
             + (r[11][i] + r[-11][i])
             - 705432*r0[i]
             - 568744368255469486080000*r24[i]
             - 166357727714724824678400000*r26[i]
             - 37042320704478727628390400000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(1124000727777607680000)
                denom_inv = inverse_mod(odd_part, m)
                r22[i] = (r22[i] * denom_inv) % m
                r22[i] = r22[i] // even_part

            r20 = [(-167960*(r[1][i] + r[-1][i])
             + 125970*(r[2][i] + r[-2][i])
             - 77520*(r[3][i] + r[-3][i])
             + 38760*(r[4][i] + r[-4][i])
             - 15504*(r[5][i] + r[-5][i])
             + 4845*(r[6][i] + r[-6][i])
             - 1140*(r[7][i] + r[-7][i])
             + 190*(r[8][i] + r[-8][i])
             - 20*(r[9][i] + r[-9][i])
             + (r[10][i] + r[-10][i])
             + 184756*r0[i]
             - 936667273148006400000*r22[i]
             - 211124803367560642560000*r24[i]
             - 36608302274885332992000000*r26[i]
             - 5425950533240873322086400000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2432902008176640000)
                denom_inv = inverse_mod(odd_part, m)
                r20[i] = (r20[i] * denom_inv) % m
                r20[i] = r20[i] // even_part

            r18 = [(43758*(r[1][i] + r[-1][i])
             - 31824*(r[2][i] + r[-2][i])
             + 18564*(r[3][i] + r[-3][i])
             - 8568*(r[4][i] + r[-4][i])
             + 3060*(r[5][i] + r[-5][i])
             - 816*(r[6][i] + r[-6][i])
             + 153*(r[7][i] + r[-7][i])
             - 18*(r[8][i] + r[-8][i])
             + (r[9][i] + r[-9][i])
             - 48620*r0[i]
             - 1824676506132480000*r20[i]
             - 309100200138842112000*r22[i]
             - 40778478784550707200000*r24[i]
             - 4645053436190368481280000*r26[i]
             - 481314610282065419059200000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6402373705728000)
                denom_inv = inverse_mod(odd_part, m)
                r18[i] = (r18[i] * denom_inv) % m
                r18[i] = r18[i] // even_part

            r16 = [(-11440*(r[1][i] + r[-1][i])
             + 8008*(r[2][i] + r[-2][i])
             - 4368*(r[3][i] + r[-3][i])
             + 1820*(r[4][i] + r[-4][i])
             - 560*(r[5][i] + r[-5][i])
             + 120*(r[6][i] + r[-6][i])
             - 16*(r[7][i] + r[-7][i])
             + (r[8][i] + r[-8][i])
             + 12870*r0[i]
             - 4268249137152000*r18[i]
             - 527128768438272000*r20[i]
             - 51442361350668288000*r22[i]
             - 4385609982489415680000*r24[i]
             - 343350594609952849920000*r26[i]
             - 25398422028160175554560000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20922789888000)
                denom_inv = inverse_mod(odd_part, m)
                r16[i] = (r16[i] * denom_inv) % m
                r16[i] = r16[i] // even_part

            r14 = [(3003*(r[1][i] + r[-1][i])
             - 2002*(r[2][i] + r[-2][i])
             + 1001*(r[3][i] + r[-3][i])
             - 364*(r[4][i] + r[-4][i])
             + 91*(r[5][i] + r[-5][i])
             - 14*(r[6][i] + r[-6][i])
             + (r[7][i] + r[-7][i])
             - 3432*r0[i]
             - 12204960768000*r16[i]
             - 1058170098585600*r18[i]
             - 73775500710912000*r20[i]
             - 4555411900194355200*r22[i]
             - 261131482210959360000*r24[i]
             - 14266599888013304832000*r26[i]
             - 754754307297469839360000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(87178291200)
                denom_inv = inverse_mod(odd_part, m)
                r14[i] = (r14[i] * denom_inv) % m
                r14[i] = r14[i] // even_part

            r12 = [(-792*(r[1][i] + r[-1][i])
             + 495*(r[2][i] + r[-2][i])
             - 220*(r[3][i] + r[-3][i])
             + 66*(r[4][i] + r[-4][i])
             - 12*(r[5][i] + r[-5][i])
             + (r[6][i] + r[-6][i])
             + 924*r0[i]
             - 43589145600*r14[i]
             - 2528170444800*r16[i]
             - 120467944396800*r18[i]
             - 5167100908569600*r20[i]
             - 208331313744153600*r22[i]
             - 8083281646573056000*r24[i]
             - 305994026290208256000*r26[i]
             - 11397316294188849024000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(479001600)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part

            r10 = [(210*(r[1][i] + r[-1][i])
             - 120*(r[2][i] + r[-2][i])
             + 45*(r[3][i] + r[-3][i])
             - 10*(r[4][i] + r[-4][i])
             + (r[5][i] + r[-5][i])
             - 252*r0[i]
             - 199584000*r12[i]
             - 7264857600*r14[i]
             - 223134912000*r16[i]
             - 6289809926400*r18[i]
             - 169058189664000*r20[i]
             - 4419351149875200*r22[i]
             - 113605204648320000*r24[i]
             - 2890388998040544000*r26[i]
             - 73066712935376160000*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3628800)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part

            r8 = [(-56*(r[1][i] + r[-1][i])
             + 28*(r[2][i] + r[-2][i])
             - 8*(r[3][i] + r[-3][i])
             + (r[4][i] + r[-4][i])
             + 70*r0[i]
             - 1209600*r10[i]
             - 25280640*r12[i]
             - 461260800*r14[i]
             - 7904856960*r16[i]
             - 131254905600*r18[i]
             - 2143293425280*r20[i]
             - 34682510016000*r22[i]
             - 558432020361600*r24[i]
             - 8966533159584000*r26[i]
             - 143749174428961920*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40320)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part

            r6 = [(15*(r[1][i] + r[-1][i])
             - 6*(r[2][i] + r[-2][i])
             + (r[3][i] + r[-3][i])
             - 20*r0[i]
             - 10080*r8[i]
             - 105840*r10[i]
             - 1013760*r12[i]
             - 9369360*r14[i]
             - 85307040*r16[i]
             - 771695280*r18[i]
             - 6960985920*r20[i]
             - 62711787600*r22[i]
             - 564657746400*r24[i]
             - 5082926350320*r26[i]
             - 45750363684480*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(720)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part

            r4 = [(-4*(r[1][i] + r[-1][i])
             + (r[2][i] + r[-2][i])
             + 6*r0[i]
             - 120*r6[i]
             - 504*r8[i]
             - 2040*r10[i]
             - 8184*r12[i]
             - 32760*r14[i]
             - 131064*r16[i]
             - 524280*r18[i]
             - 2097144*r20[i]
             - 8388600*r22[i]
             - 33554424*r24[i]
             - 134217720*r26[i]
             - 536870904*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                r4[i] = (r4[i] * denom_inv) % m
                r4[i] = r4[i] // even_part

            r2 = [((r[1][i] + r[-1][i])
             - 2*r0[i]
             - 2*r4[i]
             - 2*r6[i]
             - 2*r8[i]
             - 2*r10[i]
             - 2*r12[i]
             - 2*r14[i]
             - 2*r16[i]
             - 2*r18[i]
             - 2*r20[i]
             - 2*r22[i]
             - 2*r24[i]
             - 2*r26[i]
             - 2*r28[i]
            ) % m  for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(2)
                denom_inv = inverse_mod(odd_part, m)
                r2[i] = (r2[i] * denom_inv) % m
                r2[i] = r2[i] // even_part

            r27 = [(-2674440*r[1][i] 
             + 4345965*r[2][i]
             - 4601610*r[3][i]
             + 3749460*r[4][i]
             - 2466750*r[5][i]
             + 1332045*r[6][i]
             - 592020*r[7][i]
             + 215280*r[8][i]
             - 63180*r[9][i]
             + 14625*r[10][i]
             - 2574*r[11][i]
             + 324*r[12][i]
             - 26*r[13][i]
             + r[14][i]
             + 742900*r0[i]
             - 416024*r2[i]
             + 994840*r4[i]
             - 5454824*r6[i]
             + 55944280*r8[i]
             - 966290024*r10[i]
             + 26454613720*r12[i]
             - 1109329691624*r14[i]
             + 70212852387160*r16[i]
             - 6733670404619624*r18[i]
             + 1005318113348848600*r20[i]
             - 249581050854994274024*r22[i]
             + 120719220687299365182040*r24[i]
             - 202039976682357297272094824*r26[i]
             - 385557844336439370252173428520*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(10888869450418352160768000000)
                denom_inv = inverse_mod(odd_part, m)
                r27[i] = (r27[i] * denom_inv) % m
                r27[i] = r27[i] // even_part

            r25 = [(742900*r[1][i] 
             - 1188640*r[2][i]
             + 1225785*r[3][i]
             - 961400*r[4][i]
             + 600875*r[5][i]
             - 303600*r[6][i]
             + 123970*r[7][i]
             - 40480*r[8][i]
             + 10350*r[9][i]
             - 2000*r[10][i]
             + 275*r[11][i]
             - 24*r[12][i]
             + r[13][i]
             - 208012*r0[i]
             + 117572*r2[i]
             - 285532*r4[i]
             + 1602692*r6[i]
             - 16996252*r8[i]
             + 307475012*r10[i]
             - 8966430172*r12[i]
             + 409745686532*r14[i]
             - 29195711499292*r16[i]
             + 3312133208909252*r18[i]
             - 636215671041835612*r20[i]
             + 241648300078174135172*r22[i]
             - 321511316149669476991132*r24[i]
             - 492817676505266866078123708*r26[i]
             - 12703681025488077520896000000*r27[i]
             - 286779648275654636997381637852*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(15511210043330985984000000)
                denom_inv = inverse_mod(odd_part, m)
                r25[i] = (r25[i] * denom_inv) % m
                r25[i] = r25[i] // even_part

            r23 = [(-208012*r[1][i] 
             + 326876*r[2][i]
             - 326876*r[3][i]
             + 245157*r[4][i]
             - 144210*r[5][i]
             + 67298*r[6][i]
             - 24794*r[7][i]
             + 7084*r[8][i]
             - 1518*r[9][i]
             + 230*r[10][i]
             - 22*r[11][i]
             + r[12][i]
             + 58786*r0[i]
             - 33592*r2[i]
             + 83096*r4[i]
             - 479752*r6[i]
             + 5299736*r8[i]
             - 101549512*r10[i]
             + 3208453976*r12[i]
             - 164071220872*r14[i]
             + 13743680753816*r16[i]
             - 1993276972245832*r18[i]
             + 581947914140407256*r20[i]
             - 603916464771468176392*r22[i]
             - 730803773459954540777704*r24[i]
             - 16803810880275234816000000*r25[i]
             - 339155768243774227716964552*r26[i]
             - 6245976504198304781107200000*r27[i]
             - 107413822788261782335458993064*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(25852016738884976640000)
                denom_inv = inverse_mod(odd_part, m)
                r23[i] = (r23[i] * denom_inv) % m
                r23[i] = r23[i] // even_part

            r21 = [(58786*r[1][i] 
             - 90440*r[2][i]
             + 87210*r[3][i]
             - 62016*r[4][i]
             + 33915*r[5][i]
             - 14364*r[6][i]
             + 4655*r[7][i]
             - 1120*r[8][i]
             + 189*r[9][i]
             - 20*r[10][i]
             + r[11][i]
             - 16796*r0[i]
             + 9724*r2[i]
             - 24596*r4[i]
             + 147004*r6[i]
             - 1708916*r8[i]
             + 35240284*r10[i]
             - 1237329236*r12[i]
             + 73853629564*r14[i]
             - 7850527669556*r16[i]
             + 1717351379730844*r18[i]
             - 1359124435588313876*r20[i]
             - 1272410676942417239876*r22[i]
             - 25852016738884976640000*r23[i]
             - 462292539259962003646196*r24[i]
             - 7561714896123855667200000*r25[i]
             - 115761644587269354830466596*r26[i]
             - 1683741850203578528563200000*r27[i]
             - 23512975860299444963550050516*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(51090942171709440000)
                denom_inv = inverse_mod(odd_part, m)
                r21[i] = (r21[i] * denom_inv) % m
                r21[i] = r21[i] // even_part

            r19 = [(-16796*r[1][i] 
             + 25194*r[2][i]
             - 23256*r[3][i]
             + 15504*r[4][i]
             - 7752*r[5][i]
             + 2907*r[6][i]
             - 798*r[7][i]
             + 152*r[8][i]
             - 18*r[9][i]
             + r[10][i]
             + 4862*r0[i]
             - 2860*r2[i]
             + 7436*r4[i]
             - 46420*r6[i]
             + 576236*r8[i]
             - 13098580*r10[i]
             + 532310636*r12[i]
             - 39968611540*r14[i]
             + 6350631494636*r16[i]
             - 3730771315561300*r18[i]
             - 2637991952943407764*r20[i]
             - 46833363657400320000*r21[i]
             - 734121065118879803860*r22[i]
             - 10556240168378032128000*r23[i]
             - 142438684135271315212564*r24[i]
             - 1830415113744266649600000*r25[i]
             - 22632897298190126259675220*r26[i]
             - 271297526662043666104320000*r27[i]
             - 3170344993810020486920015764*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(121645100408832000)
                denom_inv = inverse_mod(odd_part, m)
                r19[i] = (r19[i] * denom_inv) % m
                r19[i] = r19[i] // even_part

            r17 = [(4862*r[1][i] 
             - 7072*r[2][i]
             + 6188*r[3][i]
             - 3808*r[4][i]
             + 1700*r[5][i]
             - 544*r[6][i]
             + 119*r[7][i]
             - 16*r[8][i]
             + r[9][i]
             - 1430*r0[i]
             + 858*r2[i]
             - 2310*r4[i]
             + 15258*r6[i]
             - 206790*r8[i]
             + 5386458*r10[i]
             - 272513670*r12[i]
             + 30255826458*r14[i]
             - 12765597850950*r16[i]
             - 6622557957272742*r18[i]
             - 101370917007360000*r19[i]
             - 1375210145685786630*r20[i]
             - 17172233341046784000*r21[i]
             - 201832098313986359142*r22[i]
             - 2265471043586150400000*r23[i]
             - 24529324224160803328710*r24[i]
             - 258058524232798248960000*r25[i]
             - 2652208374242713043720742*r26[i]
             - 26739700571225856614400000*r27[i]
             - 265323404171486113659725190*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(355687428096000)
                denom_inv = inverse_mod(odd_part, m)
                r17[i] = (r17[i] * denom_inv) % m
                r17[i] = r17[i] // even_part

            r15 = [(-1430*r[1][i] 
             + 2002*r[2][i]
             - 1638*r[3][i]
             + 910*r[4][i]
             - 350*r[5][i]
             + 90*r[6][i]
             - 14*r[7][i]
             + r[8][i]
             + 429*r0[i]
             - 264*r2[i]
             + 744*r4[i]
             - 5304*r6[i]
             + 81384*r8[i]
             - 2605944*r10[i]
             + 192387624*r12[i]
             - 55942352184*r14[i]
             - 20546119600536*r16[i]
             - 266765571072000*r17[i]
             - 3083760849804024*r18[i]
             - 32945548027392000*r19[i]
             - 332500281299403096*r20[i]
             - 3215147584416768000*r21[i]
             - 30076927429146721464*r22[i]
             - 274100623905588480000*r23[i]
             - 2446077617962088140056*r24[i]
             - 21459412163122053120000*r25[i]
             - 185641639183185136464504*r26[i]
             - 1587401376760010972160000*r27[i]
             - 13442713420403849918131416*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(1307674368000)
                denom_inv = inverse_mod(odd_part, m)
                r15[i] = (r15[i] * denom_inv) % m
                r15[i] = r15[i] // even_part

            r13 = [(429*r[1][i] 
             - 572*r[2][i]
             + 429*r[3][i]
             - 208*r[4][i]
             + 65*r[5][i]
             - 12*r[6][i]
             + r[7][i]
             - 132*r0[i]
             + 84*r2[i]
             - 252*r4[i]
             + 2004*r6[i]
             - 37212*r8[i]
             + 1710324*r10[i]
             - 325024572*r12[i]
             - 80789566956*r14[i]
             - 871782912000*r15[i]
             - 8422900930332*r16[i]
             - 75583578470400*r17[i]
             - 643521842437836*r18[i]
             - 5269678622208000*r19[i]
             - 41890044885642492*r20[i]
             - 325386564299596800*r21[i]
             - 2481686964269990316*r22[i]
             - 18652248729354240000*r23[i]
             - 138536531588626169052*r24[i]
             - 1019042849143807488000*r25[i]
             - 7436421488952386592396*r26[i]
             - 53911021949819274240000*r27[i]
             - 388701690499592948238012*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6227020800)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part

            r11 = [(-132*r[1][i] 
             + 165*r[2][i]
             - 110*r[3][i]
             + 44*r[4][i]
             - 10*r[5][i]
             + r[6][i]
             + 42*r0[i]
             - 28*r2[i]
             + 92*r4[i]
             - 868*r6[i]
             + 22652*r8[i]
             - 2620708*r10[i]
             - 415790788*r12[i]
             - 3632428800*r13[i]
             - 28616744548*r14[i]
             - 210680870400*r15[i]
             - 1479485236228*r16[i]
             - 10038995366400*r17[i]
             - 66394067988388*r18[i]
             - 430591742380800*r19[i]
             - 2750479262009668*r20[i]
             - 17360942812012800*r21[i]
             - 108550450893568228*r22[i]
             - 673606803881088000*r23[i]
             - 4154688725062207108*r24[i]
             - 25499502190850688000*r25[i]
             - 155878445775166700068*r26[i]
             - 949776357849070752000*r27[i]
             - 5771581202353414724548*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(39916800)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part

            r9 = [(42*r[1][i] 
             - 48*r[2][i]
             + 27*r[3][i]
             - 8*r[4][i]
             + r[5][i]
             - 14*r0[i]
             + 10*r2[i]
             - 38*r4[i]
             + 490*r6[i]
             - 31238*r8[i]
             - 2922230*r10[i]
             - 19958400*r11[i]
             - 124075238*r12[i]
             - 726485760*r13[i]
             - 4084385750*r14[i]
             - 22313491200*r15[i]
             - 119387268038*r16[i]
             - 628980992640*r17[i]
             - 3275389222070*r18[i]
             - 16905818966400*r19[i]
             - 86665431465638*r20[i]
             - 441935114987520*r21[i]
             - 2244295389943190*r22[i]
             - 11360520464832000*r23[i]
             - 57360469753884038*r24[i]
             - 289038899804054400*r25[i]
             - 1454155949521941110*r26[i]
             - 7306671293537616000*r27[i]
             - 36677059892827099238*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(362880)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part

            r7 = [(-14*r[1][i] 
             + 14*r[2][i]
             - 6*r[3][i]
             + r[4][i]
             + 5*r0[i]
             - 4*r2[i]
             + 20*r4[i]
             - 604*r6[i]
             - 29740*r8[i]
             - 151200*r9[i]
             - 708604*r10[i]
             - 3160080*r11[i]
             - 13645900*r12[i]
             - 57657600*r13[i]
             - 239967004*r14[i]
             - 988107120*r15[i]
             - 4037604460*r16[i]
             - 16406863200*r17[i]
             - 66398623804*r18[i]
             - 267911678160*r19[i]
             - 1078605601420*r20[i]
             - 4335313752000*r21[i]
             - 17403958407004*r22[i]
             - 69804002545200*r23[i]
             - 279780634372780*r24[i]
             - 1120816644948000*r25[i]
             - 4488349371924604*r26[i]
             - 17968646803620240*r27[i]
             - 71920337041294540*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(5040)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part

            r5 = [(5*r[1][i] 
             - 4*r[2][i]
             + r[3][i]
             - 2*r0[i]
             + 2*r2[i]
             - 22*r4[i]
             - 478*r6[i]
             - 1680*r7[i]
             - 5542*r8[i]
             - 17640*r9[i]
             - 54958*r10[i]
             - 168960*r11[i]
             - 515062*r12[i]
             - 1561560*r13[i]
             - 4717438*r14[i]
             - 14217840*r15[i]
             - 42784582*r16[i]
             - 128615880*r17[i]
             - 386371918*r18[i]
             - 1160164320*r19[i]
             - 3482590102*r20[i]
             - 10451964600*r21[i]
             - 31364282398*r22[i]
             - 94109624400*r23[i]
             - 282362427622*r24[i]
             - 847154391720*r25[i]
             - 2541597392878*r26[i]
             - 7625060614080*r27[i]
             - 22875718713142*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part

            r3 = [(-2*r[1][i] 
             + r[2][i]
             + 1*r0[i]
             - 2*r2[i]
             - 14*r4[i]
             - 30*r5[i]
             - 62*r6[i]
             - 126*r7[i]
             - 254*r8[i]
             - 510*r9[i]
             - 1022*r10[i]
             - 2046*r11[i]
             - 4094*r12[i]
             - 8190*r13[i]
             - 16382*r14[i]
             - 32766*r15[i]
             - 65534*r16[i]
             - 131070*r17[i]
             - 262142*r18[i]
             - 524286*r19[i]
             - 1048574*r20[i]
             - 2097150*r21[i]
             - 4194302*r22[i]
             - 8388606*r23[i]
             - 16777214*r24[i]
             - 33554430*r25[i]
             - 67108862*r26[i]
             - 134217726*r27[i]
             - 268435454*r28[i]
            ) % m for i in range(L)]

            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                r3[i] = (r3[i] * denom_inv) % m
                r3[i] = r3[i] // even_part

            r1 = [(r[1][i]
             - r0[i]
            - r2[i]
            - r3[i]
            - r4[i]
            - r5[i]
            - r6[i]
            - r7[i]
            - r8[i]
            - r9[i]
            - r10[i]
            - r11[i]
            - r12[i]
            - r13[i]
            - r14[i]
            - r15[i]
            - r16[i]
            - r17[i]
            - r18[i]
            - r19[i]
            - r20[i]
            - r21[i]
            - r22[i]
            - r23[i]
            - r24[i]
            - r25[i]
            - r26[i]
            - r27[i]
            - r28[i]
            ) % m for i in range(L)]


            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28)
        
        
    if formulas == "efficient":
        if n == 4:
            r0 = r[0]
            r6 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r6[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 64*r6[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part

            # the remaining even variables
            r4 = [((r[2][i] + r[-2][i]) % m) // 2 for i in range(L)]
            for i in range(L):
                r4[i] = (r4[i] - r0[i] - 64*r6[i]) % m
                r4[i] = r4[i] // 4
                r4[i] = (r4[i] - E1[i]) % m
                denom_inv = inverse_mod(3, m)
                r4[i] = (r4[i] * denom_inv) % m
            r2 = [(E1[i] - r4[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r0[i] - 9*r2[i] - 81*r4[i] - 729*r6[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part

            # the remaining odd variables
            r5 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                r5[i] = (r5[i] * denom_inv) % m
                r5[i] = r5[i] // even_part
            r3 = [(O2[i] - 5*r5[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6)
        if n == 5:
            r0 = r[0]
            r8 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r8[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 256*r8[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] //  even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 6561*r8[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] //  even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            # the remaining even variables
            r6 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                r6[i] = (r6[i] * denom_inv) % m
                r6[i] = r6[i] // even_part
            r4 = [(E2[i] - 5*r6[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r0[i] - 16*r2[i] - 256*r4[i] - 4096*r6[i] - 65536*r8[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part

            # layer 2
            O5 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part

            # the remaining odd variables
            r7 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                r7[i] = (r7[i] * denom_inv) % m
                r7[i] = r7[i] // even_part
            r5 = [(O5[i] - 14*r7[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8)
        
        if n == 6:
            r0 = r[0]
            r10 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r10[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 1024*r10[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 59049*r10[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 1048576*r10[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part

            # layer 2
            E5 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part

            # the remaining even variables
            r8 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                r8[i] = (r8[i] * denom_inv) % m
                r8[i] = r8[i] // even_part
            r6 = [(E5[i] - 14*r8[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r0[i] - 25*r2[i] - 625*r4[i] - 15625*r6[i] - 390625*r8[i] - 9765625*r10[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part

            # layer 2
            O6 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part

            # layer 3
            O9 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part

            # the remaining odd variables
            r9 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                r9[i] = (r9[i] * denom_inv) % m
                r9[i] = r9[i] // even_part
            r7 = [(O9[i] - 30*r9[i] ) % m for i in range(L)]
            r5 = [(O6[i] - 14*r7[i] - 147*r9[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10)
        
        if n == 7:
            r0 = r[0]
            r12 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r12[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 4096*r12[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 531441*r12[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 16777216*r12[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 244140625*r12[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part

            # layer 2
            E6 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
            E7 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
            E8 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part

            # layer 3
            E9 = [(E7[i] - E6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
            E10 = [(E8[i] - E7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part

            # the remaining even variables
            r10 = [(E10[i] - E9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                r10[i] = (r10[i] * denom_inv) % m
                r10[i] = r10[i] // even_part
            r8 = [(E9[i] - 30*r10[i] ) % m for i in range(L)]
            r6 = [(E6[i] - 14*r8[i] - 147*r10[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r0[i] - 36*r2[i] - 1296*r4[i] - 46656*r6[i] - 1679616*r8[i] - 60466176*r10[i] - 2176782336*r12[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part

            # layer 2
            O7 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
            O9 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part

            # layer 3
            O11 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
            O12 = [(O9[i] - O8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
            O13 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part

            # layer 4
            O14 = [(O12[i] - O11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part
            O15 = [(O13[i] - O12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part

            # the remaining odd variables
            r11 = [(O15[i] - O14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                r11[i] = (r11[i] * denom_inv) % m
                r11[i] = r11[i] // even_part
            r9 = [(O14[i] - 55*r11[i] ) % m for i in range(L)]
            r7 = [(O11[i] - 30*r9[i] - 627*r11[i] ) % m for i in range(L)]
            r5 = [(O7[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12)
        
        if n == 8:
            r0 = r[0]
            r14 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r14[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 16384*r14[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 4782969*r14[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 268435456*r14[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 6103515625*r14[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [((r[6][i] + r[-6][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E6[i] = (E6[i] - r0[i] - 78364164096*r14[i]) % m
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
                E6[i] = (E6[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part

            # layer 2
            E7 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
            E8 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
            E9 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
            E10 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part

            # layer 3
            E11 = [(E8[i] - E7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
            E12 = [(E9[i] - E8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part
            E13 = [(E10[i] - E9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part

            # layer 4
            E14 = [(E12[i] - E11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E14[i] = (E14[i] * denom_inv) % m
                E14[i] = E14[i] // even_part
            E15 = [(E13[i] - E12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                E15[i] = (E15[i] * denom_inv) % m
                E15[i] = E15[i] // even_part

            # the remaining even variables
            r12 = [(E15[i] - E14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                r12[i] = (r12[i] * denom_inv) % m
                r12[i] = r12[i] // even_part
            r10 = [(E14[i] - 55*r12[i] ) % m for i in range(L)]
            r8 = [(E11[i] - 30*r10[i] - 627*r12[i] ) % m for i in range(L)]
            r6 = [(E7[i] - 14*r8[i] - 147*r10[i] - 1408*r12[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] - 341*r12[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] - r12[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r[-6][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(r[7][i] - r0[i] - 49*r2[i] - 2401*r4[i] - 117649*r6[i] - 5764801*r8[i] - 282475249*r10[i] - 13841287201*r12[i] - 678223072849*r14[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
                O7[i] = (O7[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part

            # layer 2
            O8 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
            O9 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
            O11 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
            O12 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part

            # layer 3
            O13 = [(O9[i] - O8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
            O14 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part
            O15 = [(O11[i] - O10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part
            O16 = [(O12[i] - O11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O16[i] = (O16[i] * denom_inv) % m
                O16[i] = O16[i] // even_part

            # layer 4
            O17 = [(O14[i] - O13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O17[i] = (O17[i] * denom_inv) % m
                O17[i] = O17[i] // even_part
            O18 = [(O15[i] - O14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O18[i] = (O18[i] * denom_inv) % m
                O18[i] = O18[i] // even_part
            O19 = [(O16[i] - O15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                O19[i] = (O19[i] * denom_inv) % m
                O19[i] = O19[i] // even_part

            # layer 5
            O20 = [(O18[i] - O17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O20[i] = (O20[i] * denom_inv) % m
                O20[i] = O20[i] // even_part
            O21 = [(O19[i] - O18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O21[i] = (O21[i] * denom_inv) % m
                O21[i] = O21[i] // even_part

            # the remaining odd variables
            r13 = [(O21[i] - O20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                r13[i] = (r13[i] * denom_inv) % m
                r13[i] = r13[i] // even_part
            r11 = [(O20[i] - 91*r13[i] ) % m for i in range(L)]
            r9 = [(O17[i] - 55*r11[i] - 2002*r13[i] ) % m for i in range(L)]
            r7 = [(O13[i] - 30*r9[i] - 627*r11[i] - 11440*r13[i] ) % m for i in range(L)]
            r5 = [(O8[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] - 13013*r13[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] - 1365*r13[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] - r13[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14)
        
        if n == 9:
            r0 = r[0]
            r16 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r16[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 65536*r16[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 43046721*r16[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 4294967296*r16[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 152587890625*r16[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [((r[6][i] + r[-6][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E6[i] = (E6[i] - r0[i] - 2821109907456*r16[i]) % m
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
                E6[i] = (E6[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
            E7 = [((r[7][i] + r[-7][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E7[i] = (E7[i] - r0[i] - 33232930569601*r16[i]) % m
                odd_part,even_part = split_powers_of_two(49)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
                E7[i] = (E7[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part

            # layer 2
            E8 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
            E9 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
            E10 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
            E11 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
            E12 = [(E7[i] - E6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part

            # layer 3
            E13 = [(E9[i] - E8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part
            E14 = [(E10[i] - E9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E14[i] = (E14[i] * denom_inv) % m
                E14[i] = E14[i] // even_part
            E15 = [(E11[i] - E10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                E15[i] = (E15[i] * denom_inv) % m
                E15[i] = E15[i] // even_part
            E16 = [(E12[i] - E11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E16[i] = (E16[i] * denom_inv) % m
                E16[i] = E16[i] // even_part

            # layer 4
            E17 = [(E14[i] - E13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E17[i] = (E17[i] * denom_inv) % m
                E17[i] = E17[i] // even_part
            E18 = [(E15[i] - E14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                E18[i] = (E18[i] * denom_inv) % m
                E18[i] = E18[i] // even_part
            E19 = [(E16[i] - E15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                E19[i] = (E19[i] * denom_inv) % m
                E19[i] = E19[i] // even_part

            # layer 5
            E20 = [(E18[i] - E17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E20[i] = (E20[i] * denom_inv) % m
                E20[i] = E20[i] // even_part
            E21 = [(E19[i] - E18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E21[i] = (E21[i] * denom_inv) % m
                E21[i] = E21[i] // even_part

            # the remaining even variables
            r14 = [(E21[i] - E20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                r14[i] = (r14[i] * denom_inv) % m
                r14[i] = r14[i] // even_part
            r12 = [(E20[i] - 91*r14[i] ) % m for i in range(L)]
            r10 = [(E17[i] - 55*r12[i] - 2002*r14[i] ) % m for i in range(L)]
            r8 = [(E13[i] - 30*r10[i] - 627*r12[i] - 11440*r14[i] ) % m for i in range(L)]
            r6 = [(E8[i] - 14*r8[i] - 147*r10[i] - 1408*r12[i] - 13013*r14[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] - 341*r12[i] - 1365*r14[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] - r12[i] - r14[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r[-6][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(r[7][i] - r[-7][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(14)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
                O7[i] = (O7[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(r[8][i] - r0[i] - 64*r2[i] - 4096*r4[i] - 262144*r6[i] - 16777216*r8[i] - 1073741824*r10[i] - 68719476736*r12[i] - 4398046511104*r14[i] - 281474976710656*r16[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
                O8[i] = (O8[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part

            # layer 2
            O9 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
            O11 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
            O12 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
            O13 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
            O14 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part

            # layer 3
            O15 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part
            O16 = [(O11[i] - O10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O16[i] = (O16[i] * denom_inv) % m
                O16[i] = O16[i] // even_part
            O17 = [(O12[i] - O11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O17[i] = (O17[i] * denom_inv) % m
                O17[i] = O17[i] // even_part
            O18 = [(O13[i] - O12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O18[i] = (O18[i] * denom_inv) % m
                O18[i] = O18[i] // even_part
            O19 = [(O14[i] - O13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                O19[i] = (O19[i] * denom_inv) % m
                O19[i] = O19[i] // even_part

            # layer 4
            O20 = [(O16[i] - O15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O20[i] = (O20[i] * denom_inv) % m
                O20[i] = O20[i] // even_part
            O21 = [(O17[i] - O16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O21[i] = (O21[i] * denom_inv) % m
                O21[i] = O21[i] // even_part
            O22 = [(O18[i] - O17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                O22[i] = (O22[i] * denom_inv) % m
                O22[i] = O22[i] // even_part
            O23 = [(O19[i] - O18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                O23[i] = (O23[i] * denom_inv) % m
                O23[i] = O23[i] // even_part

            # layer 5
            O24 = [(O21[i] - O20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O24[i] = (O24[i] * denom_inv) % m
                O24[i] = O24[i] // even_part
            O25 = [(O22[i] - O21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O25[i] = (O25[i] * denom_inv) % m
                O25[i] = O25[i] // even_part
            O26 = [(O23[i] - O22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O26[i] = (O26[i] * denom_inv) % m
                O26[i] = O26[i] // even_part

            # layer 6
            O27 = [(O25[i] - O24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O27[i] = (O27[i] * denom_inv) % m
                O27[i] = O27[i] // even_part
            O28 = [(O26[i] - O25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                O28[i] = (O28[i] * denom_inv) % m
                O28[i] = O28[i] // even_part

            # the remaining odd variables
            r15 = [(O28[i] - O27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                r15[i] = (r15[i] * denom_inv) % m
                r15[i] = r15[i] // even_part
            r13 = [(O27[i] - 140*r15[i] ) % m for i in range(L)]
            r11 = [(O24[i] - 91*r13[i] - 5278*r15[i] ) % m for i in range(L)]
            r9 = [(O20[i] - 55*r11[i] - 2002*r13[i] - 61490*r15[i] ) % m for i in range(L)]
            r7 = [(O15[i] - 30*r9[i] - 627*r11[i] - 11440*r13[i] - 196053*r15[i] ) % m for i in range(L)]
            r5 = [(O9[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] - 13013*r13[i] - 118482*r15[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] - 1365*r13[i] - 5461*r15[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] - r13[i] - r15[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16)
        
        if n == 10:
            r0 = r[0]
            r18 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r18[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 262144*r18[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 387420489*r18[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 68719476736*r18[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 3814697265625*r18[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [((r[6][i] + r[-6][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E6[i] = (E6[i] - r0[i] - 101559956668416*r18[i]) % m
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
                E6[i] = (E6[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
            E7 = [((r[7][i] + r[-7][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E7[i] = (E7[i] - r0[i] - 1628413597910449*r18[i]) % m
                odd_part,even_part = split_powers_of_two(49)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
                E7[i] = (E7[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
            E8 = [((r[8][i] + r[-8][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E8[i] = (E8[i] - r0[i] - 18014398509481984*r18[i]) % m
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
                E8[i] = (E8[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part

            # layer 2
            E9 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
            E10 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
            E11 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
            E12 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part
            E13 = [(E7[i] - E6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part
            E14 = [(E8[i] - E7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E14[i] = (E14[i] * denom_inv) % m
                E14[i] = E14[i] // even_part

            # layer 3
            E15 = [(E10[i] - E9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E15[i] = (E15[i] * denom_inv) % m
                E15[i] = E15[i] // even_part
            E16 = [(E11[i] - E10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E16[i] = (E16[i] * denom_inv) % m
                E16[i] = E16[i] // even_part
            E17 = [(E12[i] - E11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                E17[i] = (E17[i] * denom_inv) % m
                E17[i] = E17[i] // even_part
            E18 = [(E13[i] - E12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E18[i] = (E18[i] * denom_inv) % m
                E18[i] = E18[i] // even_part
            E19 = [(E14[i] - E13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                E19[i] = (E19[i] * denom_inv) % m
                E19[i] = E19[i] // even_part

            # layer 4
            E20 = [(E16[i] - E15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E20[i] = (E20[i] * denom_inv) % m
                E20[i] = E20[i] // even_part
            E21 = [(E17[i] - E16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                E21[i] = (E21[i] * denom_inv) % m
                E21[i] = E21[i] // even_part
            E22 = [(E18[i] - E17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                E22[i] = (E22[i] * denom_inv) % m
                E22[i] = E22[i] // even_part
            E23 = [(E19[i] - E18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                E23[i] = (E23[i] * denom_inv) % m
                E23[i] = E23[i] // even_part

            # layer 5
            E24 = [(E21[i] - E20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E24[i] = (E24[i] * denom_inv) % m
                E24[i] = E24[i] // even_part
            E25 = [(E22[i] - E21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E25[i] = (E25[i] * denom_inv) % m
                E25[i] = E25[i] // even_part
            E26 = [(E23[i] - E22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E26[i] = (E26[i] * denom_inv) % m
                E26[i] = E26[i] // even_part

            # layer 6
            E27 = [(E25[i] - E24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E27[i] = (E27[i] * denom_inv) % m
                E27[i] = E27[i] // even_part
            E28 = [(E26[i] - E25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                E28[i] = (E28[i] * denom_inv) % m
                E28[i] = E28[i] // even_part

            # the remaining even variables
            r16 = [(E28[i] - E27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                r16[i] = (r16[i] * denom_inv) % m
                r16[i] = r16[i] // even_part
            r14 = [(E27[i] - 140*r16[i] ) % m for i in range(L)]
            r12 = [(E24[i] - 91*r14[i] - 5278*r16[i] ) % m for i in range(L)]
            r10 = [(E20[i] - 55*r12[i] - 2002*r14[i] - 61490*r16[i] ) % m for i in range(L)]
            r8 = [(E15[i] - 30*r10[i] - 627*r12[i] - 11440*r14[i] - 196053*r16[i] ) % m for i in range(L)]
            r6 = [(E9[i] - 14*r8[i] - 147*r10[i] - 1408*r12[i] - 13013*r14[i] - 118482*r16[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] - 341*r12[i] - 1365*r14[i] - 5461*r16[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] - r12[i] - r14[i] - r16[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r[-6][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(r[7][i] - r[-7][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(14)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
                O7[i] = (O7[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(r[8][i] - r[-8][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
                O8[i] = (O8[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
            O9 = [(r[9][i] - r0[i] - 81*r2[i] - 6561*r4[i] - 531441*r6[i] - 43046721*r8[i] - 3486784401*r10[i] - 282429536481*r12[i] - 22876792454961*r14[i] - 1853020188851841*r16[i] - 150094635296999121*r18[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
                O9[i] = (O9[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part

            # layer 2
            O10 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
            O11 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
            O12 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
            O13 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
            O14 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part
            O15 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part
            O16 = [(O9[i] - O8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                O16[i] = (O16[i] * denom_inv) % m
                O16[i] = O16[i] // even_part

            # layer 3
            O17 = [(O11[i] - O10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O17[i] = (O17[i] * denom_inv) % m
                O17[i] = O17[i] // even_part
            O18 = [(O12[i] - O11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O18[i] = (O18[i] * denom_inv) % m
                O18[i] = O18[i] // even_part
            O19 = [(O13[i] - O12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O19[i] = (O19[i] * denom_inv) % m
                O19[i] = O19[i] // even_part
            O20 = [(O14[i] - O13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O20[i] = (O20[i] * denom_inv) % m
                O20[i] = O20[i] // even_part
            O21 = [(O15[i] - O14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                O21[i] = (O21[i] * denom_inv) % m
                O21[i] = O21[i] // even_part
            O22 = [(O16[i] - O15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O22[i] = (O22[i] * denom_inv) % m
                O22[i] = O22[i] // even_part

            # layer 4
            O23 = [(O18[i] - O17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O23[i] = (O23[i] * denom_inv) % m
                O23[i] = O23[i] // even_part
            O24 = [(O19[i] - O18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O24[i] = (O24[i] * denom_inv) % m
                O24[i] = O24[i] // even_part
            O25 = [(O20[i] - O19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                O25[i] = (O25[i] * denom_inv) % m
                O25[i] = O25[i] // even_part
            O26 = [(O21[i] - O20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                O26[i] = (O26[i] * denom_inv) % m
                O26[i] = O26[i] // even_part
            O27 = [(O22[i] - O21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O27[i] = (O27[i] * denom_inv) % m
                O27[i] = O27[i] // even_part

            # layer 5
            O28 = [(O24[i] - O23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O28[i] = (O28[i] * denom_inv) % m
                O28[i] = O28[i] // even_part
            O29 = [(O25[i] - O24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O29[i] = (O29[i] * denom_inv) % m
                O29[i] = O29[i] // even_part
            O30 = [(O26[i] - O25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O30[i] = (O30[i] * denom_inv) % m
                O30[i] = O30[i] // even_part
            O31 = [(O27[i] - O26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                O31[i] = (O31[i] * denom_inv) % m
                O31[i] = O31[i] // even_part

            # layer 6
            O32 = [(O29[i] - O28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O32[i] = (O32[i] * denom_inv) % m
                O32[i] = O32[i] // even_part
            O33 = [(O30[i] - O29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                O33[i] = (O33[i] * denom_inv) % m
                O33[i] = O33[i] // even_part
            O34 = [(O31[i] - O30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                O34[i] = (O34[i] * denom_inv) % m
                O34[i] = O34[i] // even_part

            # layer 7
            O35 = [(O33[i] - O32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                O35[i] = (O35[i] * denom_inv) % m
                O35[i] = O35[i] // even_part
            O36 = [(O34[i] - O33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O36[i] = (O36[i] * denom_inv) % m
                O36[i] = O36[i] // even_part

            # the remaining odd variables
            r17 = [(O36[i] - O35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                r17[i] = (r17[i] * denom_inv) % m
                r17[i] = r17[i] // even_part
            r15 = [(O35[i] - 204*r17[i] ) % m for i in range(L)]
            r13 = [(O32[i] - 140*r15[i] - 12138*r17[i] ) % m for i in range(L)]
            r11 = [(O28[i] - 91*r13[i] - 5278*r15[i] - 251498*r17[i] ) % m for i in range(L)]
            r9 = [(O23[i] - 55*r11[i] - 2002*r13[i] - 61490*r15[i] - 1733303*r17[i] ) % m for i in range(L)]
            r7 = [(O17[i] - 30*r9[i] - 627*r11[i] - 11440*r13[i] - 196053*r15[i] - 3255330*r17[i] ) % m for i in range(L)]
            r5 = [(O10[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] - 13013*r13[i] - 118482*r15[i] - 1071799*r17[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] - 1365*r13[i] - 5461*r15[i] - 21845*r17[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] - r13[i] - r15[i] - r17[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18)
        
        if n == 11:
            r0 = r[0]
            r20 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r20[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 1048576*r20[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 3486784401*r20[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 1099511627776*r20[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 95367431640625*r20[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [((r[6][i] + r[-6][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E6[i] = (E6[i] - r0[i] - 3656158440062976*r20[i]) % m
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
                E6[i] = (E6[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
            E7 = [((r[7][i] + r[-7][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E7[i] = (E7[i] - r0[i] - 79792266297612001*r20[i]) % m
                odd_part,even_part = split_powers_of_two(49)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
                E7[i] = (E7[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
            E8 = [((r[8][i] + r[-8][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E8[i] = (E8[i] - r0[i] - 1152921504606846976*r20[i]) % m
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
                E8[i] = (E8[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
            E9 = [((r[9][i] + r[-9][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E9[i] = (E9[i] - r0[i] - 12157665459056928801*r20[i]) % m
                odd_part,even_part = split_powers_of_two(81)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
                E9[i] = (E9[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part

            # layer 2
            E10 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
            E11 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
            E12 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part
            E13 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part
            E14 = [(E7[i] - E6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                E14[i] = (E14[i] * denom_inv) % m
                E14[i] = E14[i] // even_part
            E15 = [(E8[i] - E7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E15[i] = (E15[i] * denom_inv) % m
                E15[i] = E15[i] // even_part
            E16 = [(E9[i] - E8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                E16[i] = (E16[i] * denom_inv) % m
                E16[i] = E16[i] // even_part

            # layer 3
            E17 = [(E11[i] - E10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E17[i] = (E17[i] * denom_inv) % m
                E17[i] = E17[i] // even_part
            E18 = [(E12[i] - E11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E18[i] = (E18[i] * denom_inv) % m
                E18[i] = E18[i] // even_part
            E19 = [(E13[i] - E12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                E19[i] = (E19[i] * denom_inv) % m
                E19[i] = E19[i] // even_part
            E20 = [(E14[i] - E13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E20[i] = (E20[i] * denom_inv) % m
                E20[i] = E20[i] // even_part
            E21 = [(E15[i] - E14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                E21[i] = (E21[i] * denom_inv) % m
                E21[i] = E21[i] // even_part
            E22 = [(E16[i] - E15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E22[i] = (E22[i] * denom_inv) % m
                E22[i] = E22[i] // even_part

            # layer 4
            E23 = [(E18[i] - E17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E23[i] = (E23[i] * denom_inv) % m
                E23[i] = E23[i] // even_part
            E24 = [(E19[i] - E18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                E24[i] = (E24[i] * denom_inv) % m
                E24[i] = E24[i] // even_part
            E25 = [(E20[i] - E19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                E25[i] = (E25[i] * denom_inv) % m
                E25[i] = E25[i] // even_part
            E26 = [(E21[i] - E20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                E26[i] = (E26[i] * denom_inv) % m
                E26[i] = E26[i] // even_part
            E27 = [(E22[i] - E21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E27[i] = (E27[i] * denom_inv) % m
                E27[i] = E27[i] // even_part

            # layer 5
            E28 = [(E24[i] - E23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E28[i] = (E28[i] * denom_inv) % m
                E28[i] = E28[i] // even_part
            E29 = [(E25[i] - E24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E29[i] = (E29[i] * denom_inv) % m
                E29[i] = E29[i] // even_part
            E30 = [(E26[i] - E25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E30[i] = (E30[i] * denom_inv) % m
                E30[i] = E30[i] // even_part
            E31 = [(E27[i] - E26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                E31[i] = (E31[i] * denom_inv) % m
                E31[i] = E31[i] // even_part

            # layer 6
            E32 = [(E29[i] - E28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E32[i] = (E32[i] * denom_inv) % m
                E32[i] = E32[i] // even_part
            E33 = [(E30[i] - E29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                E33[i] = (E33[i] * denom_inv) % m
                E33[i] = E33[i] // even_part
            E34 = [(E31[i] - E30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                E34[i] = (E34[i] * denom_inv) % m
                E34[i] = E34[i] // even_part

            # layer 7
            E35 = [(E33[i] - E32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                E35[i] = (E35[i] * denom_inv) % m
                E35[i] = E35[i] // even_part
            E36 = [(E34[i] - E33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                E36[i] = (E36[i] * denom_inv) % m
                E36[i] = E36[i] // even_part

            # the remaining even variables
            r18 = [(E36[i] - E35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                r18[i] = (r18[i] * denom_inv) % m
                r18[i] = r18[i] // even_part
            r16 = [(E35[i] - 204*r18[i] ) % m for i in range(L)]
            r14 = [(E32[i] - 140*r16[i] - 12138*r18[i] ) % m for i in range(L)]
            r12 = [(E28[i] - 91*r14[i] - 5278*r16[i] - 251498*r18[i] ) % m for i in range(L)]
            r10 = [(E23[i] - 55*r12[i] - 2002*r14[i] - 61490*r16[i] - 1733303*r18[i] ) % m for i in range(L)]
            r8 = [(E17[i] - 30*r10[i] - 627*r12[i] - 11440*r14[i] - 196053*r16[i] - 3255330*r18[i] ) % m for i in range(L)]
            r6 = [(E10[i] - 14*r8[i] - 147*r10[i] - 1408*r12[i] - 13013*r14[i] - 118482*r16[i] - 1071799*r18[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] - 341*r12[i] - 1365*r14[i] - 5461*r16[i] - 21845*r18[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] - r12[i] - r14[i] - r16[i] - r18[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r[-6][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(r[7][i] - r[-7][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(14)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
                O7[i] = (O7[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(r[8][i] - r[-8][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
                O8[i] = (O8[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
            O9 = [(r[9][i] - r[-9][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(18)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
                O9[i] = (O9[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(r[10][i] - r0[i] - 100*r2[i] - 10000*r4[i] - 1000000*r6[i] - 100000000*r8[i] - 10000000000*r10[i] - 1000000000000*r12[i] - 100000000000000*r14[i] - 10000000000000000*r16[i] - 1000000000000000000*r18[i] - 100000000000000000000*r20[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
                O10[i] = (O10[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part

            # layer 2
            O11 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
            O12 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
            O13 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
            O14 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part
            O15 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part
            O16 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O16[i] = (O16[i] * denom_inv) % m
                O16[i] = O16[i] // even_part
            O17 = [(O9[i] - O8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                O17[i] = (O17[i] * denom_inv) % m
                O17[i] = O17[i] // even_part
            O18 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                O18[i] = (O18[i] * denom_inv) % m
                O18[i] = O18[i] // even_part

            # layer 3
            O19 = [(O12[i] - O11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O19[i] = (O19[i] * denom_inv) % m
                O19[i] = O19[i] // even_part
            O20 = [(O13[i] - O12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O20[i] = (O20[i] * denom_inv) % m
                O20[i] = O20[i] // even_part
            O21 = [(O14[i] - O13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O21[i] = (O21[i] * denom_inv) % m
                O21[i] = O21[i] // even_part
            O22 = [(O15[i] - O14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O22[i] = (O22[i] * denom_inv) % m
                O22[i] = O22[i] // even_part
            O23 = [(O16[i] - O15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                O23[i] = (O23[i] * denom_inv) % m
                O23[i] = O23[i] // even_part
            O24 = [(O17[i] - O16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O24[i] = (O24[i] * denom_inv) % m
                O24[i] = O24[i] // even_part
            O25 = [(O18[i] - O17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                O25[i] = (O25[i] * denom_inv) % m
                O25[i] = O25[i] // even_part

            # layer 4
            O26 = [(O20[i] - O19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O26[i] = (O26[i] * denom_inv) % m
                O26[i] = O26[i] // even_part
            O27 = [(O21[i] - O20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O27[i] = (O27[i] * denom_inv) % m
                O27[i] = O27[i] // even_part
            O28 = [(O22[i] - O21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                O28[i] = (O28[i] * denom_inv) % m
                O28[i] = O28[i] // even_part
            O29 = [(O23[i] - O22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                O29[i] = (O29[i] * denom_inv) % m
                O29[i] = O29[i] // even_part
            O30 = [(O24[i] - O23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O30[i] = (O30[i] * denom_inv) % m
                O30[i] = O30[i] // even_part
            O31 = [(O25[i] - O24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                O31[i] = (O31[i] * denom_inv) % m
                O31[i] = O31[i] // even_part

            # layer 5
            O32 = [(O27[i] - O26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O32[i] = (O32[i] * denom_inv) % m
                O32[i] = O32[i] // even_part
            O33 = [(O28[i] - O27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O33[i] = (O33[i] * denom_inv) % m
                O33[i] = O33[i] // even_part
            O34 = [(O29[i] - O28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O34[i] = (O34[i] * denom_inv) % m
                O34[i] = O34[i] // even_part
            O35 = [(O30[i] - O29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                O35[i] = (O35[i] * denom_inv) % m
                O35[i] = O35[i] // even_part
            O36 = [(O31[i] - O30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                O36[i] = (O36[i] * denom_inv) % m
                O36[i] = O36[i] // even_part

            # layer 6
            O37 = [(O33[i] - O32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O37[i] = (O37[i] * denom_inv) % m
                O37[i] = O37[i] // even_part
            O38 = [(O34[i] - O33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                O38[i] = (O38[i] * denom_inv) % m
                O38[i] = O38[i] // even_part
            O39 = [(O35[i] - O34[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                O39[i] = (O39[i] * denom_inv) % m
                O39[i] = O39[i] // even_part
            O40 = [(O36[i] - O35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                O40[i] = (O40[i] * denom_inv) % m
                O40[i] = O40[i] // even_part

            # layer 7
            O41 = [(O38[i] - O37[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                O41[i] = (O41[i] * denom_inv) % m
                O41[i] = O41[i] // even_part
            O42 = [(O39[i] - O38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O42[i] = (O42[i] * denom_inv) % m
                O42[i] = O42[i] // even_part
            O43 = [(O40[i] - O39[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                O43[i] = (O43[i] * denom_inv) % m
                O43[i] = O43[i] // even_part

            # layer 8
            O44 = [(O42[i] - O41[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                O44[i] = (O44[i] * denom_inv) % m
                O44[i] = O44[i] // even_part
            O45 = [(O43[i] - O42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                O45[i] = (O45[i] * denom_inv) % m
                O45[i] = O45[i] // even_part

            # the remaining odd variables
            r19 = [(O45[i] - O44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                r19[i] = (r19[i] * denom_inv) % m
                r19[i] = r19[i] // even_part
            r17 = [(O44[i] - 285*r19[i] ) % m for i in range(L)]
            r15 = [(O41[i] - 204*r17[i] - 25194*r19[i] ) % m for i in range(L)]
            r13 = [(O37[i] - 140*r15[i] - 12138*r17[i] - 846260*r19[i] ) % m for i in range(L)]
            r11 = [(O32[i] - 91*r13[i] - 5278*r15[i] - 251498*r17[i] - 10787231*r19[i] ) % m for i in range(L)]
            r9 = [(O26[i] - 55*r11[i] - 2002*r13[i] - 61490*r15[i] - 1733303*r17[i] - 46587905*r19[i] ) % m for i in range(L)]
            r7 = [(O19[i] - 30*r9[i] - 627*r11[i] - 11440*r13[i] - 196053*r15[i] - 3255330*r17[i] - 53157079*r19[i] ) % m for i in range(L)]
            r5 = [(O11[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] - 13013*r13[i] - 118482*r15[i] - 1071799*r17[i] - 9668036*r19[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] - 1365*r13[i] - 5461*r15[i] - 21845*r17[i] - 87381*r19[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] - r13[i] - r15[i] - r17[i] - r19[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20)
        
        if n == 12:
            r0 = r[0]
            r22 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r22[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 4194304*r22[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 31381059609*r22[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 17592186044416*r22[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 2384185791015625*r22[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [((r[6][i] + r[-6][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E6[i] = (E6[i] - r0[i] - 131621703842267136*r22[i]) % m
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
                E6[i] = (E6[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
            E7 = [((r[7][i] + r[-7][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E7[i] = (E7[i] - r0[i] - 3909821048582988049*r22[i]) % m
                odd_part,even_part = split_powers_of_two(49)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
                E7[i] = (E7[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
            E8 = [((r[8][i] + r[-8][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E8[i] = (E8[i] - r0[i] - 73786976294838206464*r22[i]) % m
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
                E8[i] = (E8[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
            E9 = [((r[9][i] + r[-9][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E9[i] = (E9[i] - r0[i] - 984770902183611232881*r22[i]) % m
                odd_part,even_part = split_powers_of_two(81)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
                E9[i] = (E9[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
            E10 = [((r[10][i] + r[-10][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E10[i] = (E10[i] - r0[i] - 10000000000000000000000*r22[i]) % m
                odd_part,even_part = split_powers_of_two(100)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
                E10[i] = (E10[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part

            # layer 2
            E11 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
            E12 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part
            E13 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part
            E14 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                E14[i] = (E14[i] * denom_inv) % m
                E14[i] = E14[i] // even_part
            E15 = [(E7[i] - E6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                E15[i] = (E15[i] * denom_inv) % m
                E15[i] = E15[i] // even_part
            E16 = [(E8[i] - E7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E16[i] = (E16[i] * denom_inv) % m
                E16[i] = E16[i] // even_part
            E17 = [(E9[i] - E8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                E17[i] = (E17[i] * denom_inv) % m
                E17[i] = E17[i] // even_part
            E18 = [(E10[i] - E9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                E18[i] = (E18[i] * denom_inv) % m
                E18[i] = E18[i] // even_part

            # layer 3
            E19 = [(E12[i] - E11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E19[i] = (E19[i] * denom_inv) % m
                E19[i] = E19[i] // even_part
            E20 = [(E13[i] - E12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E20[i] = (E20[i] * denom_inv) % m
                E20[i] = E20[i] // even_part
            E21 = [(E14[i] - E13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                E21[i] = (E21[i] * denom_inv) % m
                E21[i] = E21[i] // even_part
            E22 = [(E15[i] - E14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E22[i] = (E22[i] * denom_inv) % m
                E22[i] = E22[i] // even_part
            E23 = [(E16[i] - E15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                E23[i] = (E23[i] * denom_inv) % m
                E23[i] = E23[i] // even_part
            E24 = [(E17[i] - E16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E24[i] = (E24[i] * denom_inv) % m
                E24[i] = E24[i] // even_part
            E25 = [(E18[i] - E17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E25[i] = (E25[i] * denom_inv) % m
                E25[i] = E25[i] // even_part

            # layer 4
            E26 = [(E20[i] - E19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E26[i] = (E26[i] * denom_inv) % m
                E26[i] = E26[i] // even_part
            E27 = [(E21[i] - E20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                E27[i] = (E27[i] * denom_inv) % m
                E27[i] = E27[i] // even_part
            E28 = [(E22[i] - E21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                E28[i] = (E28[i] * denom_inv) % m
                E28[i] = E28[i] // even_part
            E29 = [(E23[i] - E22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                E29[i] = (E29[i] * denom_inv) % m
                E29[i] = E29[i] // even_part
            E30 = [(E24[i] - E23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E30[i] = (E30[i] * denom_inv) % m
                E30[i] = E30[i] // even_part
            E31 = [(E25[i] - E24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                E31[i] = (E31[i] * denom_inv) % m
                E31[i] = E31[i] // even_part

            # layer 5
            E32 = [(E27[i] - E26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E32[i] = (E32[i] * denom_inv) % m
                E32[i] = E32[i] // even_part
            E33 = [(E28[i] - E27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E33[i] = (E33[i] * denom_inv) % m
                E33[i] = E33[i] // even_part
            E34 = [(E29[i] - E28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E34[i] = (E34[i] * denom_inv) % m
                E34[i] = E34[i] // even_part
            E35 = [(E30[i] - E29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                E35[i] = (E35[i] * denom_inv) % m
                E35[i] = E35[i] // even_part
            E36 = [(E31[i] - E30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E36[i] = (E36[i] * denom_inv) % m
                E36[i] = E36[i] // even_part

            # layer 6
            E37 = [(E33[i] - E32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E37[i] = (E37[i] * denom_inv) % m
                E37[i] = E37[i] // even_part
            E38 = [(E34[i] - E33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                E38[i] = (E38[i] * denom_inv) % m
                E38[i] = E38[i] // even_part
            E39 = [(E35[i] - E34[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                E39[i] = (E39[i] * denom_inv) % m
                E39[i] = E39[i] // even_part
            E40 = [(E36[i] - E35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                E40[i] = (E40[i] * denom_inv) % m
                E40[i] = E40[i] // even_part

            # layer 7
            E41 = [(E38[i] - E37[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                E41[i] = (E41[i] * denom_inv) % m
                E41[i] = E41[i] // even_part
            E42 = [(E39[i] - E38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                E42[i] = (E42[i] * denom_inv) % m
                E42[i] = E42[i] // even_part
            E43 = [(E40[i] - E39[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                E43[i] = (E43[i] * denom_inv) % m
                E43[i] = E43[i] // even_part

            # layer 8
            E44 = [(E42[i] - E41[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                E44[i] = (E44[i] * denom_inv) % m
                E44[i] = E44[i] // even_part
            E45 = [(E43[i] - E42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                E45[i] = (E45[i] * denom_inv) % m
                E45[i] = E45[i] // even_part

            # the remaining even variables
            r20 = [(E45[i] - E44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                r20[i] = (r20[i] * denom_inv) % m
                r20[i] = r20[i] // even_part
            r18 = [(E44[i] - 285*r20[i] ) % m for i in range(L)]
            r16 = [(E41[i] - 204*r18[i] - 25194*r20[i] ) % m for i in range(L)]
            r14 = [(E37[i] - 140*r16[i] - 12138*r18[i] - 846260*r20[i] ) % m for i in range(L)]
            r12 = [(E32[i] - 91*r14[i] - 5278*r16[i] - 251498*r18[i] - 10787231*r20[i] ) % m for i in range(L)]
            r10 = [(E26[i] - 55*r12[i] - 2002*r14[i] - 61490*r16[i] - 1733303*r18[i] - 46587905*r20[i] ) % m for i in range(L)]
            r8 = [(E19[i] - 30*r10[i] - 627*r12[i] - 11440*r14[i] - 196053*r16[i] - 3255330*r18[i] - 53157079*r20[i] ) % m for i in range(L)]
            r6 = [(E11[i] - 14*r8[i] - 147*r10[i] - 1408*r12[i] - 13013*r14[i] - 118482*r16[i] - 1071799*r18[i] - 9668036*r20[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] - 341*r12[i] - 1365*r14[i] - 5461*r16[i] - 21845*r18[i] - 87381*r20[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] - r12[i] - r14[i] - r16[i] - r18[i] - r20[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r[-6][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(r[7][i] - r[-7][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(14)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
                O7[i] = (O7[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(r[8][i] - r[-8][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
                O8[i] = (O8[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
            O9 = [(r[9][i] - r[-9][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(18)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
                O9[i] = (O9[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(r[10][i] - r[-10][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
                O10[i] = (O10[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
            O11 = [(r[11][i] - r0[i] - 121*r2[i] - 14641*r4[i] - 1771561*r6[i] - 214358881*r8[i] - 25937424601*r10[i] - 3138428376721*r12[i] - 379749833583241*r14[i] - 45949729863572161*r16[i] - 5559917313492231481*r18[i] - 672749994932560009201*r20[i] - 81402749386839761113321*r22[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
                O11[i] = (O11[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part

            # layer 2
            O12 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
            O13 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
            O14 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part
            O15 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part
            O16 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O16[i] = (O16[i] * denom_inv) % m
                O16[i] = O16[i] // even_part
            O17 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O17[i] = (O17[i] * denom_inv) % m
                O17[i] = O17[i] // even_part
            O18 = [(O9[i] - O8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                O18[i] = (O18[i] * denom_inv) % m
                O18[i] = O18[i] // even_part
            O19 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                O19[i] = (O19[i] * denom_inv) % m
                O19[i] = O19[i] // even_part
            O20 = [(O11[i] - O10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O20[i] = (O20[i] * denom_inv) % m
                O20[i] = O20[i] // even_part

            # layer 3
            O21 = [(O13[i] - O12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O21[i] = (O21[i] * denom_inv) % m
                O21[i] = O21[i] // even_part
            O22 = [(O14[i] - O13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O22[i] = (O22[i] * denom_inv) % m
                O22[i] = O22[i] // even_part
            O23 = [(O15[i] - O14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O23[i] = (O23[i] * denom_inv) % m
                O23[i] = O23[i] // even_part
            O24 = [(O16[i] - O15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O24[i] = (O24[i] * denom_inv) % m
                O24[i] = O24[i] // even_part
            O25 = [(O17[i] - O16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                O25[i] = (O25[i] * denom_inv) % m
                O25[i] = O25[i] // even_part
            O26 = [(O18[i] - O17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O26[i] = (O26[i] * denom_inv) % m
                O26[i] = O26[i] // even_part
            O27 = [(O19[i] - O18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                O27[i] = (O27[i] * denom_inv) % m
                O27[i] = O27[i] // even_part
            O28 = [(O20[i] - O19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O28[i] = (O28[i] * denom_inv) % m
                O28[i] = O28[i] // even_part

            # layer 4
            O29 = [(O22[i] - O21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O29[i] = (O29[i] * denom_inv) % m
                O29[i] = O29[i] // even_part
            O30 = [(O23[i] - O22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O30[i] = (O30[i] * denom_inv) % m
                O30[i] = O30[i] // even_part
            O31 = [(O24[i] - O23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                O31[i] = (O31[i] * denom_inv) % m
                O31[i] = O31[i] // even_part
            O32 = [(O25[i] - O24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                O32[i] = (O32[i] * denom_inv) % m
                O32[i] = O32[i] // even_part
            O33 = [(O26[i] - O25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O33[i] = (O33[i] * denom_inv) % m
                O33[i] = O33[i] // even_part
            O34 = [(O27[i] - O26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                O34[i] = (O34[i] * denom_inv) % m
                O34[i] = O34[i] // even_part
            O35 = [(O28[i] - O27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(57)
                denom_inv = inverse_mod(odd_part, m)
                O35[i] = (O35[i] * denom_inv) % m
                O35[i] = O35[i] // even_part

            # layer 5
            O36 = [(O30[i] - O29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O36[i] = (O36[i] * denom_inv) % m
                O36[i] = O36[i] // even_part
            O37 = [(O31[i] - O30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O37[i] = (O37[i] * denom_inv) % m
                O37[i] = O37[i] // even_part
            O38 = [(O32[i] - O31[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O38[i] = (O38[i] * denom_inv) % m
                O38[i] = O38[i] // even_part
            O39 = [(O33[i] - O32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                O39[i] = (O39[i] * denom_inv) % m
                O39[i] = O39[i] // even_part
            O40 = [(O34[i] - O33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                O40[i] = (O40[i] * denom_inv) % m
                O40[i] = O40[i] // even_part
            O41 = [(O35[i] - O34[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O41[i] = (O41[i] * denom_inv) % m
                O41[i] = O41[i] // even_part

            # layer 6
            O42 = [(O37[i] - O36[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O42[i] = (O42[i] * denom_inv) % m
                O42[i] = O42[i] // even_part
            O43 = [(O38[i] - O37[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                O43[i] = (O43[i] * denom_inv) % m
                O43[i] = O43[i] // even_part
            O44 = [(O39[i] - O38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                O44[i] = (O44[i] * denom_inv) % m
                O44[i] = O44[i] // even_part
            O45 = [(O40[i] - O39[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                O45[i] = (O45[i] * denom_inv) % m
                O45[i] = O45[i] // even_part
            O46 = [(O41[i] - O40[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(85)
                denom_inv = inverse_mod(odd_part, m)
                O46[i] = (O46[i] * denom_inv) % m
                O46[i] = O46[i] // even_part

            # layer 7
            O47 = [(O43[i] - O42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                O47[i] = (O47[i] * denom_inv) % m
                O47[i] = O47[i] // even_part
            O48 = [(O44[i] - O43[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O48[i] = (O48[i] * denom_inv) % m
                O48[i] = O48[i] // even_part
            O49 = [(O45[i] - O44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                O49[i] = (O49[i] * denom_inv) % m
                O49[i] = O49[i] // even_part
            O50 = [(O46[i] - O45[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O50[i] = (O50[i] * denom_inv) % m
                O50[i] = O50[i] // even_part

            # layer 8
            O51 = [(O48[i] - O47[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                O51[i] = (O51[i] * denom_inv) % m
                O51[i] = O51[i] // even_part
            O52 = [(O49[i] - O48[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                O52[i] = (O52[i] * denom_inv) % m
                O52[i] = O52[i] // even_part
            O53 = [(O50[i] - O49[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                O53[i] = (O53[i] * denom_inv) % m
                O53[i] = O53[i] // even_part

            # layer 9
            O54 = [(O52[i] - O51[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O54[i] = (O54[i] * denom_inv) % m
                O54[i] = O54[i] // even_part
            O55 = [(O53[i] - O52[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(112)
                denom_inv = inverse_mod(odd_part, m)
                O55[i] = (O55[i] * denom_inv) % m
                O55[i] = O55[i] // even_part

            # the remaining odd variables
            r21 = [(O55[i] - O54[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(117)
                denom_inv = inverse_mod(odd_part, m)
                r21[i] = (r21[i] * denom_inv) % m
                r21[i] = r21[i] // even_part
            r19 = [(O54[i] - 385*r21[i] ) % m for i in range(L)]
            r17 = [(O51[i] - 285*r19[i] - 48279*r21[i] ) % m for i in range(L)]
            r15 = [(O47[i] - 204*r17[i] - 25194*r19[i] - 2458676*r21[i] ) % m for i in range(L)]
            r13 = [(O42[i] - 140*r15[i] - 12138*r17[i] - 846260*r19[i] - 52253971*r21[i] ) % m for i in range(L)]
            r11 = [(O36[i] - 91*r13[i] - 5278*r15[i] - 251498*r17[i] - 10787231*r19[i] - 434928221*r21[i] ) % m for i in range(L)]
            r9 = [(O29[i] - 55*r11[i] - 2002*r13[i] - 61490*r15[i] - 1733303*r17[i] - 46587905*r19[i] - 1217854704*r21[i] ) % m for i in range(L)]
            r7 = [(O21[i] - 30*r9[i] - 627*r11[i] - 11440*r13[i] - 196053*r15[i] - 3255330*r17[i] - 53157079*r19[i] - 860181300*r21[i] ) % m for i in range(L)]
            r5 = [(O12[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] - 13013*r13[i] - 118482*r15[i] - 1071799*r17[i] - 9668036*r19[i] - 87099705*r21[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] - 1365*r13[i] - 5461*r15[i] - 21845*r17[i] - 87381*r19[i] - 349525*r21[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] - r13[i] - r15[i] - r17[i] - r19[i] - r21[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22)
        
        if n == 13:
            r0 = r[0]
            r24 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r24[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 16777216*r24[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 282429536481*r24[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 281474976710656*r24[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 59604644775390625*r24[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [((r[6][i] + r[-6][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E6[i] = (E6[i] - r0[i] - 4738381338321616896*r24[i]) % m
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
                E6[i] = (E6[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
            E7 = [((r[7][i] + r[-7][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E7[i] = (E7[i] - r0[i] - 191581231380566414401*r24[i]) % m
                odd_part,even_part = split_powers_of_two(49)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
                E7[i] = (E7[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
            E8 = [((r[8][i] + r[-8][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E8[i] = (E8[i] - r0[i] - 4722366482869645213696*r24[i]) % m
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
                E8[i] = (E8[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
            E9 = [((r[9][i] + r[-9][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E9[i] = (E9[i] - r0[i] - 79766443076872509863361*r24[i]) % m
                odd_part,even_part = split_powers_of_two(81)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
                E9[i] = (E9[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
            E10 = [((r[10][i] + r[-10][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E10[i] = (E10[i] - r0[i] - 1000000000000000000000000*r24[i]) % m
                odd_part,even_part = split_powers_of_two(100)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
                E10[i] = (E10[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
            E11 = [((r[11][i] + r[-11][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E11[i] = (E11[i] - r0[i] - 9849732675807611094711841*r24[i]) % m
                odd_part,even_part = split_powers_of_two(121)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
                E11[i] = (E11[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part

            # layer 2
            E12 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part
            E13 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part
            E14 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E14[i] = (E14[i] * denom_inv) % m
                E14[i] = E14[i] // even_part
            E15 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                E15[i] = (E15[i] * denom_inv) % m
                E15[i] = E15[i] // even_part
            E16 = [(E7[i] - E6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                E16[i] = (E16[i] * denom_inv) % m
                E16[i] = E16[i] // even_part
            E17 = [(E8[i] - E7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E17[i] = (E17[i] * denom_inv) % m
                E17[i] = E17[i] // even_part
            E18 = [(E9[i] - E8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                E18[i] = (E18[i] * denom_inv) % m
                E18[i] = E18[i] // even_part
            E19 = [(E10[i] - E9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                E19[i] = (E19[i] * denom_inv) % m
                E19[i] = E19[i] // even_part
            E20 = [(E11[i] - E10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E20[i] = (E20[i] * denom_inv) % m
                E20[i] = E20[i] // even_part

            # layer 3
            E21 = [(E13[i] - E12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E21[i] = (E21[i] * denom_inv) % m
                E21[i] = E21[i] // even_part
            E22 = [(E14[i] - E13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E22[i] = (E22[i] * denom_inv) % m
                E22[i] = E22[i] // even_part
            E23 = [(E15[i] - E14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                E23[i] = (E23[i] * denom_inv) % m
                E23[i] = E23[i] // even_part
            E24 = [(E16[i] - E15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E24[i] = (E24[i] * denom_inv) % m
                E24[i] = E24[i] // even_part
            E25 = [(E17[i] - E16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                E25[i] = (E25[i] * denom_inv) % m
                E25[i] = E25[i] // even_part
            E26 = [(E18[i] - E17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E26[i] = (E26[i] * denom_inv) % m
                E26[i] = E26[i] // even_part
            E27 = [(E19[i] - E18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E27[i] = (E27[i] * denom_inv) % m
                E27[i] = E27[i] // even_part
            E28 = [(E20[i] - E19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E28[i] = (E28[i] * denom_inv) % m
                E28[i] = E28[i] // even_part

            # layer 4
            E29 = [(E22[i] - E21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E29[i] = (E29[i] * denom_inv) % m
                E29[i] = E29[i] // even_part
            E30 = [(E23[i] - E22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                E30[i] = (E30[i] * denom_inv) % m
                E30[i] = E30[i] // even_part
            E31 = [(E24[i] - E23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                E31[i] = (E31[i] * denom_inv) % m
                E31[i] = E31[i] // even_part
            E32 = [(E25[i] - E24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                E32[i] = (E32[i] * denom_inv) % m
                E32[i] = E32[i] // even_part
            E33 = [(E26[i] - E25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E33[i] = (E33[i] * denom_inv) % m
                E33[i] = E33[i] // even_part
            E34 = [(E27[i] - E26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                E34[i] = (E34[i] * denom_inv) % m
                E34[i] = E34[i] // even_part
            E35 = [(E28[i] - E27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(57)
                denom_inv = inverse_mod(odd_part, m)
                E35[i] = (E35[i] * denom_inv) % m
                E35[i] = E35[i] // even_part

            # layer 5
            E36 = [(E30[i] - E29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E36[i] = (E36[i] * denom_inv) % m
                E36[i] = E36[i] // even_part
            E37 = [(E31[i] - E30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E37[i] = (E37[i] * denom_inv) % m
                E37[i] = E37[i] // even_part
            E38 = [(E32[i] - E31[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E38[i] = (E38[i] * denom_inv) % m
                E38[i] = E38[i] // even_part
            E39 = [(E33[i] - E32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                E39[i] = (E39[i] * denom_inv) % m
                E39[i] = E39[i] // even_part
            E40 = [(E34[i] - E33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E40[i] = (E40[i] * denom_inv) % m
                E40[i] = E40[i] // even_part
            E41 = [(E35[i] - E34[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                E41[i] = (E41[i] * denom_inv) % m
                E41[i] = E41[i] // even_part

            # layer 6
            E42 = [(E37[i] - E36[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E42[i] = (E42[i] * denom_inv) % m
                E42[i] = E42[i] // even_part
            E43 = [(E38[i] - E37[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                E43[i] = (E43[i] * denom_inv) % m
                E43[i] = E43[i] // even_part
            E44 = [(E39[i] - E38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                E44[i] = (E44[i] * denom_inv) % m
                E44[i] = E44[i] // even_part
            E45 = [(E40[i] - E39[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                E45[i] = (E45[i] * denom_inv) % m
                E45[i] = E45[i] // even_part
            E46 = [(E41[i] - E40[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(85)
                denom_inv = inverse_mod(odd_part, m)
                E46[i] = (E46[i] * denom_inv) % m
                E46[i] = E46[i] // even_part

            # layer 7
            E47 = [(E43[i] - E42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                E47[i] = (E47[i] * denom_inv) % m
                E47[i] = E47[i] // even_part
            E48 = [(E44[i] - E43[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                E48[i] = (E48[i] * denom_inv) % m
                E48[i] = E48[i] // even_part
            E49 = [(E45[i] - E44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                E49[i] = (E49[i] * denom_inv) % m
                E49[i] = E49[i] // even_part
            E50 = [(E46[i] - E45[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                E50[i] = (E50[i] * denom_inv) % m
                E50[i] = E50[i] // even_part

            # layer 8
            E51 = [(E48[i] - E47[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                E51[i] = (E51[i] * denom_inv) % m
                E51[i] = E51[i] // even_part
            E52 = [(E49[i] - E48[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                E52[i] = (E52[i] * denom_inv) % m
                E52[i] = E52[i] // even_part
            E53 = [(E50[i] - E49[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                E53[i] = (E53[i] * denom_inv) % m
                E53[i] = E53[i] // even_part

            # layer 9
            E54 = [(E52[i] - E51[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                E54[i] = (E54[i] * denom_inv) % m
                E54[i] = E54[i] // even_part
            E55 = [(E53[i] - E52[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(112)
                denom_inv = inverse_mod(odd_part, m)
                E55[i] = (E55[i] * denom_inv) % m
                E55[i] = E55[i] // even_part

            # the remaining even variables
            r22 = [(E55[i] - E54[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(117)
                denom_inv = inverse_mod(odd_part, m)
                r22[i] = (r22[i] * denom_inv) % m
                r22[i] = r22[i] // even_part
            r20 = [(E54[i] - 385*r22[i] ) % m for i in range(L)]
            r18 = [(E51[i] - 285*r20[i] - 48279*r22[i] ) % m for i in range(L)]
            r16 = [(E47[i] - 204*r18[i] - 25194*r20[i] - 2458676*r22[i] ) % m for i in range(L)]
            r14 = [(E42[i] - 140*r16[i] - 12138*r18[i] - 846260*r20[i] - 52253971*r22[i] ) % m for i in range(L)]
            r12 = [(E36[i] - 91*r14[i] - 5278*r16[i] - 251498*r18[i] - 10787231*r20[i] - 434928221*r22[i] ) % m for i in range(L)]
            r10 = [(E29[i] - 55*r12[i] - 2002*r14[i] - 61490*r16[i] - 1733303*r18[i] - 46587905*r20[i] - 1217854704*r22[i] ) % m for i in range(L)]
            r8 = [(E21[i] - 30*r10[i] - 627*r12[i] - 11440*r14[i] - 196053*r16[i] - 3255330*r18[i] - 53157079*r20[i] - 860181300*r22[i] ) % m for i in range(L)]
            r6 = [(E12[i] - 14*r8[i] - 147*r10[i] - 1408*r12[i] - 13013*r14[i] - 118482*r16[i] - 1071799*r18[i] - 9668036*r20[i] - 87099705*r22[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] - 341*r12[i] - 1365*r14[i] - 5461*r16[i] - 21845*r18[i] - 87381*r20[i] - 349525*r22[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] - r12[i] - r14[i] - r16[i] - r18[i] - r20[i] - r22[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r[-6][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(r[7][i] - r[-7][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(14)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
                O7[i] = (O7[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(r[8][i] - r[-8][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
                O8[i] = (O8[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
            O9 = [(r[9][i] - r[-9][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(18)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
                O9[i] = (O9[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(r[10][i] - r[-10][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
                O10[i] = (O10[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
            O11 = [(r[11][i] - r[-11][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(22)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
                O11[i] = (O11[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
            O12 = [(r[12][i] - r0[i] - 144*r2[i] - 20736*r4[i] - 2985984*r6[i] - 429981696*r8[i] - 61917364224*r10[i] - 8916100448256*r12[i] - 1283918464548864*r14[i] - 184884258895036416*r16[i] - 26623333280885243904*r18[i] - 3833759992447475122176*r20[i] - 552061438912436417593344*r22[i] - 79496847203390844133441536*r24[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
                O12[i] = (O12[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(143)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part

            # layer 2
            O13 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
            O14 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part
            O15 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part
            O16 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O16[i] = (O16[i] * denom_inv) % m
                O16[i] = O16[i] // even_part
            O17 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O17[i] = (O17[i] * denom_inv) % m
                O17[i] = O17[i] // even_part
            O18 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O18[i] = (O18[i] * denom_inv) % m
                O18[i] = O18[i] // even_part
            O19 = [(O9[i] - O8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                O19[i] = (O19[i] * denom_inv) % m
                O19[i] = O19[i] // even_part
            O20 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                O20[i] = (O20[i] * denom_inv) % m
                O20[i] = O20[i] // even_part
            O21 = [(O11[i] - O10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O21[i] = (O21[i] * denom_inv) % m
                O21[i] = O21[i] // even_part
            O22 = [(O12[i] - O11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(23)
                denom_inv = inverse_mod(odd_part, m)
                O22[i] = (O22[i] * denom_inv) % m
                O22[i] = O22[i] // even_part

            # layer 3
            O23 = [(O14[i] - O13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O23[i] = (O23[i] * denom_inv) % m
                O23[i] = O23[i] // even_part
            O24 = [(O15[i] - O14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O24[i] = (O24[i] * denom_inv) % m
                O24[i] = O24[i] // even_part
            O25 = [(O16[i] - O15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O25[i] = (O25[i] * denom_inv) % m
                O25[i] = O25[i] // even_part
            O26 = [(O17[i] - O16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O26[i] = (O26[i] * denom_inv) % m
                O26[i] = O26[i] // even_part
            O27 = [(O18[i] - O17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                O27[i] = (O27[i] * denom_inv) % m
                O27[i] = O27[i] // even_part
            O28 = [(O19[i] - O18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O28[i] = (O28[i] * denom_inv) % m
                O28[i] = O28[i] // even_part
            O29 = [(O20[i] - O19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                O29[i] = (O29[i] * denom_inv) % m
                O29[i] = O29[i] // even_part
            O30 = [(O21[i] - O20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O30[i] = (O30[i] * denom_inv) % m
                O30[i] = O30[i] // even_part
            O31 = [(O22[i] - O21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(44)
                denom_inv = inverse_mod(odd_part, m)
                O31[i] = (O31[i] * denom_inv) % m
                O31[i] = O31[i] // even_part

            # layer 4
            O32 = [(O24[i] - O23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O32[i] = (O32[i] * denom_inv) % m
                O32[i] = O32[i] // even_part
            O33 = [(O25[i] - O24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O33[i] = (O33[i] * denom_inv) % m
                O33[i] = O33[i] // even_part
            O34 = [(O26[i] - O25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                O34[i] = (O34[i] * denom_inv) % m
                O34[i] = O34[i] // even_part
            O35 = [(O27[i] - O26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                O35[i] = (O35[i] * denom_inv) % m
                O35[i] = O35[i] // even_part
            O36 = [(O28[i] - O27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O36[i] = (O36[i] * denom_inv) % m
                O36[i] = O36[i] // even_part
            O37 = [(O29[i] - O28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                O37[i] = (O37[i] * denom_inv) % m
                O37[i] = O37[i] // even_part
            O38 = [(O30[i] - O29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(57)
                denom_inv = inverse_mod(odd_part, m)
                O38[i] = (O38[i] * denom_inv) % m
                O38[i] = O38[i] // even_part
            O39 = [(O31[i] - O30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O39[i] = (O39[i] * denom_inv) % m
                O39[i] = O39[i] // even_part

            # layer 5
            O40 = [(O33[i] - O32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O40[i] = (O40[i] * denom_inv) % m
                O40[i] = O40[i] // even_part
            O41 = [(O34[i] - O33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O41[i] = (O41[i] * denom_inv) % m
                O41[i] = O41[i] // even_part
            O42 = [(O35[i] - O34[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O42[i] = (O42[i] * denom_inv) % m
                O42[i] = O42[i] // even_part
            O43 = [(O36[i] - O35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                O43[i] = (O43[i] * denom_inv) % m
                O43[i] = O43[i] // even_part
            O44 = [(O37[i] - O36[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                O44[i] = (O44[i] * denom_inv) % m
                O44[i] = O44[i] // even_part
            O45 = [(O38[i] - O37[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O45[i] = (O45[i] * denom_inv) % m
                O45[i] = O45[i] // even_part
            O46 = [(O39[i] - O38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O46[i] = (O46[i] * denom_inv) % m
                O46[i] = O46[i] // even_part

            # layer 6
            O47 = [(O41[i] - O40[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O47[i] = (O47[i] * denom_inv) % m
                O47[i] = O47[i] // even_part
            O48 = [(O42[i] - O41[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                O48[i] = (O48[i] * denom_inv) % m
                O48[i] = O48[i] // even_part
            O49 = [(O43[i] - O42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                O49[i] = (O49[i] * denom_inv) % m
                O49[i] = O49[i] // even_part
            O50 = [(O44[i] - O43[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                O50[i] = (O50[i] * denom_inv) % m
                O50[i] = O50[i] // even_part
            O51 = [(O45[i] - O44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(85)
                denom_inv = inverse_mod(odd_part, m)
                O51[i] = (O51[i] * denom_inv) % m
                O51[i] = O51[i] // even_part
            O52 = [(O46[i] - O45[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(95)
                denom_inv = inverse_mod(odd_part, m)
                O52[i] = (O52[i] * denom_inv) % m
                O52[i] = O52[i] // even_part

            # layer 7
            O53 = [(O48[i] - O47[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                O53[i] = (O53[i] * denom_inv) % m
                O53[i] = O53[i] // even_part
            O54 = [(O49[i] - O48[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O54[i] = (O54[i] * denom_inv) % m
                O54[i] = O54[i] // even_part
            O55 = [(O50[i] - O49[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                O55[i] = (O55[i] * denom_inv) % m
                O55[i] = O55[i] // even_part
            O56 = [(O51[i] - O50[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O56[i] = (O56[i] * denom_inv) % m
                O56[i] = O56[i] // even_part
            O57 = [(O52[i] - O51[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(108)
                denom_inv = inverse_mod(odd_part, m)
                O57[i] = (O57[i] * denom_inv) % m
                O57[i] = O57[i] // even_part

            # layer 8
            O58 = [(O54[i] - O53[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                O58[i] = (O58[i] * denom_inv) % m
                O58[i] = O58[i] // even_part
            O59 = [(O55[i] - O54[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                O59[i] = (O59[i] * denom_inv) % m
                O59[i] = O59[i] // even_part
            O60 = [(O56[i] - O55[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                O60[i] = (O60[i] * denom_inv) % m
                O60[i] = O60[i] // even_part
            O61 = [(O57[i] - O56[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(119)
                denom_inv = inverse_mod(odd_part, m)
                O61[i] = (O61[i] * denom_inv) % m
                O61[i] = O61[i] // even_part

            # layer 9
            O62 = [(O59[i] - O58[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O62[i] = (O62[i] * denom_inv) % m
                O62[i] = O62[i] // even_part
            O63 = [(O60[i] - O59[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(112)
                denom_inv = inverse_mod(odd_part, m)
                O63[i] = (O63[i] * denom_inv) % m
                O63[i] = O63[i] // even_part
            O64 = [(O61[i] - O60[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(128)
                denom_inv = inverse_mod(odd_part, m)
                O64[i] = (O64[i] * denom_inv) % m
                O64[i] = O64[i] // even_part

            # layer 10
            O65 = [(O63[i] - O62[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(117)
                denom_inv = inverse_mod(odd_part, m)
                O65[i] = (O65[i] * denom_inv) % m
                O65[i] = O65[i] // even_part
            O66 = [(O64[i] - O63[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(135)
                denom_inv = inverse_mod(odd_part, m)
                O66[i] = (O66[i] * denom_inv) % m
                O66[i] = O66[i] // even_part

            # the remaining odd variables
            r23 = [(O66[i] - O65[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(140)
                denom_inv = inverse_mod(odd_part, m)
                r23[i] = (r23[i] * denom_inv) % m
                r23[i] = r23[i] // even_part
            r21 = [(O65[i] - 506*r23[i] ) % m for i in range(L)]
            r19 = [(O62[i] - 385*r21[i] - 86779*r23[i] ) % m for i in range(L)]
            r17 = [(O58[i] - 285*r19[i] - 48279*r21[i] - 6369275*r23[i] ) % m for i in range(L)]
            r15 = [(O53[i] - 204*r17[i] - 25194*r19[i] - 2458676*r21[i] - 209609235*r23[i] ) % m for i in range(L)]
            r13 = [(O47[i] - 140*r15[i] - 12138*r17[i] - 846260*r19[i] - 52253971*r21[i] - 2995372800*r23[i] ) % m for i in range(L)]
            r11 = [(O40[i] - 91*r13[i] - 5278*r15[i] - 251498*r17[i] - 10787231*r19[i] - 434928221*r21[i] - 16875270660*r23[i] ) % m for i in range(L)]
            r9 = [(O32[i] - 55*r11[i] - 2002*r13[i] - 61490*r15[i] - 1733303*r17[i] - 46587905*r19[i] - 1217854704*r21[i] - 31306548900*r23[i] ) % m for i in range(L)]
            r7 = [(O23[i] - 30*r9[i] - 627*r11[i] - 11440*r13[i] - 196053*r15[i] - 3255330*r17[i] - 53157079*r19[i] - 860181300*r21[i] - 13850000505*r23[i] ) % m for i in range(L)]
            r5 = [(O13[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] - 13013*r13[i] - 118482*r15[i] - 1071799*r17[i] - 9668036*r19[i] - 87099705*r21[i] - 784246870*r23[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] - 1365*r13[i] - 5461*r15[i] - 21845*r17[i] - 87381*r19[i] - 349525*r21[i] - 1398101*r23[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] - r13[i] - r15[i] - r17[i] - r19[i] - r21[i] - r23[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24)
        
        if n == 14:
            r0 = r[0]
            r26 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r26[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 67108864*r26[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 2541865828329*r26[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 4503599627370496*r26[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 1490116119384765625*r26[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [((r[6][i] + r[-6][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E6[i] = (E6[i] - r0[i] - 170581728179578208256*r26[i]) % m
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
                E6[i] = (E6[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
            E7 = [((r[7][i] + r[-7][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E7[i] = (E7[i] - r0[i] - 9387480337647754305649*r26[i]) % m
                odd_part,even_part = split_powers_of_two(49)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
                E7[i] = (E7[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
            E8 = [((r[8][i] + r[-8][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E8[i] = (E8[i] - r0[i] - 302231454903657293676544*r26[i]) % m
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
                E8[i] = (E8[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
            E9 = [((r[9][i] + r[-9][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E9[i] = (E9[i] - r0[i] - 6461081889226673298932241*r26[i]) % m
                odd_part,even_part = split_powers_of_two(81)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
                E9[i] = (E9[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
            E10 = [((r[10][i] + r[-10][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E10[i] = (E10[i] - r0[i] - 100000000000000000000000000*r26[i]) % m
                odd_part,even_part = split_powers_of_two(100)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
                E10[i] = (E10[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
            E11 = [((r[11][i] + r[-11][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E11[i] = (E11[i] - r0[i] - 1191817653772720942460132761*r26[i]) % m
                odd_part,even_part = split_powers_of_two(121)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
                E11[i] = (E11[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
            E12 = [((r[12][i] + r[-12][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E12[i] = (E12[i] - r0[i] - 11447545997288281555215581184*r26[i]) % m
                odd_part,even_part = split_powers_of_two(144)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part
                E12[i] = (E12[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(143)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part

            # layer 2
            E13 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part
            E14 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E14[i] = (E14[i] * denom_inv) % m
                E14[i] = E14[i] // even_part
            E15 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E15[i] = (E15[i] * denom_inv) % m
                E15[i] = E15[i] // even_part
            E16 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                E16[i] = (E16[i] * denom_inv) % m
                E16[i] = E16[i] // even_part
            E17 = [(E7[i] - E6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                E17[i] = (E17[i] * denom_inv) % m
                E17[i] = E17[i] // even_part
            E18 = [(E8[i] - E7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E18[i] = (E18[i] * denom_inv) % m
                E18[i] = E18[i] // even_part
            E19 = [(E9[i] - E8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                E19[i] = (E19[i] * denom_inv) % m
                E19[i] = E19[i] // even_part
            E20 = [(E10[i] - E9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                E20[i] = (E20[i] * denom_inv) % m
                E20[i] = E20[i] // even_part
            E21 = [(E11[i] - E10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E21[i] = (E21[i] * denom_inv) % m
                E21[i] = E21[i] // even_part
            E22 = [(E12[i] - E11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(23)
                denom_inv = inverse_mod(odd_part, m)
                E22[i] = (E22[i] * denom_inv) % m
                E22[i] = E22[i] // even_part

            # layer 3
            E23 = [(E14[i] - E13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E23[i] = (E23[i] * denom_inv) % m
                E23[i] = E23[i] // even_part
            E24 = [(E15[i] - E14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E24[i] = (E24[i] * denom_inv) % m
                E24[i] = E24[i] // even_part
            E25 = [(E16[i] - E15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                E25[i] = (E25[i] * denom_inv) % m
                E25[i] = E25[i] // even_part
            E26 = [(E17[i] - E16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E26[i] = (E26[i] * denom_inv) % m
                E26[i] = E26[i] // even_part
            E27 = [(E18[i] - E17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                E27[i] = (E27[i] * denom_inv) % m
                E27[i] = E27[i] // even_part
            E28 = [(E19[i] - E18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E28[i] = (E28[i] * denom_inv) % m
                E28[i] = E28[i] // even_part
            E29 = [(E20[i] - E19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E29[i] = (E29[i] * denom_inv) % m
                E29[i] = E29[i] // even_part
            E30 = [(E21[i] - E20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E30[i] = (E30[i] * denom_inv) % m
                E30[i] = E30[i] // even_part
            E31 = [(E22[i] - E21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(44)
                denom_inv = inverse_mod(odd_part, m)
                E31[i] = (E31[i] * denom_inv) % m
                E31[i] = E31[i] // even_part

            # layer 4
            E32 = [(E24[i] - E23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E32[i] = (E32[i] * denom_inv) % m
                E32[i] = E32[i] // even_part
            E33 = [(E25[i] - E24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                E33[i] = (E33[i] * denom_inv) % m
                E33[i] = E33[i] // even_part
            E34 = [(E26[i] - E25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                E34[i] = (E34[i] * denom_inv) % m
                E34[i] = E34[i] // even_part
            E35 = [(E27[i] - E26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                E35[i] = (E35[i] * denom_inv) % m
                E35[i] = E35[i] // even_part
            E36 = [(E28[i] - E27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E36[i] = (E36[i] * denom_inv) % m
                E36[i] = E36[i] // even_part
            E37 = [(E29[i] - E28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                E37[i] = (E37[i] * denom_inv) % m
                E37[i] = E37[i] // even_part
            E38 = [(E30[i] - E29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(57)
                denom_inv = inverse_mod(odd_part, m)
                E38[i] = (E38[i] * denom_inv) % m
                E38[i] = E38[i] // even_part
            E39 = [(E31[i] - E30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                E39[i] = (E39[i] * denom_inv) % m
                E39[i] = E39[i] // even_part

            # layer 5
            E40 = [(E33[i] - E32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E40[i] = (E40[i] * denom_inv) % m
                E40[i] = E40[i] // even_part
            E41 = [(E34[i] - E33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E41[i] = (E41[i] * denom_inv) % m
                E41[i] = E41[i] // even_part
            E42 = [(E35[i] - E34[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E42[i] = (E42[i] * denom_inv) % m
                E42[i] = E42[i] // even_part
            E43 = [(E36[i] - E35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                E43[i] = (E43[i] * denom_inv) % m
                E43[i] = E43[i] // even_part
            E44 = [(E37[i] - E36[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E44[i] = (E44[i] * denom_inv) % m
                E44[i] = E44[i] // even_part
            E45 = [(E38[i] - E37[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                E45[i] = (E45[i] * denom_inv) % m
                E45[i] = E45[i] // even_part
            E46 = [(E39[i] - E38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                E46[i] = (E46[i] * denom_inv) % m
                E46[i] = E46[i] // even_part

            # layer 6
            E47 = [(E41[i] - E40[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E47[i] = (E47[i] * denom_inv) % m
                E47[i] = E47[i] // even_part
            E48 = [(E42[i] - E41[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                E48[i] = (E48[i] * denom_inv) % m
                E48[i] = E48[i] // even_part
            E49 = [(E43[i] - E42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                E49[i] = (E49[i] * denom_inv) % m
                E49[i] = E49[i] // even_part
            E50 = [(E44[i] - E43[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                E50[i] = (E50[i] * denom_inv) % m
                E50[i] = E50[i] // even_part
            E51 = [(E45[i] - E44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(85)
                denom_inv = inverse_mod(odd_part, m)
                E51[i] = (E51[i] * denom_inv) % m
                E51[i] = E51[i] // even_part
            E52 = [(E46[i] - E45[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(95)
                denom_inv = inverse_mod(odd_part, m)
                E52[i] = (E52[i] * denom_inv) % m
                E52[i] = E52[i] // even_part

            # layer 7
            E53 = [(E48[i] - E47[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                E53[i] = (E53[i] * denom_inv) % m
                E53[i] = E53[i] // even_part
            E54 = [(E49[i] - E48[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                E54[i] = (E54[i] * denom_inv) % m
                E54[i] = E54[i] // even_part
            E55 = [(E50[i] - E49[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                E55[i] = (E55[i] * denom_inv) % m
                E55[i] = E55[i] // even_part
            E56 = [(E51[i] - E50[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                E56[i] = (E56[i] * denom_inv) % m
                E56[i] = E56[i] // even_part
            E57 = [(E52[i] - E51[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(108)
                denom_inv = inverse_mod(odd_part, m)
                E57[i] = (E57[i] * denom_inv) % m
                E57[i] = E57[i] // even_part

            # layer 8
            E58 = [(E54[i] - E53[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                E58[i] = (E58[i] * denom_inv) % m
                E58[i] = E58[i] // even_part
            E59 = [(E55[i] - E54[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                E59[i] = (E59[i] * denom_inv) % m
                E59[i] = E59[i] // even_part
            E60 = [(E56[i] - E55[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                E60[i] = (E60[i] * denom_inv) % m
                E60[i] = E60[i] // even_part
            E61 = [(E57[i] - E56[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(119)
                denom_inv = inverse_mod(odd_part, m)
                E61[i] = (E61[i] * denom_inv) % m
                E61[i] = E61[i] // even_part

            # layer 9
            E62 = [(E59[i] - E58[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                E62[i] = (E62[i] * denom_inv) % m
                E62[i] = E62[i] // even_part
            E63 = [(E60[i] - E59[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(112)
                denom_inv = inverse_mod(odd_part, m)
                E63[i] = (E63[i] * denom_inv) % m
                E63[i] = E63[i] // even_part
            E64 = [(E61[i] - E60[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(128)
                denom_inv = inverse_mod(odd_part, m)
                E64[i] = (E64[i] * denom_inv) % m
                E64[i] = E64[i] // even_part

            # layer 10
            E65 = [(E63[i] - E62[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(117)
                denom_inv = inverse_mod(odd_part, m)
                E65[i] = (E65[i] * denom_inv) % m
                E65[i] = E65[i] // even_part
            E66 = [(E64[i] - E63[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(135)
                denom_inv = inverse_mod(odd_part, m)
                E66[i] = (E66[i] * denom_inv) % m
                E66[i] = E66[i] // even_part

            # the remaining even variables
            r24 = [(E66[i] - E65[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(140)
                denom_inv = inverse_mod(odd_part, m)
                r24[i] = (r24[i] * denom_inv) % m
                r24[i] = r24[i] // even_part
            r22 = [(E65[i] - 506*r24[i] ) % m for i in range(L)]
            r20 = [(E62[i] - 385*r22[i] - 86779*r24[i] ) % m for i in range(L)]
            r18 = [(E58[i] - 285*r20[i] - 48279*r22[i] - 6369275*r24[i] ) % m for i in range(L)]
            r16 = [(E53[i] - 204*r18[i] - 25194*r20[i] - 2458676*r22[i] - 209609235*r24[i] ) % m for i in range(L)]
            r14 = [(E47[i] - 140*r16[i] - 12138*r18[i] - 846260*r20[i] - 52253971*r22[i] - 2995372800*r24[i] ) % m for i in range(L)]
            r12 = [(E40[i] - 91*r14[i] - 5278*r16[i] - 251498*r18[i] - 10787231*r20[i] - 434928221*r22[i] - 16875270660*r24[i] ) % m for i in range(L)]
            r10 = [(E32[i] - 55*r12[i] - 2002*r14[i] - 61490*r16[i] - 1733303*r18[i] - 46587905*r20[i] - 1217854704*r22[i] - 31306548900*r24[i] ) % m for i in range(L)]
            r8 = [(E23[i] - 30*r10[i] - 627*r12[i] - 11440*r14[i] - 196053*r16[i] - 3255330*r18[i] - 53157079*r20[i] - 860181300*r22[i] - 13850000505*r24[i] ) % m for i in range(L)]
            r6 = [(E13[i] - 14*r8[i] - 147*r10[i] - 1408*r12[i] - 13013*r14[i] - 118482*r16[i] - 1071799*r18[i] - 9668036*r20[i] - 87099705*r22[i] - 784246870*r24[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] - 341*r12[i] - 1365*r14[i] - 5461*r16[i] - 21845*r18[i] - 87381*r20[i] - 349525*r22[i] - 1398101*r24[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] - r12[i] - r14[i] - r16[i] - r18[i] - r20[i] - r22[i] - r24[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r[-6][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(r[7][i] - r[-7][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(14)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
                O7[i] = (O7[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(r[8][i] - r[-8][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
                O8[i] = (O8[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
            O9 = [(r[9][i] - r[-9][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(18)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
                O9[i] = (O9[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(r[10][i] - r[-10][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
                O10[i] = (O10[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
            O11 = [(r[11][i] - r[-11][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(22)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
                O11[i] = (O11[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
            O12 = [(r[12][i] - r[-12][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
                O12[i] = (O12[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(143)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
            O13 = [(r[13][i] - r0[i] - 169*r2[i] - 28561*r4[i] - 4826809*r6[i] - 815730721*r8[i] - 137858491849*r10[i] - 23298085122481*r12[i] - 3937376385699289*r14[i] - 665416609183179841*r16[i] - 112455406951957393129*r18[i] - 19004963774880799438801*r20[i] - 3211838877954855105157369*r22[i] - 542800770374370512771595361*r24[i] - 91733330193268616658399616009*r26[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
                O13[i] = (O13[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(168)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part

            # layer 2
            O14 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part
            O15 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part
            O16 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O16[i] = (O16[i] * denom_inv) % m
                O16[i] = O16[i] // even_part
            O17 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O17[i] = (O17[i] * denom_inv) % m
                O17[i] = O17[i] // even_part
            O18 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O18[i] = (O18[i] * denom_inv) % m
                O18[i] = O18[i] // even_part
            O19 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O19[i] = (O19[i] * denom_inv) % m
                O19[i] = O19[i] // even_part
            O20 = [(O9[i] - O8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                O20[i] = (O20[i] * denom_inv) % m
                O20[i] = O20[i] // even_part
            O21 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                O21[i] = (O21[i] * denom_inv) % m
                O21[i] = O21[i] // even_part
            O22 = [(O11[i] - O10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O22[i] = (O22[i] * denom_inv) % m
                O22[i] = O22[i] // even_part
            O23 = [(O12[i] - O11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(23)
                denom_inv = inverse_mod(odd_part, m)
                O23[i] = (O23[i] * denom_inv) % m
                O23[i] = O23[i] // even_part
            O24 = [(O13[i] - O12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                O24[i] = (O24[i] * denom_inv) % m
                O24[i] = O24[i] // even_part

            # layer 3
            O25 = [(O15[i] - O14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O25[i] = (O25[i] * denom_inv) % m
                O25[i] = O25[i] // even_part
            O26 = [(O16[i] - O15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O26[i] = (O26[i] * denom_inv) % m
                O26[i] = O26[i] // even_part
            O27 = [(O17[i] - O16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O27[i] = (O27[i] * denom_inv) % m
                O27[i] = O27[i] // even_part
            O28 = [(O18[i] - O17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O28[i] = (O28[i] * denom_inv) % m
                O28[i] = O28[i] // even_part
            O29 = [(O19[i] - O18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                O29[i] = (O29[i] * denom_inv) % m
                O29[i] = O29[i] // even_part
            O30 = [(O20[i] - O19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O30[i] = (O30[i] * denom_inv) % m
                O30[i] = O30[i] // even_part
            O31 = [(O21[i] - O20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                O31[i] = (O31[i] * denom_inv) % m
                O31[i] = O31[i] // even_part
            O32 = [(O22[i] - O21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O32[i] = (O32[i] * denom_inv) % m
                O32[i] = O32[i] // even_part
            O33 = [(O23[i] - O22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(44)
                denom_inv = inverse_mod(odd_part, m)
                O33[i] = (O33[i] * denom_inv) % m
                O33[i] = O33[i] // even_part
            O34 = [(O24[i] - O23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O34[i] = (O34[i] * denom_inv) % m
                O34[i] = O34[i] // even_part

            # layer 4
            O35 = [(O26[i] - O25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O35[i] = (O35[i] * denom_inv) % m
                O35[i] = O35[i] // even_part
            O36 = [(O27[i] - O26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O36[i] = (O36[i] * denom_inv) % m
                O36[i] = O36[i] // even_part
            O37 = [(O28[i] - O27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                O37[i] = (O37[i] * denom_inv) % m
                O37[i] = O37[i] // even_part
            O38 = [(O29[i] - O28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                O38[i] = (O38[i] * denom_inv) % m
                O38[i] = O38[i] // even_part
            O39 = [(O30[i] - O29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O39[i] = (O39[i] * denom_inv) % m
                O39[i] = O39[i] // even_part
            O40 = [(O31[i] - O30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                O40[i] = (O40[i] * denom_inv) % m
                O40[i] = O40[i] // even_part
            O41 = [(O32[i] - O31[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(57)
                denom_inv = inverse_mod(odd_part, m)
                O41[i] = (O41[i] * denom_inv) % m
                O41[i] = O41[i] // even_part
            O42 = [(O33[i] - O32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O42[i] = (O42[i] * denom_inv) % m
                O42[i] = O42[i] // even_part
            O43 = [(O34[i] - O33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(69)
                denom_inv = inverse_mod(odd_part, m)
                O43[i] = (O43[i] * denom_inv) % m
                O43[i] = O43[i] // even_part

            # layer 5
            O44 = [(O36[i] - O35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O44[i] = (O44[i] * denom_inv) % m
                O44[i] = O44[i] // even_part
            O45 = [(O37[i] - O36[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O45[i] = (O45[i] * denom_inv) % m
                O45[i] = O45[i] // even_part
            O46 = [(O38[i] - O37[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O46[i] = (O46[i] * denom_inv) % m
                O46[i] = O46[i] // even_part
            O47 = [(O39[i] - O38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                O47[i] = (O47[i] * denom_inv) % m
                O47[i] = O47[i] // even_part
            O48 = [(O40[i] - O39[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                O48[i] = (O48[i] * denom_inv) % m
                O48[i] = O48[i] // even_part
            O49 = [(O41[i] - O40[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O49[i] = (O49[i] * denom_inv) % m
                O49[i] = O49[i] // even_part
            O50 = [(O42[i] - O41[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O50[i] = (O50[i] * denom_inv) % m
                O50[i] = O50[i] // even_part
            O51 = [(O43[i] - O42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(88)
                denom_inv = inverse_mod(odd_part, m)
                O51[i] = (O51[i] * denom_inv) % m
                O51[i] = O51[i] // even_part

            # layer 6
            O52 = [(O45[i] - O44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O52[i] = (O52[i] * denom_inv) % m
                O52[i] = O52[i] // even_part
            O53 = [(O46[i] - O45[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                O53[i] = (O53[i] * denom_inv) % m
                O53[i] = O53[i] // even_part
            O54 = [(O47[i] - O46[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                O54[i] = (O54[i] * denom_inv) % m
                O54[i] = O54[i] // even_part
            O55 = [(O48[i] - O47[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                O55[i] = (O55[i] * denom_inv) % m
                O55[i] = O55[i] // even_part
            O56 = [(O49[i] - O48[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(85)
                denom_inv = inverse_mod(odd_part, m)
                O56[i] = (O56[i] * denom_inv) % m
                O56[i] = O56[i] // even_part
            O57 = [(O50[i] - O49[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(95)
                denom_inv = inverse_mod(odd_part, m)
                O57[i] = (O57[i] * denom_inv) % m
                O57[i] = O57[i] // even_part
            O58 = [(O51[i] - O50[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                O58[i] = (O58[i] * denom_inv) % m
                O58[i] = O58[i] // even_part

            # layer 7
            O59 = [(O53[i] - O52[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                O59[i] = (O59[i] * denom_inv) % m
                O59[i] = O59[i] // even_part
            O60 = [(O54[i] - O53[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O60[i] = (O60[i] * denom_inv) % m
                O60[i] = O60[i] // even_part
            O61 = [(O55[i] - O54[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                O61[i] = (O61[i] * denom_inv) % m
                O61[i] = O61[i] // even_part
            O62 = [(O56[i] - O55[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O62[i] = (O62[i] * denom_inv) % m
                O62[i] = O62[i] // even_part
            O63 = [(O57[i] - O56[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(108)
                denom_inv = inverse_mod(odd_part, m)
                O63[i] = (O63[i] * denom_inv) % m
                O63[i] = O63[i] // even_part
            O64 = [(O58[i] - O57[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                O64[i] = (O64[i] * denom_inv) % m
                O64[i] = O64[i] // even_part

            # layer 8
            O65 = [(O60[i] - O59[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                O65[i] = (O65[i] * denom_inv) % m
                O65[i] = O65[i] // even_part
            O66 = [(O61[i] - O60[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                O66[i] = (O66[i] * denom_inv) % m
                O66[i] = O66[i] // even_part
            O67 = [(O62[i] - O61[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                O67[i] = (O67[i] * denom_inv) % m
                O67[i] = O67[i] // even_part
            O68 = [(O63[i] - O62[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(119)
                denom_inv = inverse_mod(odd_part, m)
                O68[i] = (O68[i] * denom_inv) % m
                O68[i] = O68[i] // even_part
            O69 = [(O64[i] - O63[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(133)
                denom_inv = inverse_mod(odd_part, m)
                O69[i] = (O69[i] * denom_inv) % m
                O69[i] = O69[i] // even_part

            # layer 9
            O70 = [(O66[i] - O65[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O70[i] = (O70[i] * denom_inv) % m
                O70[i] = O70[i] // even_part
            O71 = [(O67[i] - O66[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(112)
                denom_inv = inverse_mod(odd_part, m)
                O71[i] = (O71[i] * denom_inv) % m
                O71[i] = O71[i] // even_part
            O72 = [(O68[i] - O67[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(128)
                denom_inv = inverse_mod(odd_part, m)
                O72[i] = (O72[i] * denom_inv) % m
                O72[i] = O72[i] // even_part
            O73 = [(O69[i] - O68[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(144)
                denom_inv = inverse_mod(odd_part, m)
                O73[i] = (O73[i] * denom_inv) % m
                O73[i] = O73[i] // even_part

            # layer 10
            O74 = [(O71[i] - O70[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(117)
                denom_inv = inverse_mod(odd_part, m)
                O74[i] = (O74[i] * denom_inv) % m
                O74[i] = O74[i] // even_part
            O75 = [(O72[i] - O71[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(135)
                denom_inv = inverse_mod(odd_part, m)
                O75[i] = (O75[i] * denom_inv) % m
                O75[i] = O75[i] // even_part
            O76 = [(O73[i] - O72[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(153)
                denom_inv = inverse_mod(odd_part, m)
                O76[i] = (O76[i] * denom_inv) % m
                O76[i] = O76[i] // even_part

            # layer 11
            O77 = [(O75[i] - O74[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(140)
                denom_inv = inverse_mod(odd_part, m)
                O77[i] = (O77[i] * denom_inv) % m
                O77[i] = O77[i] // even_part
            O78 = [(O76[i] - O75[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(160)
                denom_inv = inverse_mod(odd_part, m)
                O78[i] = (O78[i] * denom_inv) % m
                O78[i] = O78[i] // even_part

            # the remaining odd variables
            r25 = [(O78[i] - O77[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(165)
                denom_inv = inverse_mod(odd_part, m)
                r25[i] = (r25[i] * denom_inv) % m
                r25[i] = r25[i] // even_part
            r23 = [(O77[i] - 650*r25[i] ) % m for i in range(L)]
            r21 = [(O74[i] - 506*r23[i] - 148005*r25[i] ) % m for i in range(L)]
            r19 = [(O70[i] - 385*r21[i] - 86779*r23[i] - 15047175*r25[i] ) % m for i in range(L)]
            r17 = [(O65[i] - 285*r19[i] - 48279*r21[i] - 6369275*r23[i] - 725520510*r25[i] ) % m for i in range(L)]
            r15 = [(O59[i] - 204*r17[i] - 25194*r19[i] - 2458676*r21[i] - 209609235*r23[i] - 16410363840*r25[i] ) % m for i in range(L)]
            r13 = [(O52[i] - 140*r15[i] - 12138*r17[i] - 846260*r19[i] - 52253971*r21[i] - 2995372800*r23[i] - 163648537860*r25[i] ) % m for i in range(L)]
            r11 = [(O44[i] - 91*r13[i] - 5278*r15[i] - 251498*r17[i] - 10787231*r19[i] - 434928221*r21[i] - 16875270660*r23[i] - 638816292660*r25[i] ) % m for i in range(L)]
            r9 = [(O35[i] - 55*r11[i] - 2002*r13[i] - 61490*r15[i] - 1733303*r17[i] - 46587905*r19[i] - 1217854704*r21[i] - 31306548900*r23[i] - 796513723005*r25[i] ) % m for i in range(L)]
            r7 = [(O25[i] - 30*r9[i] - 627*r11[i] - 11440*r13[i] - 196053*r15[i] - 3255330*r17[i] - 53157079*r19[i] - 860181300*r21[i] - 13850000505*r23[i] - 222384254950*r25[i] ) % m for i in range(L)]
            r5 = [(O14[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] - 13013*r13[i] - 118482*r15[i] - 1071799*r17[i] - 9668036*r19[i] - 87099705*r21[i] - 784246870*r23[i] - 7059619931*r25[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] - 1365*r13[i] - 5461*r15[i] - 21845*r17[i] - 87381*r19[i] - 349525*r21[i] - 1398101*r23[i] - 5592405*r25[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] - r13[i] - r15[i] - r17[i] - r19[i] - r21[i] - r23[i] - r25[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26)
        
        if n == 15:
            r0 = r[0]
            r28 = r['infinity']

            L = len(r0)

            # the even temp variables

            # start with E1, since it's not in a layer
            E1 = [(((r[1][i] + r[-1][i]) % m)//2 - r0[i] - r28[i]) % m for i in range(L)]

            # layer 1
            E2 = [((r[2][i] + r[-2][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E2[i] = (E2[i] - r0[i] - 268435456*r28[i]) % m
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
                E2[i] = (E2[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                E2[i] = (E2[i] * denom_inv) % m
                E2[i] = E2[i] // even_part
            E3 = [((r[3][i] + r[-3][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E3[i] = (E3[i] - r0[i] - 22876792454961*r28[i]) % m
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
                E3[i] = (E3[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                E3[i] = (E3[i] * denom_inv) % m
                E3[i] = E3[i] // even_part
            E4 = [((r[4][i] + r[-4][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E4[i] = (E4[i] - r0[i] - 72057594037927936*r28[i]) % m
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
                E4[i] = (E4[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E4[i] = (E4[i] * denom_inv) % m
                E4[i] = E4[i] // even_part
            E5 = [((r[5][i] + r[-5][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E5[i] = (E5[i] - r0[i] - 37252902984619140625*r28[i]) % m
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
                E5[i] = (E5[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E5[i] = (E5[i] * denom_inv) % m
                E5[i] = E5[i] // even_part
            E6 = [((r[6][i] + r[-6][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E6[i] = (E6[i] - r0[i] - 6140942214464815497216*r28[i]) % m
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
                E6[i] = (E6[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                E6[i] = (E6[i] * denom_inv) % m
                E6[i] = E6[i] // even_part
            E7 = [((r[7][i] + r[-7][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E7[i] = (E7[i] - r0[i] - 459986536544739960976801*r28[i]) % m
                odd_part,even_part = split_powers_of_two(49)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
                E7[i] = (E7[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E7[i] = (E7[i] * denom_inv) % m
                E7[i] = E7[i] // even_part
            E8 = [((r[8][i] + r[-8][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E8[i] = (E8[i] - r0[i] - 19342813113834066795298816*r28[i]) % m
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
                E8[i] = (E8[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                E8[i] = (E8[i] * denom_inv) % m
                E8[i] = E8[i] // even_part
            E9 = [((r[9][i] + r[-9][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E9[i] = (E9[i] - r0[i] - 523347633027360537213511521*r28[i]) % m
                odd_part,even_part = split_powers_of_two(81)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
                E9[i] = (E9[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                E9[i] = (E9[i] * denom_inv) % m
                E9[i] = E9[i] // even_part
            E10 = [((r[10][i] + r[-10][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E10[i] = (E10[i] - r0[i] - 10000000000000000000000000000*r28[i]) % m
                odd_part,even_part = split_powers_of_two(100)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
                E10[i] = (E10[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                E10[i] = (E10[i] * denom_inv) % m
                E10[i] = E10[i] // even_part
            E11 = [((r[11][i] + r[-11][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E11[i] = (E11[i] - r0[i] - 144209936106499234037676064081*r28[i]) % m
                odd_part,even_part = split_powers_of_two(121)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
                E11[i] = (E11[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                E11[i] = (E11[i] * denom_inv) % m
                E11[i] = E11[i] // even_part
            E12 = [((r[12][i] + r[-12][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E12[i] = (E12[i] - r0[i] - 1648446623609512543951043690496*r28[i]) % m
                odd_part,even_part = split_powers_of_two(144)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part
                E12[i] = (E12[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(143)
                denom_inv = inverse_mod(odd_part, m)
                E12[i] = (E12[i] * denom_inv) % m
                E12[i] = E12[i] // even_part
            E13 = [((r[13][i] + r[-13][i]) % m)//2 for i in range(L)]
            for i in range(L):
                E13[i] = (E13[i] - r0[i] - 15502932802662396215269535105521*r28[i]) % m
                odd_part,even_part = split_powers_of_two(169)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part
                E13[i] = (E13[i] - E1[i]) % m
                odd_part,even_part = split_powers_of_two(168)
                denom_inv = inverse_mod(odd_part, m)
                E13[i] = (E13[i] * denom_inv) % m
                E13[i] = E13[i] // even_part

            # layer 2
            E14 = [(E3[i] - E2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                E14[i] = (E14[i] * denom_inv) % m
                E14[i] = E14[i] // even_part
            E15 = [(E4[i] - E3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                E15[i] = (E15[i] * denom_inv) % m
                E15[i] = E15[i] // even_part
            E16 = [(E5[i] - E4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                E16[i] = (E16[i] * denom_inv) % m
                E16[i] = E16[i] // even_part
            E17 = [(E6[i] - E5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                E17[i] = (E17[i] * denom_inv) % m
                E17[i] = E17[i] // even_part
            E18 = [(E7[i] - E6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                E18[i] = (E18[i] * denom_inv) % m
                E18[i] = E18[i] // even_part
            E19 = [(E8[i] - E7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                E19[i] = (E19[i] * denom_inv) % m
                E19[i] = E19[i] // even_part
            E20 = [(E9[i] - E8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                E20[i] = (E20[i] * denom_inv) % m
                E20[i] = E20[i] // even_part
            E21 = [(E10[i] - E9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                E21[i] = (E21[i] * denom_inv) % m
                E21[i] = E21[i] // even_part
            E22 = [(E11[i] - E10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E22[i] = (E22[i] * denom_inv) % m
                E22[i] = E22[i] // even_part
            E23 = [(E12[i] - E11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(23)
                denom_inv = inverse_mod(odd_part, m)
                E23[i] = (E23[i] * denom_inv) % m
                E23[i] = E23[i] // even_part
            E24 = [(E13[i] - E12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                E24[i] = (E24[i] * denom_inv) % m
                E24[i] = E24[i] // even_part

            # layer 3
            E25 = [(E15[i] - E14[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                E25[i] = (E25[i] * denom_inv) % m
                E25[i] = E25[i] // even_part
            E26 = [(E16[i] - E15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                E26[i] = (E26[i] * denom_inv) % m
                E26[i] = E26[i] // even_part
            E27 = [(E17[i] - E16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                E27[i] = (E27[i] * denom_inv) % m
                E27[i] = E27[i] // even_part
            E28 = [(E18[i] - E17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                E28[i] = (E28[i] * denom_inv) % m
                E28[i] = E28[i] // even_part
            E29 = [(E19[i] - E18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                E29[i] = (E29[i] * denom_inv) % m
                E29[i] = E29[i] // even_part
            E30 = [(E20[i] - E19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E30[i] = (E30[i] * denom_inv) % m
                E30[i] = E30[i] // even_part
            E31 = [(E21[i] - E20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                E31[i] = (E31[i] * denom_inv) % m
                E31[i] = E31[i] // even_part
            E32 = [(E22[i] - E21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E32[i] = (E32[i] * denom_inv) % m
                E32[i] = E32[i] // even_part
            E33 = [(E23[i] - E22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(44)
                denom_inv = inverse_mod(odd_part, m)
                E33[i] = (E33[i] * denom_inv) % m
                E33[i] = E33[i] // even_part
            E34 = [(E24[i] - E23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E34[i] = (E34[i] * denom_inv) % m
                E34[i] = E34[i] // even_part

            # layer 4
            E35 = [(E26[i] - E25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                E35[i] = (E35[i] * denom_inv) % m
                E35[i] = E35[i] // even_part
            E36 = [(E27[i] - E26[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                E36[i] = (E36[i] * denom_inv) % m
                E36[i] = E36[i] // even_part
            E37 = [(E28[i] - E27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                E37[i] = (E37[i] * denom_inv) % m
                E37[i] = E37[i] // even_part
            E38 = [(E29[i] - E28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                E38[i] = (E38[i] * denom_inv) % m
                E38[i] = E38[i] // even_part
            E39 = [(E30[i] - E29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E39[i] = (E39[i] * denom_inv) % m
                E39[i] = E39[i] // even_part
            E40 = [(E31[i] - E30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                E40[i] = (E40[i] * denom_inv) % m
                E40[i] = E40[i] // even_part
            E41 = [(E32[i] - E31[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(57)
                denom_inv = inverse_mod(odd_part, m)
                E41[i] = (E41[i] * denom_inv) % m
                E41[i] = E41[i] // even_part
            E42 = [(E33[i] - E32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                E42[i] = (E42[i] * denom_inv) % m
                E42[i] = E42[i] // even_part
            E43 = [(E34[i] - E33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(69)
                denom_inv = inverse_mod(odd_part, m)
                E43[i] = (E43[i] * denom_inv) % m
                E43[i] = E43[i] // even_part

            # layer 5
            E44 = [(E36[i] - E35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                E44[i] = (E44[i] * denom_inv) % m
                E44[i] = E44[i] // even_part
            E45 = [(E37[i] - E36[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                E45[i] = (E45[i] * denom_inv) % m
                E45[i] = E45[i] // even_part
            E46 = [(E38[i] - E37[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                E46[i] = (E46[i] * denom_inv) % m
                E46[i] = E46[i] // even_part
            E47 = [(E39[i] - E38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                E47[i] = (E47[i] * denom_inv) % m
                E47[i] = E47[i] // even_part
            E48 = [(E40[i] - E39[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                E48[i] = (E48[i] * denom_inv) % m
                E48[i] = E48[i] // even_part
            E49 = [(E41[i] - E40[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                E49[i] = (E49[i] * denom_inv) % m
                E49[i] = E49[i] // even_part
            E50 = [(E42[i] - E41[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                E50[i] = (E50[i] * denom_inv) % m
                E50[i] = E50[i] // even_part
            E51 = [(E43[i] - E42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(88)
                denom_inv = inverse_mod(odd_part, m)
                E51[i] = (E51[i] * denom_inv) % m
                E51[i] = E51[i] // even_part

            # layer 6
            E52 = [(E45[i] - E44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                E52[i] = (E52[i] * denom_inv) % m
                E52[i] = E52[i] // even_part
            E53 = [(E46[i] - E45[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                E53[i] = (E53[i] * denom_inv) % m
                E53[i] = E53[i] // even_part
            E54 = [(E47[i] - E46[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                E54[i] = (E54[i] * denom_inv) % m
                E54[i] = E54[i] // even_part
            E55 = [(E48[i] - E47[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                E55[i] = (E55[i] * denom_inv) % m
                E55[i] = E55[i] // even_part
            E56 = [(E49[i] - E48[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(85)
                denom_inv = inverse_mod(odd_part, m)
                E56[i] = (E56[i] * denom_inv) % m
                E56[i] = E56[i] // even_part
            E57 = [(E50[i] - E49[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(95)
                denom_inv = inverse_mod(odd_part, m)
                E57[i] = (E57[i] * denom_inv) % m
                E57[i] = E57[i] // even_part
            E58 = [(E51[i] - E50[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                E58[i] = (E58[i] * denom_inv) % m
                E58[i] = E58[i] // even_part

            # layer 7
            E59 = [(E53[i] - E52[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                E59[i] = (E59[i] * denom_inv) % m
                E59[i] = E59[i] // even_part
            E60 = [(E54[i] - E53[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                E60[i] = (E60[i] * denom_inv) % m
                E60[i] = E60[i] // even_part
            E61 = [(E55[i] - E54[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                E61[i] = (E61[i] * denom_inv) % m
                E61[i] = E61[i] // even_part
            E62 = [(E56[i] - E55[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                E62[i] = (E62[i] * denom_inv) % m
                E62[i] = E62[i] // even_part
            E63 = [(E57[i] - E56[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(108)
                denom_inv = inverse_mod(odd_part, m)
                E63[i] = (E63[i] * denom_inv) % m
                E63[i] = E63[i] // even_part
            E64 = [(E58[i] - E57[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                E64[i] = (E64[i] * denom_inv) % m
                E64[i] = E64[i] // even_part

            # layer 8
            E65 = [(E60[i] - E59[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                E65[i] = (E65[i] * denom_inv) % m
                E65[i] = E65[i] // even_part
            E66 = [(E61[i] - E60[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                E66[i] = (E66[i] * denom_inv) % m
                E66[i] = E66[i] // even_part
            E67 = [(E62[i] - E61[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                E67[i] = (E67[i] * denom_inv) % m
                E67[i] = E67[i] // even_part
            E68 = [(E63[i] - E62[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(119)
                denom_inv = inverse_mod(odd_part, m)
                E68[i] = (E68[i] * denom_inv) % m
                E68[i] = E68[i] // even_part
            E69 = [(E64[i] - E63[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(133)
                denom_inv = inverse_mod(odd_part, m)
                E69[i] = (E69[i] * denom_inv) % m
                E69[i] = E69[i] // even_part

            # layer 9
            E70 = [(E66[i] - E65[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                E70[i] = (E70[i] * denom_inv) % m
                E70[i] = E70[i] // even_part
            E71 = [(E67[i] - E66[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(112)
                denom_inv = inverse_mod(odd_part, m)
                E71[i] = (E71[i] * denom_inv) % m
                E71[i] = E71[i] // even_part
            E72 = [(E68[i] - E67[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(128)
                denom_inv = inverse_mod(odd_part, m)
                E72[i] = (E72[i] * denom_inv) % m
                E72[i] = E72[i] // even_part
            E73 = [(E69[i] - E68[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(144)
                denom_inv = inverse_mod(odd_part, m)
                E73[i] = (E73[i] * denom_inv) % m
                E73[i] = E73[i] // even_part

            # layer 10
            E74 = [(E71[i] - E70[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(117)
                denom_inv = inverse_mod(odd_part, m)
                E74[i] = (E74[i] * denom_inv) % m
                E74[i] = E74[i] // even_part
            E75 = [(E72[i] - E71[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(135)
                denom_inv = inverse_mod(odd_part, m)
                E75[i] = (E75[i] * denom_inv) % m
                E75[i] = E75[i] // even_part
            E76 = [(E73[i] - E72[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(153)
                denom_inv = inverse_mod(odd_part, m)
                E76[i] = (E76[i] * denom_inv) % m
                E76[i] = E76[i] // even_part

            # layer 11
            E77 = [(E75[i] - E74[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(140)
                denom_inv = inverse_mod(odd_part, m)
                E77[i] = (E77[i] * denom_inv) % m
                E77[i] = E77[i] // even_part
            E78 = [(E76[i] - E75[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(160)
                denom_inv = inverse_mod(odd_part, m)
                E78[i] = (E78[i] * denom_inv) % m
                E78[i] = E78[i] // even_part

            # the remaining even variables
            r26 = [(E78[i] - E77[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(165)
                denom_inv = inverse_mod(odd_part, m)
                r26[i] = (r26[i] * denom_inv) % m
                r26[i] = r26[i] // even_part
            r24 = [(E77[i] - 650*r26[i] ) % m for i in range(L)]
            r22 = [(E74[i] - 506*r24[i] - 148005*r26[i] ) % m for i in range(L)]
            r20 = [(E70[i] - 385*r22[i] - 86779*r24[i] - 15047175*r26[i] ) % m for i in range(L)]
            r18 = [(E65[i] - 285*r20[i] - 48279*r22[i] - 6369275*r24[i] - 725520510*r26[i] ) % m for i in range(L)]
            r16 = [(E59[i] - 204*r18[i] - 25194*r20[i] - 2458676*r22[i] - 209609235*r24[i] - 16410363840*r26[i] ) % m for i in range(L)]
            r14 = [(E52[i] - 140*r16[i] - 12138*r18[i] - 846260*r20[i] - 52253971*r22[i] - 2995372800*r24[i] - 163648537860*r26[i] ) % m for i in range(L)]
            r12 = [(E44[i] - 91*r14[i] - 5278*r16[i] - 251498*r18[i] - 10787231*r20[i] - 434928221*r22[i] - 16875270660*r24[i] - 638816292660*r26[i] ) % m for i in range(L)]
            r10 = [(E35[i] - 55*r12[i] - 2002*r14[i] - 61490*r16[i] - 1733303*r18[i] - 46587905*r20[i] - 1217854704*r22[i] - 31306548900*r24[i] - 796513723005*r26[i] ) % m for i in range(L)]
            r8 = [(E25[i] - 30*r10[i] - 627*r12[i] - 11440*r14[i] - 196053*r16[i] - 3255330*r18[i] - 53157079*r20[i] - 860181300*r22[i] - 13850000505*r24[i] - 222384254950*r26[i] ) % m for i in range(L)]
            r6 = [(E14[i] - 14*r8[i] - 147*r10[i] - 1408*r12[i] - 13013*r14[i] - 118482*r16[i] - 1071799*r18[i] - 9668036*r20[i] - 87099705*r22[i] - 784246870*r24[i] - 7059619931*r26[i] ) % m for i in range(L)]
            r4 = [(E2[i] - 5*r6[i] - 21*r8[i] - 85*r10[i] - 341*r12[i] - 1365*r14[i] - 5461*r16[i] - 21845*r18[i] - 87381*r20[i] - 349525*r22[i] - 1398101*r24[i] - 5592405*r26[i] ) % m for i in range(L)]
            r2 = [(E1[i] - r4[i] - r6[i] - r8[i] - r10[i] - r12[i] - r14[i] - r16[i] - r18[i] - r20[i] - r22[i] - r24[i] - r26[i] ) % m for i in range(L)]

            # the odd temp variables

            # start with O1, since it's not in a layer
            O1 = [((r[1][i] - r[-1][i]) % m)//2 for i in range(L)]

            # layer 1
            O2 = [(r[2][i] - r[-2][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(4)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
                O2[i] = (O2[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(3)
                denom_inv = inverse_mod(odd_part, m)
                O2[i] = (O2[i] * denom_inv) % m
                O2[i] = O2[i] // even_part
            O3 = [(r[3][i] - r[-3][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(6)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
                O3[i] = (O3[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O3[i] = (O3[i] * denom_inv) % m
                O3[i] = O3[i] // even_part
            O4 = [(r[4][i] - r[-4][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(8)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
                O4[i] = (O4[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O4[i] = (O4[i] * denom_inv) % m
                O4[i] = O4[i] // even_part
            O5 = [(r[5][i] - r[-5][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(10)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
                O5[i] = (O5[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O5[i] = (O5[i] * denom_inv) % m
                O5[i] = O5[i] // even_part
            O6 = [(r[6][i] - r[-6][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
                O6[i] = (O6[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(35)
                denom_inv = inverse_mod(odd_part, m)
                O6[i] = (O6[i] * denom_inv) % m
                O6[i] = O6[i] // even_part
            O7 = [(r[7][i] - r[-7][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(14)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
                O7[i] = (O7[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O7[i] = (O7[i] * denom_inv) % m
                O7[i] = O7[i] // even_part
            O8 = [(r[8][i] - r[-8][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
                O8[i] = (O8[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O8[i] = (O8[i] * denom_inv) % m
                O8[i] = O8[i] // even_part
            O9 = [(r[9][i] - r[-9][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(18)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
                O9[i] = (O9[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O9[i] = (O9[i] * denom_inv) % m
                O9[i] = O9[i] // even_part
            O10 = [(r[10][i] - r[-10][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
                O10[i] = (O10[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(99)
                denom_inv = inverse_mod(odd_part, m)
                O10[i] = (O10[i] * denom_inv) % m
                O10[i] = O10[i] // even_part
            O11 = [(r[11][i] - r[-11][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(22)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
                O11[i] = (O11[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                O11[i] = (O11[i] * denom_inv) % m
                O11[i] = O11[i] // even_part
            O12 = [(r[12][i] - r[-12][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
                O12[i] = (O12[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(143)
                denom_inv = inverse_mod(odd_part, m)
                O12[i] = (O12[i] * denom_inv) % m
                O12[i] = O12[i] // even_part
            O13 = [(r[13][i] - r[-13][i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(26)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
                O13[i] = (O13[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(168)
                denom_inv = inverse_mod(odd_part, m)
                O13[i] = (O13[i] * denom_inv) % m
                O13[i] = O13[i] // even_part
            O14 = [(r[14][i] - r0[i] - 196*r2[i] - 38416*r4[i] - 7529536*r6[i] - 1475789056*r8[i] - 289254654976*r10[i] - 56693912375296*r12[i] - 11112006825558016*r14[i] - 2177953337809371136*r16[i] - 426878854210636742656*r18[i] - 83668255425284801560576*r20[i] - 16398978063355821105872896*r22[i] - 3214199700417740936751087616*r24[i] - 629983141281877223603213172736*r26[i] - 123476695691247935826229781856256*r28[i] ) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(14)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part
                O14[i] = (O14[i] - O1[i]) % m
                odd_part,even_part = split_powers_of_two(195)
                denom_inv = inverse_mod(odd_part, m)
                O14[i] = (O14[i] * denom_inv) % m
                O14[i] = O14[i] // even_part

            # layer 2
            O15 = [(O3[i] - O2[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(5)
                denom_inv = inverse_mod(odd_part, m)
                O15[i] = (O15[i] * denom_inv) % m
                O15[i] = O15[i] // even_part
            O16 = [(O4[i] - O3[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(7)
                denom_inv = inverse_mod(odd_part, m)
                O16[i] = (O16[i] * denom_inv) % m
                O16[i] = O16[i] // even_part
            O17 = [(O5[i] - O4[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(9)
                denom_inv = inverse_mod(odd_part, m)
                O17[i] = (O17[i] * denom_inv) % m
                O17[i] = O17[i] // even_part
            O18 = [(O6[i] - O5[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(11)
                denom_inv = inverse_mod(odd_part, m)
                O18[i] = (O18[i] * denom_inv) % m
                O18[i] = O18[i] // even_part
            O19 = [(O7[i] - O6[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(13)
                denom_inv = inverse_mod(odd_part, m)
                O19[i] = (O19[i] * denom_inv) % m
                O19[i] = O19[i] // even_part
            O20 = [(O8[i] - O7[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(15)
                denom_inv = inverse_mod(odd_part, m)
                O20[i] = (O20[i] * denom_inv) % m
                O20[i] = O20[i] // even_part
            O21 = [(O9[i] - O8[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(17)
                denom_inv = inverse_mod(odd_part, m)
                O21[i] = (O21[i] * denom_inv) % m
                O21[i] = O21[i] // even_part
            O22 = [(O10[i] - O9[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(19)
                denom_inv = inverse_mod(odd_part, m)
                O22[i] = (O22[i] * denom_inv) % m
                O22[i] = O22[i] // even_part
            O23 = [(O11[i] - O10[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O23[i] = (O23[i] * denom_inv) % m
                O23[i] = O23[i] // even_part
            O24 = [(O12[i] - O11[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(23)
                denom_inv = inverse_mod(odd_part, m)
                O24[i] = (O24[i] * denom_inv) % m
                O24[i] = O24[i] // even_part
            O25 = [(O13[i] - O12[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(25)
                denom_inv = inverse_mod(odd_part, m)
                O25[i] = (O25[i] * denom_inv) % m
                O25[i] = O25[i] // even_part
            O26 = [(O14[i] - O13[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O26[i] = (O26[i] * denom_inv) % m
                O26[i] = O26[i] // even_part

            # layer 3
            O27 = [(O16[i] - O15[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(12)
                denom_inv = inverse_mod(odd_part, m)
                O27[i] = (O27[i] * denom_inv) % m
                O27[i] = O27[i] // even_part
            O28 = [(O17[i] - O16[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(16)
                denom_inv = inverse_mod(odd_part, m)
                O28[i] = (O28[i] * denom_inv) % m
                O28[i] = O28[i] // even_part
            O29 = [(O18[i] - O17[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(20)
                denom_inv = inverse_mod(odd_part, m)
                O29[i] = (O29[i] * denom_inv) % m
                O29[i] = O29[i] // even_part
            O30 = [(O19[i] - O18[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(24)
                denom_inv = inverse_mod(odd_part, m)
                O30[i] = (O30[i] * denom_inv) % m
                O30[i] = O30[i] // even_part
            O31 = [(O20[i] - O19[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(28)
                denom_inv = inverse_mod(odd_part, m)
                O31[i] = (O31[i] * denom_inv) % m
                O31[i] = O31[i] // even_part
            O32 = [(O21[i] - O20[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O32[i] = (O32[i] * denom_inv) % m
                O32[i] = O32[i] // even_part
            O33 = [(O22[i] - O21[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(36)
                denom_inv = inverse_mod(odd_part, m)
                O33[i] = (O33[i] * denom_inv) % m
                O33[i] = O33[i] // even_part
            O34 = [(O23[i] - O22[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O34[i] = (O34[i] * denom_inv) % m
                O34[i] = O34[i] // even_part
            O35 = [(O24[i] - O23[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(44)
                denom_inv = inverse_mod(odd_part, m)
                O35[i] = (O35[i] * denom_inv) % m
                O35[i] = O35[i] // even_part
            O36 = [(O25[i] - O24[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O36[i] = (O36[i] * denom_inv) % m
                O36[i] = O36[i] // even_part
            O37 = [(O26[i] - O25[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(52)
                denom_inv = inverse_mod(odd_part, m)
                O37[i] = (O37[i] * denom_inv) % m
                O37[i] = O37[i] // even_part

            # layer 4
            O38 = [(O28[i] - O27[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(21)
                denom_inv = inverse_mod(odd_part, m)
                O38[i] = (O38[i] * denom_inv) % m
                O38[i] = O38[i] // even_part
            O39 = [(O29[i] - O28[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(27)
                denom_inv = inverse_mod(odd_part, m)
                O39[i] = (O39[i] * denom_inv) % m
                O39[i] = O39[i] // even_part
            O40 = [(O30[i] - O29[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(33)
                denom_inv = inverse_mod(odd_part, m)
                O40[i] = (O40[i] * denom_inv) % m
                O40[i] = O40[i] // even_part
            O41 = [(O31[i] - O30[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(39)
                denom_inv = inverse_mod(odd_part, m)
                O41[i] = (O41[i] * denom_inv) % m
                O41[i] = O41[i] // even_part
            O42 = [(O32[i] - O31[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O42[i] = (O42[i] * denom_inv) % m
                O42[i] = O42[i] // even_part
            O43 = [(O33[i] - O32[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(51)
                denom_inv = inverse_mod(odd_part, m)
                O43[i] = (O43[i] * denom_inv) % m
                O43[i] = O43[i] // even_part
            O44 = [(O34[i] - O33[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(57)
                denom_inv = inverse_mod(odd_part, m)
                O44[i] = (O44[i] * denom_inv) % m
                O44[i] = O44[i] // even_part
            O45 = [(O35[i] - O34[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(63)
                denom_inv = inverse_mod(odd_part, m)
                O45[i] = (O45[i] * denom_inv) % m
                O45[i] = O45[i] // even_part
            O46 = [(O36[i] - O35[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(69)
                denom_inv = inverse_mod(odd_part, m)
                O46[i] = (O46[i] * denom_inv) % m
                O46[i] = O46[i] // even_part
            O47 = [(O37[i] - O36[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                O47[i] = (O47[i] * denom_inv) % m
                O47[i] = O47[i] // even_part

            # layer 5
            O48 = [(O39[i] - O38[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(32)
                denom_inv = inverse_mod(odd_part, m)
                O48[i] = (O48[i] * denom_inv) % m
                O48[i] = O48[i] // even_part
            O49 = [(O40[i] - O39[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(40)
                denom_inv = inverse_mod(odd_part, m)
                O49[i] = (O49[i] * denom_inv) % m
                O49[i] = O49[i] // even_part
            O50 = [(O41[i] - O40[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(48)
                denom_inv = inverse_mod(odd_part, m)
                O50[i] = (O50[i] * denom_inv) % m
                O50[i] = O50[i] // even_part
            O51 = [(O42[i] - O41[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(56)
                denom_inv = inverse_mod(odd_part, m)
                O51[i] = (O51[i] * denom_inv) % m
                O51[i] = O51[i] // even_part
            O52 = [(O43[i] - O42[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(64)
                denom_inv = inverse_mod(odd_part, m)
                O52[i] = (O52[i] * denom_inv) % m
                O52[i] = O52[i] // even_part
            O53 = [(O44[i] - O43[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O53[i] = (O53[i] * denom_inv) % m
                O53[i] = O53[i] // even_part
            O54 = [(O45[i] - O44[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(80)
                denom_inv = inverse_mod(odd_part, m)
                O54[i] = (O54[i] * denom_inv) % m
                O54[i] = O54[i] // even_part
            O55 = [(O46[i] - O45[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(88)
                denom_inv = inverse_mod(odd_part, m)
                O55[i] = (O55[i] * denom_inv) % m
                O55[i] = O55[i] // even_part
            O56 = [(O47[i] - O46[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O56[i] = (O56[i] * denom_inv) % m
                O56[i] = O56[i] // even_part

            # layer 6
            O57 = [(O49[i] - O48[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(45)
                denom_inv = inverse_mod(odd_part, m)
                O57[i] = (O57[i] * denom_inv) % m
                O57[i] = O57[i] // even_part
            O58 = [(O50[i] - O49[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(55)
                denom_inv = inverse_mod(odd_part, m)
                O58[i] = (O58[i] * denom_inv) % m
                O58[i] = O58[i] // even_part
            O59 = [(O51[i] - O50[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(65)
                denom_inv = inverse_mod(odd_part, m)
                O59[i] = (O59[i] * denom_inv) % m
                O59[i] = O59[i] // even_part
            O60 = [(O52[i] - O51[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(75)
                denom_inv = inverse_mod(odd_part, m)
                O60[i] = (O60[i] * denom_inv) % m
                O60[i] = O60[i] // even_part
            O61 = [(O53[i] - O52[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(85)
                denom_inv = inverse_mod(odd_part, m)
                O61[i] = (O61[i] * denom_inv) % m
                O61[i] = O61[i] // even_part
            O62 = [(O54[i] - O53[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(95)
                denom_inv = inverse_mod(odd_part, m)
                O62[i] = (O62[i] * denom_inv) % m
                O62[i] = O62[i] // even_part
            O63 = [(O55[i] - O54[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                O63[i] = (O63[i] * denom_inv) % m
                O63[i] = O63[i] // even_part
            O64 = [(O56[i] - O55[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(115)
                denom_inv = inverse_mod(odd_part, m)
                O64[i] = (O64[i] * denom_inv) % m
                O64[i] = O64[i] // even_part

            # layer 7
            O65 = [(O58[i] - O57[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(60)
                denom_inv = inverse_mod(odd_part, m)
                O65[i] = (O65[i] * denom_inv) % m
                O65[i] = O65[i] // even_part
            O66 = [(O59[i] - O58[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(72)
                denom_inv = inverse_mod(odd_part, m)
                O66[i] = (O66[i] * denom_inv) % m
                O66[i] = O66[i] // even_part
            O67 = [(O60[i] - O59[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(84)
                denom_inv = inverse_mod(odd_part, m)
                O67[i] = (O67[i] * denom_inv) % m
                O67[i] = O67[i] // even_part
            O68 = [(O61[i] - O60[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O68[i] = (O68[i] * denom_inv) % m
                O68[i] = O68[i] // even_part
            O69 = [(O62[i] - O61[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(108)
                denom_inv = inverse_mod(odd_part, m)
                O69[i] = (O69[i] * denom_inv) % m
                O69[i] = O69[i] // even_part
            O70 = [(O63[i] - O62[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(120)
                denom_inv = inverse_mod(odd_part, m)
                O70[i] = (O70[i] * denom_inv) % m
                O70[i] = O70[i] // even_part
            O71 = [(O64[i] - O63[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(132)
                denom_inv = inverse_mod(odd_part, m)
                O71[i] = (O71[i] * denom_inv) % m
                O71[i] = O71[i] // even_part

            # layer 8
            O72 = [(O66[i] - O65[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(77)
                denom_inv = inverse_mod(odd_part, m)
                O72[i] = (O72[i] * denom_inv) % m
                O72[i] = O72[i] // even_part
            O73 = [(O67[i] - O66[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(91)
                denom_inv = inverse_mod(odd_part, m)
                O73[i] = (O73[i] * denom_inv) % m
                O73[i] = O73[i] // even_part
            O74 = [(O68[i] - O67[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(105)
                denom_inv = inverse_mod(odd_part, m)
                O74[i] = (O74[i] * denom_inv) % m
                O74[i] = O74[i] // even_part
            O75 = [(O69[i] - O68[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(119)
                denom_inv = inverse_mod(odd_part, m)
                O75[i] = (O75[i] * denom_inv) % m
                O75[i] = O75[i] // even_part
            O76 = [(O70[i] - O69[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(133)
                denom_inv = inverse_mod(odd_part, m)
                O76[i] = (O76[i] * denom_inv) % m
                O76[i] = O76[i] // even_part
            O77 = [(O71[i] - O70[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(147)
                denom_inv = inverse_mod(odd_part, m)
                O77[i] = (O77[i] * denom_inv) % m
                O77[i] = O77[i] // even_part

            # layer 9
            O78 = [(O73[i] - O72[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(96)
                denom_inv = inverse_mod(odd_part, m)
                O78[i] = (O78[i] * denom_inv) % m
                O78[i] = O78[i] // even_part
            O79 = [(O74[i] - O73[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(112)
                denom_inv = inverse_mod(odd_part, m)
                O79[i] = (O79[i] * denom_inv) % m
                O79[i] = O79[i] // even_part
            O80 = [(O75[i] - O74[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(128)
                denom_inv = inverse_mod(odd_part, m)
                O80[i] = (O80[i] * denom_inv) % m
                O80[i] = O80[i] // even_part
            O81 = [(O76[i] - O75[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(144)
                denom_inv = inverse_mod(odd_part, m)
                O81[i] = (O81[i] * denom_inv) % m
                O81[i] = O81[i] // even_part
            O82 = [(O77[i] - O76[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(160)
                denom_inv = inverse_mod(odd_part, m)
                O82[i] = (O82[i] * denom_inv) % m
                O82[i] = O82[i] // even_part

            # layer 10
            O83 = [(O79[i] - O78[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(117)
                denom_inv = inverse_mod(odd_part, m)
                O83[i] = (O83[i] * denom_inv) % m
                O83[i] = O83[i] // even_part
            O84 = [(O80[i] - O79[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(135)
                denom_inv = inverse_mod(odd_part, m)
                O84[i] = (O84[i] * denom_inv) % m
                O84[i] = O84[i] // even_part
            O85 = [(O81[i] - O80[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(153)
                denom_inv = inverse_mod(odd_part, m)
                O85[i] = (O85[i] * denom_inv) % m
                O85[i] = O85[i] // even_part
            O86 = [(O82[i] - O81[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(171)
                denom_inv = inverse_mod(odd_part, m)
                O86[i] = (O86[i] * denom_inv) % m
                O86[i] = O86[i] // even_part

            # layer 11
            O87 = [(O84[i] - O83[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(140)
                denom_inv = inverse_mod(odd_part, m)
                O87[i] = (O87[i] * denom_inv) % m
                O87[i] = O87[i] // even_part
            O88 = [(O85[i] - O84[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(160)
                denom_inv = inverse_mod(odd_part, m)
                O88[i] = (O88[i] * denom_inv) % m
                O88[i] = O88[i] // even_part
            O89 = [(O86[i] - O85[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(180)
                denom_inv = inverse_mod(odd_part, m)
                O89[i] = (O89[i] * denom_inv) % m
                O89[i] = O89[i] // even_part

            # layer 12
            O90 = [(O88[i] - O87[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(165)
                denom_inv = inverse_mod(odd_part, m)
                O90[i] = (O90[i] * denom_inv) % m
                O90[i] = O90[i] // even_part
            O91 = [(O89[i] - O88[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(187)
                denom_inv = inverse_mod(odd_part, m)
                O91[i] = (O91[i] * denom_inv) % m
                O91[i] = O91[i] // even_part

            # the remaining odd variables
            r27 = [(O91[i] - O90[i]) % m for i in range(L)]
            for i in range(L):
                odd_part,even_part = split_powers_of_two(192)
                denom_inv = inverse_mod(odd_part, m)
                r27[i] = (r27[i] * denom_inv) % m
                r27[i] = r27[i] // even_part
            r25 = [(O90[i] - 819*r27[i] ) % m for i in range(L)]
            r23 = [(O87[i] - 650*r25[i] - 241605*r27[i] ) % m for i in range(L)]
            r21 = [(O83[i] - 506*r23[i] - 148005*r25[i] - 32955780*r27[i] ) % m for i in range(L)]
            r19 = [(O78[i] - 385*r21[i] - 86779*r23[i] - 15047175*r25[i] - 2230238010*r27[i] ) % m for i in range(L)]
            r17 = [(O72[i] - 285*r19[i] - 48279*r21[i] - 6369275*r23[i] - 725520510*r25[i] - 75177525150*r27[i] ) % m for i in range(L)]
            r15 = [(O65[i] - 204*r17[i] - 25194*r19[i] - 2458676*r21[i] - 209609235*r23[i] - 16410363840*r25[i] - 1213911823620*r27[i] ) % m for i in range(L)]
            r13 = [(O57[i] - 140*r15[i] - 12138*r17[i] - 846260*r19[i] - 52253971*r21[i] - 2995372800*r23[i] - 163648537860*r25[i] - 8657594647800*r27[i] ) % m for i in range(L)]
            r11 = [(O48[i] - 91*r13[i] - 5278*r15[i] - 251498*r17[i] - 10787231*r19[i] - 434928221*r21[i] - 16875270660*r23[i] - 638816292660*r25[i] - 23793900258765*r27[i] ) % m for i in range(L)]
            r9 = [(O38[i] - 55*r11[i] - 2002*r13[i] - 61490*r15[i] - 1733303*r17[i] - 46587905*r19[i] - 1217854704*r21[i] - 31306548900*r23[i] - 796513723005*r25[i] - 20135227330075*r27[i] ) % m for i in range(L)]
            r7 = [(O27[i] - 30*r9[i] - 627*r11[i] - 11440*r13[i] - 196053*r15[i] - 3255330*r17[i] - 53157079*r19[i] - 860181300*r21[i] - 13850000505*r23[i] - 222384254950*r25[i] - 3565207699131*r27[i] ) % m for i in range(L)]
            r5 = [(O15[i] - 14*r7[i] - 147*r9[i] - 1408*r11[i] - 13013*r13[i] - 118482*r15[i] - 1071799*r17[i] - 9668036*r19[i] - 87099705*r21[i] - 784246870*r23[i] - 7059619931*r25[i] - 63542171784*r27[i] ) % m for i in range(L)]
            r3 = [(O2[i] - 5*r5[i] - 21*r7[i] - 85*r9[i] - 341*r11[i] - 1365*r13[i] - 5461*r15[i] - 21845*r17[i] - 87381*r19[i] - 349525*r21[i] - 1398101*r23[i] - 5592405*r25[i] - 22369621*r27[i] ) % m for i in range(L)]
            r1 = [(O1[i] - r3[i] - r5[i] - r7[i] - r9[i] - r11[i] - r13[i] - r15[i] - r17[i] - r19[i] - r21[i] - r23[i] - r25[i] - r27[i] ) % m for i in range(L)]

            return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28)
        
# ========================================
#
#      The Multiplication Function
#
# ========================================
def multiply(f, g, n, m, formulas="efficient"):
    """ This multiplies f and g mod 2^m using Toom-n."""
       
    if len(f) != len(g):
        raise ValueError("Can only multiply polys of the same length")
            
    fblocks = split(f, n)
    gblocks = split(g, n)
    
    # the list of evaluating numbers
    eval_list =  make_eval_list(n)
    
    # plug the numbers in
    f_eval = evaluate_blocks_list_mod(fblocks, eval_list, 2**m)
    g_eval = evaluate_blocks_list_mod(gblocks, eval_list, 2**m)
    
    # perform the recursive multiplication
    r = {eval_list[i]:schoolbook_mod(f_eval[i], g_eval[i], 2**m)
            for i in range(len(f_eval))}
    
    # Solve for the coefficients    
    r_coefs = solve_for_coefficients_mod(n, r, 2**m, formulas)
    
    # recombination
    k = int(np.ceil(len(f) / n))
    prod = r_coefs[0][:k]
    for j in range(1, 2*n-2):
        prod = prod + [(r_coefs[j-1][k+i] + r_coefs[j][i]) % 2**m for i in range(k-1)]
        prod = prod + [r_coefs[j][k-1]]

    prod = prod + [(r_coefs[2*n-3][k+i] + r_coefs[2*n-2][i]) % 2**m for i in range(k-1)]

    prod = prod + r_coefs[2*n-2][k-1:]

    return prod[:2*len(f)-1]

# ========================================
#
#            Precision Loss
#
# ========================================
def strongest_congruence(a, b, max_pow):
    """ This returns the largest positive int m such that
        a = b mod 2^m, up to max_pow"""
    cur_pow = 1
    prev_pow = 0
    while (a % 2**cur_pow) == (b % 2**cur_pow) and prev_pow < max_pow:
        cur_pow += 1
        prev_pow += 1
    return prev_pow

def strongest_congruence_list(f, g, max_pow):
    return min([strongest_congruence(f[i], g[i], max_pow) for i in range(len(f))])

def bits_lost(f, g, m):
    return m - strongest_congruence_list(f, g, m)

def precision_lost_single_trial(f, g, n, m=32, formulas="efficient"):
    """ Returns the number of bits of precision lost by multiplying f
        and g according to Toom-n mod 2^m with the specified
        interpolation formulas."""
    true_answer = schoolbook_mod(f, g, 2**m)
    toom_answer = multiply(f, g, n, m, formulas)
    return bits_lost(true_answer, toom_answer, m)

def precision_lost_many_trials(n, m=32, formulas="efficient", num_trials=100):
    max_loss = 0;
    for _ in range(num_trials):
        degree = int(np.random.randint(2*n, 10*n))
        f = [int(x) for x in np.random.randint(0, 2**m, degree)]
        g = [int(x) for x in np.random.randint(0, 2**m, degree)]
        loss = precision_lost_single_trial(f, g, n, m, formulas)
        if loss > max_loss:
            max_loss = loss
    print("Toom-{} with the {} interpolation formulas loses {} bits of precision.".format(n, formulas, max_loss))
    return max_loss

precision_lost_many_trials(15, m=31, formulas="natural")