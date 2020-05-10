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

precision_lost_many_trials(7, m=31)

