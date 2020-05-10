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

precision_lost_many_trials(12, m=31)

