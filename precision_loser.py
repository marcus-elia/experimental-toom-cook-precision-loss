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
    if n == 2:
        return (0, 1, 'infinity')
    if n == 3:
        return (0, 1, -1, 2, 'infinity')
    if n == 4:
        return (0, 1, -1, 2, -2, 3, 'infinity')
    if n == 5:
        return (0, 1, -1, 2, -2, 3, -3, 4, 'infinity')
    if n == 6:
        return (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, 'infinity')
    else:
        raise ValueError("We haven't implemented Toom-" + str(n))

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

precision_lost_many_trials(5, m=16)

