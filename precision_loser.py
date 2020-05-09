# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:15:41 2020

@author: Marcus

This file is for experimentally determining the precision loss, for all three
(matrix, natural, efficient), hopefully up to at least Toom-10.
"""

# ==============================
#
#         Math Functions
#
# ==============================
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
    v = (g - a*u) / b    
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
        odd_part /= 2
        even_part *= 2
    return (odd_part, even_part)