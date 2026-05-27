#!/usr/bin/env python3
''''df creation from file'''
import pandas as pd


def from_file(filename, delimiter=','):
    '''creates a pd.DataFrame from a file'''
    return pd.read_csv(filename, delimiter=delimiter)
