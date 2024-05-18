#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
-------------------------------------------------------------------------------
   Name:         t1
   Description:  
   Author:       ASUS
   Date:           2023/12/17
-------------------------------------------------------------------------------
   Change Activity:
                 2023/12/17
"""

__author__ = "Xwanga"
__version__ = "1.0.0"
import pickle
if __name__ == '__main__':
    with open("feature_database.pkl", "rb") as f:
        all_features = pickle.load(f)
    print(all_features)