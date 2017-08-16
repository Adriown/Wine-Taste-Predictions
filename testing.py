
from FinalPython import *
import unittest

class URLTestCase_checkEmptyURL(unittest.TestCase): # inherit from unittest.TestCase
    # Unit testing __init__() method in the function I wrote
    
    def test_is_empty_url(self):

        # Set up
        output = funcGetDataFromURLAndFormat('')
        
        # Test
        self.assertEqual(output, 'empty URL') # should be 'empty URL'


class URLTestCase_checkWineURL(unittest.TestCase): # inherit from unittest.TestCase
    
    def test_is_wine_url(self):
        # Is __init__() method successfully implementing the account number attribute

        # Set up
        output = funcGetDataFromURLAndFormat("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
        
        # Test
        self.assertEqual(output.iloc[0,0], 7.0) # should be 7.0

    
        
        
if __name__ == '__main__':
    unittest.main()            