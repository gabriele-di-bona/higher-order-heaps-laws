import os
import fnmatch

def find_pattern(pattern, directory, check_only_in_file_name = False):
    '''
        Finds all the paths inside the given directory that are of the given pattern.

        E.g.: find_pattern("*.pkl", "./") to find all the files terminating with .pkl inside the directory or subdirectory.
        
        If check_only_in_file_name == True, the pattern recognition only applies to the name of the file inside the directory or subdirecory inside directory.
        Otherwise, the pattern recognition applies also to the absolute path of the current directory or subdirectory.

        Returns the list of all such paths (if empty then none is found).
    '''
    result = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if (check_only_in_file_name == True and fnmatch.fnmatch(name, pattern)) or (check_only_in_file_name == False and fnmatch.fnmatch(os.path.join(root, name), pattern)):
                result.append(os.path.join(root, name))
    return result