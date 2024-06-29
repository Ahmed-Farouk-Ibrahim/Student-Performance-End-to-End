from setuptools import find_packages, setup
from typing import List

# Constant for the editable install flag used in requirements files
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    Reads a requirements file and returns a list of dependencies.
    
    Args:
    file_path (str): The path to the requirements file.
    
    Returns:
    List[str]: A list of package requirements.
    '''
    requirements = []
    
    # Open the requirements file and read its contents
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        
        # Remove newline characters from each line
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove the editable install flag if present
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

# Setup configuration for the mlproject package
setup(
    name='mlproject',
    version='0.0.1',
    author='Krish',
    author_email='krishnaik06@gmail.com',
    packages=find_packages(),  # Automatically find and include all packages in the project
    install_requires=get_requirements('requirements.txt')  # Install dependencies from the requirements file
)
