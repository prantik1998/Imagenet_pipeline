# AIM -

* GUIDELINES TO BE FOLLOWED WHILE WRITING A PROGRAM 
* NEVER REPEAT A SINGLE LINE OF CODE
* EXTENSIVELY MAKE USE OF HIERARCHY, CLASSES AND FUNCTIONS*

# Naming - 

* All class names should follow "CamelHump" naming style
* All file name should follow "this_naming_style"
* All function name should follow "verySimilarToCamelHump" style

# Documentation - 

* Every function should have a 1-2 line description of every input argument and output argument
* There should be a 3-4 line description of what the function does
* There should be a "requirements.txt" file which contains all the packages used as well as their version

# Pipeline - 

* All programs will be stored in the "src" folder
* All constants related to your program will be stored in "config/congif.py" file.
* There would be only one "main.py" file which will be used to drive all the other functions

# Experiment Folder - 

* All files of the current run should be stored in the "Exp" directory whose path will be given in the "config/config.py" file.
* If the Experiment folder already exists, user should be given a choice to rename the older folder or to replace it.

# Logger - 

* There should be a logger function which stores the time-stamp of each output in the file "Exp/log.txt". It should also print the output on the console screen

# Saving Work - 

* The "work" should be saved such that worst case scenario only 5 mins of work is lost. 
*Care should be taken not to store the "work" too often which would cause hard disk overload*

# Misc - 

* All the other "things" which has not been mentioned in the pipeline can be kept in the folder misc. 
This folder is not versioned.

# Running the Code - 

* Open a terminal and type python main.py <function_name>
* For example, for training - python main.py train

# Dataset format - 

For this ImageNet pipeline the data-set should have the following structure

Images/<br>
├── 0<br>
├── 1<br>
└── 2<br>

Every class should have specific folder. The way the data is loaded can be found in src/data_loader.py