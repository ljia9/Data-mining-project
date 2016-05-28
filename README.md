# Data-mining-project

# Description
This is the final project I used to compete in the Driven Data challenge: From Fog Nets to Neural Nets where I placed in the 85th percentile. My script uses a basic scipy python
implementation of a random forest regressor to predict the rainfall yields of a given season when provided microclimate data. Also it has a naive data generator to produce 
pseudorandom values to fill in missing values and fix many formatting issues.

The files in the datasets folder show the training and test datasets I use for my classifier. The original_datasets folder shows the original, unedited datasets I downloaded 
from the challenge that I use in my project. The submission_files folder shows the files that were involved in my turnins to the online challenge. Finally, the format folder 
shows the files used to ensure that my output file the format needed in the competition submission. 

There are some superfluous files that I kept in case for whatever reason I lost my scripts and had to start over; these files are used for data manipulation.

# How to Run my Programs
The program create my classifier and runs it on the test data provided by the challenge is PyScript.py and so it outputs an ans.txt that shows the classifier's predicted values.The test dataset is based on the original dataset by it is bin and smoothed using the DataGenerate.py script which produces new attribute values for the 500 or so missing values
based on the mean value for each attribute and the standard deviation of that value.

Note that I use several python libraries (sklearn and numpy) so it is necessary to download and install these before using my program with "pip install <>"

Therefore to run the programs just do:

    python DataGenerate.py
    python PyScript.py


# Note
In order to output a correctly formatted submission file, there is a bit of hard coding. I did not find the time to completely genericize the entire process so in order to 
really produce an output ready for submission you have to do some manual file manipulation.

In particular you have to copy and paste the output of DataGenerate.py (in the new-data.txt file) and place it in the correct order into the z-test.txt file. Then edit the file
to remove the title and the unnecessary attribute of the data. Also, to outpute a correct submission file, after running PyScript.py edit the ans.csv file to include the
parameters: ',yield'. 
