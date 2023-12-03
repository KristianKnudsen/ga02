
# Running Graded assignment 02

## Setup
The application should work out of the box provided you you have the necessary requirements in your enviorment.

To get the requirements use the completeEnv.yaml file in conda to create a compatible envirment.

Alternatively you can use pip like this:
pip install -r requirements_all.txt

## Run
To run the program you're going to use training.py

simply run:
    python training.py

from the terminal while being located in the folder where the file lies.

The current code is all linked to version 22 and does not need to be changed.


However in training.py you can choose to change the amount of episodes, the log frequency, the decay and the epsilon end.

If you want it to converge abit faster you can try and change the episdoes to 200k the log frequency to 500. The decay to 0.97 and the epsilon_end to 0.01.

The results will be displayed and updated in model_logs/v22.csv

If you want to generate images open the game_visualization and change the iteration list to the desired iteration to run at. Then run python game_visualization.py

The output will be located in the images folder.

To visualize other version simply change the version number and the iteration list to a matching model found in /models/.
