
# CNN_Quaternet_evaluation
This repository has two folders as follows:
- Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics
- QuaterNet

### To run evaluations on Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics follows below steps:
1) Download the data from the Google drive link below and copy the folder to the root directory.
[Model Link](https://drive.google.com/file/d/1bHZCH6u7YmloEwzDO--w91qmAmvCcv3A/view)
2) To run the model and produce output. Go to the root directory and run

		python src/AC_main.py

3) To run the evaluations 

		python src/in_out_visualization/forward_kinematics_evaluation.py	

### To run evaluations on QuaterNet follows below steps:
1)  Download the data by running 
		

	    python prepare_data_short_term

2)  Run the model and produce output. 
		

	    python test_short_term.py

3)  To run the evaluations 

	    python test_short_term_fk_evaluate.py
