# SIEVE

## Space Efficient Viterbi Decoding

* Viterbi.py contains class for handling data specified through transition and emission matrices 
* Viterbi_forced_alignment.py contains class for handling forced alignment data specified through transition lists, emission follows a GMM model and both emitting and non-emitting states are admitted.


##### Cython
To use the code, if Cython is available, first 'python setup.py build_ext --inplace'. command builds the .c files created by Cython. Alternatively, without running the mentioned command, it is possible to directly execute the Python code.

#### Examples 

Run "python example_script.py" to run an example comparing Viterbi and Sieve solution with arbitrary transitions and run "python example_script_dag.py" to run an example comparing Viterbi and Sieve solution with DAG transitions. 
