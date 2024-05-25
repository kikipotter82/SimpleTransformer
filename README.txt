The directory "PA2_code" contains the starter code.
Feel free to ignore the starter code and implement from scratch.
However, you should use the base settings for the hyperparameters, as specified in main.py.

The directory "speechesdataset" contains the data to be used for this assignment.

The directory "PA2_code" contains 5 python file: 
dataset.py
main.py: the main program
tokenizer.py
transformer.py: the encoder and the decoder
utilities.py: slightly changed to better fit my model

To run Part 1, Part 2 and Part 3. Using the following code:
'''
    % python PA2_code/main.py --part part1
    % python PA2_code/main.py --part part2
    % python PA2_code/main.py --part part3
'''

Besides, for the part 3, the current code is the final result of exploration and improvement, 
i.e. the integration of part 3.1 and part 3.2. 
If you want to see the results of part3.1 and 3.2 respectively, please feel free to change 
the corresponding comment codes in the part3() function, and then run the program. 
corresponding comment code including:
'''
    lr_new = 0.01
    # lr_new = 0.1
    # lr_new = 1e-3 
'''
and
'''
    optimizer = torch.optim.Adagrad(decoder.parameters(), lr=lr_new)
    # optimizer = torch.optim.RMSprop(decoder.parameters(), lr=lr_new)
    # optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr_new)
'''

Notice that the file path of "speechesdataset" might be different in different devices. if the
program cannot run, please check the file path.