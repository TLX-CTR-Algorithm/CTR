import xlearn as xl
import config

# Training task
ffm_model = xl.create_ffm() # Use field-aware factorization machine
ffm_model.disableEarlyStop()
ffm_model.setTrain("./train_ffm.txt")  # Training data
ffm_model.setValidate("./valid_ffm.txt")  # Validation data

# param:
#  0. binary classification
#  1. learning rate: 0.2
#  2. regular lambda: 0.002
#  3. evaluation metric: accuracy
param = {'task': 'binary', 'lr': 0.1,
         'lambda': 0.002, 'metric': 'auc',
         'opt': 'ftrl',  'epoch': 20}
# param = {'task': 'binary', 'lr': 0.15,
#          'lambda': 0.00002, 'metric': 'auc',
#          'opt': 'ftrl',  'epoch': 10}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model.out')

# Prediction task
ffm_model.setTest("./test_ffm.txt")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")


