#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:34:45 2017

@author: chris
"""
#class ConvNet():
#    import numpy as np
#    import tensorflow as tf
#        
#    def __init__(self, seqLength, predLength):
#        self.seqLength = seqLength
#        self.predLength = predLength
#
#    

def em_loss(y_coefficients, y_pred):
    return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

    
def cnn_model_fn(features, labels, mode, seqLength, predLength):
  """Model function for CNN."""
  #%%
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, seqLength, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=32,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=64,
      kernel_size=[5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=[2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, np.prod(pool2.get_shape().as_list()[1:])])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  output = tf.layers.dense(inputs=dropout, units=predLength)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
#          "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "values": tf.identity(output, name="value_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


#  batch_size = np.prod(pool2.get_shape().as_list()[0])
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.mean_squared_error(labels, output)
  # calculate discriminator's loss
#  loss = em_loss(tf.ones(batch_size), output) - \
#    em_loss(tf.ones(batch_size), labels)
  
#  lossSummary = tf.summary.scalar('loss', loss )
  
#  tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)



  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Define loss and optimizer
#        cost = tf.reduce_mean(tf.square(pred-y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)        
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    
#    gradients, variables = zip(*optimizer.compute_gradients(loss))
#    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
#    train_op = optimizer.apply_gradients(zip(gradients, variables),
#        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.mean_squared_error(labels=labels,
          predictions=predictions["values"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  
  
  #%%
if __name__ == '__main__':    
    #%%
    import numpy as np
    import tensorflow as tf
    import ConvNet
    import matplotlib.pyplot as plt
    import pandas
    import shutil
    import os
    from datetime import datetime
    currTime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    tf.reset_default_graph()
    
    if(os.name == "nt"):
        dirName = "c:/Users/Karl/tmp/"
    else:
        dirName = "/tmp/"
        
    dirName += "convNetModel/" +currTime
        
          #
    try:
        shutil.rmtree(dirName)
    except:
        print('File Handling Error')        
        

    #%%
    
    dataCSV = pandas.read_csv(
            "./bitcoin-historical-data/btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv",
            verbose=False, ) 
    
    data = dataCSV['Weighted_Price']
    
    #%%
    y = np.asarray(data)
    
    foundNans=True
    while foundNans:
        nanIdx = np.where(np.isnan(y))[0]
        y[nanIdx]=y[nanIdx+1]
        foundNans = len(nanIdx)
    
#    normalizerMean = np.mean(y)
#    normalizerStd = np.std(y)
#    y -= normalizerMean
#    y /= normalizerStd
    
    #%%
    seqLength = 100
    predStart = -20
    predLength = 40
    nSamples = y.shape[0]-(seqLength+predLength+predStart)
    
    features = np.zeros((nSamples, seqLength), dtype=np.float32)
    targets = np.zeros((nSamples, predLength), dtype=np.float32)
    

    
    for i in np.arange(y.shape[0]-(seqLength+predLength+predStart) ):
            features[i,:] =  y[i:i+seqLength]  
    for i in np.arange(y.shape[0]-(seqLength+predLength+predStart) ):
        targets[i,:] = y[i+seqLength+predStart:i+seqLength+predStart+predLength] 
        
    
    

    #%%
    rTrain = 0.7
    iTest = int(features.shape[0]*rTrain)
    featuresTrain = features[:iTest,:]
    targetsTrain = targets[:iTest,:]

    featuresTest = features[iTest:,:]
    targetsTest = targets[iTest:,:]        
    
  
    
    # Create the Estimator
    regressor = tf.estimator.Estimator(
    model_fn=lambda features, labels, mode : cnn_model_fn(
            features, labels, mode, seqLength, predLength), 
            model_dir=dirName) 
    
    # Set up logging for predictions
    tensors_to_log = {"values": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=1)
    
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": featuresTrain},
            y=targetsTrain,
            batch_size=500,
            num_epochs=1,
            shuffle=True)
    
     # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": featuresTest},
        y=targetsTest,
        num_epochs=1,
        shuffle=False)
    
     # Evaluate the model and print results
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": featuresTest[-50000:,:]},
        y=targetsTest[-50000:,:],
        num_epochs=1,
        shuffle=False)
        #%%
    for iter in np.arange(1000):
    # Train the model
        #
        regressor.train(
            input_fn=train_input_fn,
            steps=None,
            hooks=None)
        
       
        eval_results = regressor.evaluate(input_fn=eval_input_fn)
        print(eval_results)
        
        #%%
        # predict Results
        predRes = regressor.predict(pred_input_fn)
        yPred = [ p["values"][-predStart + 5] for p in predRes]
        yPred = np.asarray(yPred) 
        plt.figure(1)
        plt.clf()
        plt.plot(targetsTest[-50000:,-predStart + 5])
        plt.plot(yPred, 'r.')
        plt.title( "RMSE: " + str(eval_results['loss']) )
        plt.draw()
        plt.pause(1)
            
        print(predRes)
