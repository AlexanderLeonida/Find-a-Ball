# Video Examples of Program Running

[Here](https://drive.google.com/drive/u/1/folders/1fSQVrUMbTzIXCglkt4jBT3UMeBTuRV-9) are videos of the program running. 

The videos listed are examples that I thought about which could be functionally interesting to show. 

In the overview.mov video, I first show an example of the linear regression model, then of the LSTM model. Every other video showcases the power of the LSTM model. 

**NOTE** For a majority of the videos, it takes a little while for the LSTM neural network to train itself. When the white ball on the screen stops moving (I'm no longer able to enter user input), the AI is training itself. You'll want to fast forward until the ball turns red and you can see the AI output prediction.

## What You're Actually Seeing

When the ball is **white**, I (the user) am moving the ball. When the ball is **red**, the specified model has taken over and is moving the ball in the way that it thought you would have continued based on your previous movements of the ball on the screen. 

In the overview video, we first see the linear regression model at play. Using both position and velocity, the model moves the ball quickly, increasing in both size and velocity as time goes on. This is because as velocity increases, it too will increase in speed. 

Then, when we rerun the program and the LSTM model takes over, it will take some time to train. I show the output of the training at the very end of the video in the terminal. 

There, what you are seeing describes the model over iterations of backpropogating on itself (an epoch). As loss's of both X and Y decrease, the model gets closer to training itself correctly based on the testing data that we've split up the user's input into. 

The final printed array is of the predicted X and Y coordinates of the model. 