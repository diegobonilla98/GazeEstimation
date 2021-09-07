# GazeEstimation
Some UNet segmentation task for gaze estimation. I will do some fun project with these results.

## Network
Just simple UNet with residual connections, relu activations and sigmoid output. BCE with Dice loss. Little more than 1M parameters. Not amazing but does the work, pretty good accuracy on non-tricky images. Trained on the Gi4e dataset (available [here](http://www.unavarra.es/gi4e)).

## Results
I used some haar cascade detectors for face and eyes detections (not seeking state-of-the-art results here).
![](./ezgif.com-gif-maker.gif)

## What for then
Well, I was thinking about a pretty amazing project and I couldn't find any simple implementations of gaze estimations (very good ones, but not simple). **STAY PUT FOR UPDATES IN THE NEW PROJECT!!!**  ❤️❤️🖤



