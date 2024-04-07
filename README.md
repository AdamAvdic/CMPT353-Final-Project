# CMPT353-Final-Project

Project Topic: Sensors, Noise, and Walking
Most of us are carrying an amazing collection of sensors in our pocket right now. Every smartphone has many sensors available, including a 3D accelerometer. Can we use the accelerometers on our phones to tell us something about our users' gait?

This project idea was heavily inspired by Maria Yousefian's MSc thesis work: she compared phone-based data with clinical-quality motion capture, for analysis of gait and fall risk in elderly patients.

We will set our target a little lower, but will try to get what information we can from the phone's sensor array. If you prefer, you can think of this as analogous to the task done by a Fitbit or other wearable.

Possible Questions
There are a lot of directions this could go.

How does walking pace (steps/minute or similar) differ between people? Does it vary by age, gender, height, …?
Can you calculate walking speed (m/s) from just the accelerometer? How accurate can you get? Can you then determine differences between subjects as above?
If you compare right-foot vs left-foot, can you determine if someone's gait is asymmetrical? Perhaps this can be used to detect an injury. (This would probably be easier if you have somebody with an injury to work with. Please do not injure your friends to answer this question.)
Are the results better depending on where the phone is? Ankle vs pocket vs hand? Ankle above or below the joint?
Getting Data
Is project is going to involve true data collection: you don't have any data, but you have a smartphone (or presumably, one group member does) and can use it to collect some data.

There are (of course) apps to record data from the accelerometers. For Android, Physics Toolbox Sensor Suite seems very good. [iOS suggestions welcome: email Steven.] For this project, we want an app that does not filter noise from the sensor: we want the noise so we can filter it in a way appropriate to the problem.

You'll need to attach the phone somewhere. The best results will likely come from attaching it near the ankle. Attaching it more firmly will help reduce noise. I don't want to say “duct tape” here, but… I also can't think of anything better.

Data Analysis
The data from the phone sensors will contain a lot of noise. In her thesis, Maria used a Butterworth filter to deal with it, so my first suggestion would be to leverage her wisdom and try that. I gave a terse example of using a Python Butterworth filter in lecture.

Once you have some not-horribly-noisy data, you can try a Fourier transform to get the frequency of steps. There are FFT implementations both in NumPy and in SciPy.

You can also do what amounts to numerical integration to transform acceleration to velocity and then to position: Δv = a⋅Δt and Δp = v⋅Δt.

Of course, your results aren't going to be perfect: we're asking a lot from a cheap noisy sensor. How accurate and useful can you make your results?

I can't speak for your phone, but my sensor has some bias and/or drift that makes everything go slightly wrong. I had to make the rule “stand still for a second at the start and end” and use that data to un-bias the readings.

Maria offers a hint for the best results: it's possible to look at the phone's gyroscope to determine the change in orientation. As the phone moves, there isn't a single one of the x, y, z directions that corresponds to “forward”. If you can rotate the signal to keep “forward” and “down” correct, you'll get better results.
