# asen5044-odet-project
Orbit determination final project for ASEN 5044: Statistical Estimation for Dynamical Systems

# Setup
1. Create a new virtual environment: `$ python3 -m venv venv/`
2. Activate the virtual environment: `$ source venv/bin/activate`
3. Install dependencies: `$ pip install -r requirements.txt`

# Remaining todos:
1. LKF
    * figure out initialization for dx and P0. need to use this to initalize dx for true trajectory.
        * ideas: initial pert to zero and covariances acc to what's normal for LEO. Maybe base this on the limits of the Clohessy-Wiltshire to set the x and y lims (https://space.stackexchange.com/questions/9618/limitations-of-clohessy-wiltshire-equations#comment30043_9618 comment from Artas)
    * tune Q
    * Plot typical output from a single sim with LKF est: truth states, truth measurements, and 2 sigma error graph
    * Monte Carlo
        * choose alpha and number of runs
        * plot NEES and NIS test results
2. EKF
    * convert dx0 and P0 from LKF to full state and use this
    * tune Q
    * Plot typical output from a single sim with EKF est: truth states, truth measurements, and 2 sigma error graph
    * Monte Carlo
        * use same alpha and number of runs from LKF
        * plot NEES and NIS test results
3. estimate state trajectory from Canvas measurements
    * LKF: plots of estimated states and 2 sigma err
    * EKF: plots of estimated states and 2 sigma err