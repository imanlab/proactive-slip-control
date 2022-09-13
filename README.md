# Proactive slip control by learned slip model and trajectory adaptation


This repository includes the codes and sample dataset to run the experiments for Reactive Slip Control (**RSC**) and Proactive Slip Control (**PSC**) by robot trajectory adaptation.

There are two stages to run the experiments:
* offline training of slip classification (detection/prediction) models
* online slip control by trajectory adaptation


**Offline Training**

We collected a dataset of pick-and-move tasks by sending a reference trajectory for robot's task space velocity. The Panda Franka robot executes a linear motion in XY plane in the base frame. A box shaped object is used as the training object and two uSkin tactile sensors from XELA ROBOTICS are attached to the 3d printed fingers mounted on franka gripper. A wrist camera reads the pose of an Aruco marker attached on the objects to detect objec's relative motion w.r.t the EE and generating the binary slip ground truth values. The data set include synchronized tactile (2 x $D \in R^{48}$), robot ($q \in R^7$, $dq \in R^7$, $EE_{pose} \in R^{16}$, and $EE_{vel} \in R^6$), and marker pose data ($P \in R^7$).

<p align="left">
  <img src="photos/setup.png" width="348" />
  <img src="photos/taskgif.gif" width="300" />    
  <center>Figure1. Experimental Setup</center>
</p>

The *gen_dataset.py* in both /slip_detection and /slip_preditction directories should be run first to preprocess and save sequences of data in a specified directory. Since we don't want to load all of the dataset into memory when training, we save them as sequences and load them in batches by using the *map.csv* (file which contains all file names) and torch data_loader class.


        run gen_dataset.py to create data sequences for training.

After this stage, we can run the *model_train.py* script in each directory to train and save the slip detection and slip prediction models.

        run model_train.py to train slip classification models.


Now that we have the (i) trained torch models and (ii) saved scalars for preprocessing, we can go to the real time test stage.

**Online slip control**

We use ROS noetic for real-time test cases. To be able to use *libfranka* in you ROS package you need to add it to CMakeLists.txt of your package as follows:

        find_package(Franka 0.8.0 EXACT REQUIRED PATHS /libfranka/build NO_DEFAULT_PATH)

        add_library(examples_common STATIC
        examples_common.cpp
        )

To decrease the inference time of the pyTorch models we use ONNX library. You can use rt_robot_test/ONNX_torch/torch2onnx.py to convert trained torch models to onnx files. Use this [page](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) as reference.

To bring up the hardware we need to first run *xela_server* for uSkin sensot:

        ./rt_robot_test/XELA/u20/xela_server

And then use this launch file to publish synchronized tactiel sensor, realsense, and ArUco marker data under one topic:

        roslaunch rt_robot_test/launch/slip_control.launch

Now we can run the **RSC** and **PSC** models! For each test we have slip detection/prediction + optimization script (.py) and Cartesian velocity control script (.cpp) to run together. For RSC we need:

        rosrun rt_robot_test/python scripts/CoRL/RSC.py
        rosrun rt_robot_test/slip_controller.cpp robot_ip

And for the **PSC** we need:

        rosrun rt_robot_test/python scripts/CoRL/PSC.py
        rosrun rt_robot_test/slip_controller.cpp robot_ip
