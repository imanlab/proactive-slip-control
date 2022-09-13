#include <cmath>
#include <iostream>
#include <fstream> // include this for saving .csv files
#include <string>
#include <unistd.h> // include this to use time delay
#include "client.hpp" // include this for socket client in cpp. refer to https://github.com/OleguerCanal/cpp-python_socket


#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/gripper.h>

#include "examples_common.h"

// import ROS
#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Int16.h"
#include <sstream>
#include <math.h> 

/**
 * This is based on the cartesian velocity controller example to have 
 * Trapezodal velocity profiles in cartesian space for post grasp motion with various V_max values.
 */

float slip_value;
void slipCallback(const std_msgs::Float64& slip)
{
  slip_value = slip.data;
}

float optimal_v_x = 0;
float optimal_v_y = 0;

void trajCallback(const std_msgs::Float64MultiArray& traj)
{
  optimal_v_x = traj.data[0];
  optimal_v_y = traj.data[1];
  std::cout << "optimal_cb: " << optimal_v_x << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl; // The only argument given is robot ip
    return -1;
  }
  try {
    ros::init(argc, argv, "robotPub");
    ros::NodeHandle n;
    ros::Publisher robotPose_pub = n.advertise<std_msgs::Float64MultiArray>("robotPose", 1000);
    ros::Publisher slip_data_pub = n.advertise<std_msgs::Float64MultiArray>("slipData", 1000);
    ros::Subscriber sub = n.subscribe("slip_prediction", 1000, slipCallback);
    ros::Subscriber traj_sub = n.subscribe("optimal_traj", 1000, trajCallback);
    int count = 0;

    franka::Robot robot(argv[1]);
    franka::Gripper gripper(argv[1]);
    setDefaultBehavior(robot);

    // move the robot to a suitable home joint configuration
    std::array<double, 7> q_goal_1 = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    MotionGenerator motion_generator_1(0.4, q_goal_1);
    robot.control(motion_generator_1);

    franka::GripperState gripper_state = gripper.readOnce();
    gripper.move(gripper_state.max_width-0.005, 0.4); // first arg=width, second=speed

    // move the robot to the pregrasp joint configuration
    // std::array<double, 7> q_goal_2 = {{0.46594042874768027, 0.5736332212894555, -0.121005812794016, -1.427936394524156, 0.08442840939097934, 2.0023333249621924, 0.15335033177618285}};
    std::array<double, 7> q_goal_2 = {{0.4651429217714512, 0.5451831926466233, -0.1241411501164842, -1.3537963933302153, 0.0792001422961141, 1.89436454651777, 0.1592959655348413}};
    MotionGenerator motion_generator_2(0.4, q_goal_2);
    robot.control(motion_generator_2);

    // theBest old
    // std::array<double, 7> q_goal_3 = {{0.4622160324939316, 0.6747957952584068, -0.11265158838587289, -1.5016724266821633, 0.09726375189268904, 2.169628276489114, 0.1445644477809302}};
    // theBest new
    std::array<double, 7> q_goal_3 = {{0.46573687304994976, 0.607229224859898, -0.1190305153536798, -1.462855484084864, 0.08778084733934619, 2.0633288509728755, 0.1533736983142435}};
    // snail killer bottle
    // std::array<double, 7> q_goal_3 = {{0.4302862460415172, 0.6226217721051964, -0.0733636891335603, -1.484176657599378, 0.062308337121343794, 2.099884310483682, 0.16476227917879832}};
    // Chilli can
    // std::array<double, 7> q_goal_3 = {{-0.19898747290664703, 0.8878438630912647, 0.754183526641446, -1.4858119658601288, -0.6488857671072086, 2.097742687524623, 0.5420324414059704}};
    MotionGenerator motion_generator_3(0.4, q_goal_3);
    robot.control(motion_generator_3);

    // Grasp the object.
    // gripper.move(0.0448, 0.5); // original for data collection
    // theBest new
    // gripper.move(0.044, 0.5);
    // snail killer bottle
    // gripper.move(0.065, 0.5); // 0.066 original
    //  Thins
    // gripper.move(0.042, 0.5);
    // Rice
    // gripper.move(0.046, 0.5);
    // Carrs
    // gripper.move(0.057, 0.5);
    // Monster
    gripper.move(0.062, 0.5); // actual value for testing is 0.06

    
   robot.setCollisionBehavior({{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                               {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                               {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}},
                               {{1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0}});
                               
    // move the robot to the pregrasp joint configuration
    // std::array<double, 7> q_goal_4 = {{0.46594042874768027, 0.5736332212894555, -0.121005812794016, -1.427936394524156, 0.08442840939097934, 2.0023333249621924, 0.15335033177618285}};
    std::array<double, 7> q_goal_4 = {{0.4651429217714512, 0.5451831926466233, -0.1241411501164842, -1.3537963933302153, 0.0792001422961141, 1.89436454651777, 0.1592959655348413}};
    MotionGenerator motion_generator_4(0.05, q_goal_4);
    robot.control(motion_generator_4);
    
    socket_communication::Client client("127.0.0.1", 5003);
    client.Send("Pre-grasp!");
    unsigned int microsecond = 1000000;
    usleep(0.3 * microsecond);//sleeps for 3 second

    // wait a bit before starting the linear motion in -X direction
    usleep(1000000);
    client.Send("Object_Grasped!");

    // Set the joint impedance.
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});

    // Create a vector of vectors which is a 2D matrix to push_back robot state data into.
    // To learn more about "The C++ Standard Template Library (STL)" refer to https://www.geeksforgeeks.org/the-c-standard-template-library-stl/
    std::vector< std::vector<double> > robot_pose;
    std::vector< std::vector<double> > joint_position;
    std::vector< std::vector<double> > joint_velocity;

    bool slip_flag_old = false;
    bool slip_flag_new = false;
    bool task_ending_phase = false;
    double slip_onset_time = 0;
    double slip_onset_v_x = 0;
    double slip_onset_v_y = 0;
    double slip_onset_delta_x = 0;
    double slip_onset_delta_y = 0;
    double remained_delta_x = 0;
    double T_slip = 0;
    double acc_slip = 0;
    double task_ending_v_x = 0;
    double task_ending_acc = 0;
    double task_ending_time = 0;

    
    double V_MAX_x = 0.4; // max of V_MAX_x = 0.4 for translation_x = 0.35 m and translation_y = 0.5

    double translation_x = 0.35;
    double translation_y = 0.46;
    double V_MAX_y = (translation_y/translation_x) * V_MAX_x;
    
    double time = 0.0;
    double acc = 0;
    double v_x, v_y;
    double v_x_0, v_y_0;
    double T;
    T = translation_x / (2*V_MAX_x);
    std::string filename = "/home/kiyanoush/Desktop/robot_pose_cont_wocont.csv";
    std::vector<double> v_x_save;
    std::vector<double> v_y_save;
   
    franka::RobotState grasp_state = robot.readOnce();
    double grasp_x = grasp_state.O_T_EE[12];

    // Here the Cartesian velocity control loop starts. It has a lambda definition which inholds the control callback function in itself.
    robot.control([=, &time, &robot_pose, &gripper](const franka::RobotState& rob,
                             franka::Duration period) mutable -> franka::CartesianVelocities {
      
      std::vector<double> EE_pose_vec = {rob.O_T_EE[0], rob.O_T_EE[1], rob.O_T_EE[2], rob.O_T_EE[3], rob.O_T_EE[4], rob.O_T_EE[5], rob.O_T_EE[6],
                                   rob.O_T_EE[7], rob.O_T_EE[8], rob.O_T_EE[9], rob.O_T_EE[10], rob.O_T_EE[11], rob.O_T_EE[12], rob.O_T_EE[13],
                                  rob.O_T_EE[14], rob.O_T_EE[15]};
      robot_pose.push_back(EE_pose_vec);

      time += period.toSec();
      
      std_msgs::Float64MultiArray robot_pose_msg;
      robot_pose_msg.data.clear();
      robot_pose_msg.data.resize(20);
      EE_pose_vec.push_back(v_x);
      EE_pose_vec.push_back(v_y);
      EE_pose_vec.push_back(time);
      EE_pose_vec.push_back(0.0); //this is the flag to continue/end subscribing fo python script subscriber
      robot_pose_msg.data = EE_pose_vec;
      robotPose_pub.publish(robot_pose_msg);
      

      if (slip_value == 1){
        slip_flag_new = true;
      }
     
      if (slip_flag_new != slip_flag_old){
        slip_onset_time = time;
        slip_onset_v_x = v_x;
        slip_onset_v_y = v_y;

        slip_onset_delta_x = 0.5 * (slip_onset_time - 1) * (v_x - 0.02) - 0.5 * 0.02;
        slip_onset_delta_y = 0.5 * (slip_onset_time - 1) * (v_y - (translation_y/translation_x)*0.02) - 0.5 * (translation_y/translation_x)*0.02;

        remained_delta_x = translation_x + slip_onset_delta_x + 0.2 * slip_onset_v_x; // third term is because of 1000 -> 60 hz delay
        T_slip = slip_onset_time - 2*remained_delta_x/slip_onset_v_x;
        acc_slip = slip_onset_v_x / (T_slip - slip_onset_time);
        // std::cout << "T_slip: " << T_slip << std::endl;
        std::cout << "time: " << time << std::endl;
        std::cout << "slip_onset_v_x: " << slip_onset_v_x << std::endl;
      }

      std::vector<double> slip_onset_data = {slip_onset_v_x, (T_slip - slip_onset_time)};
      std_msgs::Float64MultiArray slip_msg;
      slip_msg.data.clear();
      slip_msg.data.resize(2);
      slip_msg.data = slip_onset_data;
      slip_data_pub.publish(slip_msg);

      // This if condition creates the cartesian trapezoidal velocity profile in which the duration of all three parts are T.
      if (time < 1){
        // std::cout << "if 1 v_x : " << v_x << std::endl;
        v_x = -0.02 * time;
        v_y = (translation_y/translation_x) * v_x;
        v_x_0 = v_x;
        v_y_0 = v_y;
      }

      if (slip_flag_new == false){
      if ((time >= 1) && (time < (T+1))){
        // std::cout << "if 2 v_x : " << v_x << std::endl;
        acc = V_MAX_x/T ;
        v_x = v_x_0 - acc * (time-1);
        acc = V_MAX_y/T ;
        v_y = v_y_0 - acc * (time-1);
      } else if((time > (T+1)) && (time < (2*T+1))){
        // std::cout << "if 3 v_x : " << v_x << std::endl;
        v_x = v_x_0 - V_MAX_x;
        v_y = v_y_0 - V_MAX_y;
      } else if((time > (2*T+1)) && (time < (3*T+1))){
        // std::cout << "if 4 v_x : " << v_x << std::endl;
        acc = -V_MAX_x/T;
        v_x = -(V_MAX_x-v_x_0 + acc * ((time-1) - 2*T));
        acc = -V_MAX_y/T;
        v_y = -(V_MAX_y-v_y_0 + acc * ((time-1) - 2*T));
      } else if((time > (3*T+1)) && (time < (3*T+1+1))){
        // std::cout << "if 5 v_x : " << v_x << std::endl;
        v_x = v_x_0 + 0.02*(time-3*T-1);
        v_y = (translation_y/translation_x) * v_x;
      } else if((time >= (3*T+1+1)) && (time < (3*T+1+2))){
        // std::cout << "if 6 v_x : " << v_x << std::endl;
        v_x = 0.0;//0001;
        v_y = (translation_y/translation_x) * v_x;
      }
      } else if ((slip_flag_new == true) && (time < T_slip)){
        // std::cout << "if 7 v_x : " << v_x << std::endl;
      //   // v_x = slip_onset_v_x - acc_slip * (time - slip_onset_time);
      //   // v_y = slip_onset_v_y - (translation_y/translation_x)*acc_slip*(time - slip_onset_time);
        if (optimal_v_x != 0){
          v_x = v_x + (optimal_v_x - v_x)/200;
          v_y = v_y + (optimal_v_y - v_y)/200; 
        } 
       
      }
      
      if (slip_flag_new == true){
          sub.shutdown();
      }

      v_x_save.push_back(v_x);
      v_y_save.push_back(v_y);
    
      slip_flag_old = slip_flag_new;
      ros::spinOnce();
      ++count;
      
      if ((rob.O_T_EE[12] - grasp_x) < -0.28){
        if (task_ending_phase == false){
          task_ending_v_x = v_x;
          task_ending_time = time;
          task_ending_acc = task_ending_v_x * task_ending_v_x / 0.218;
          task_ending_phase = true;
          std::cout << "v_x: " << task_ending_v_x << std::endl;
          std::cout << "acc: " << task_ending_acc << std::endl;
        } else {
          v_x = task_ending_v_x + task_ending_acc * (time - task_ending_time);
          v_y = (translation_y/translation_x) * v_x;
          std::cout << "end phase: " << v_x << std::endl;
        }
        
      }

      if (v_x > 0.0){
        v_x = -0.03;
        v_y = (translation_y/translation_x) * (-0.03);
      }
      
      franka::CartesianVelocities output = {{v_x, v_y, 0, 0, 0, 0}}; //Desired Cartesian velocity w.r.t. O-frame (base) {dx, dy, dz in [m/s], omegax, omegay, omegaz in [rad/s]
     
      if ((slip_onset_time==0 && time > 3*T+1+2) || (slip_onset_time>0 && time > T_slip) || ((rob.O_T_EE[12] - grasp_x) < -0.42) || optimal_v_x > 0) {
        std::cout << "task completion time: " << time << std::endl;
        int b = 0;
        unsigned int milisecond = 1000;
        std_msgs::Float64MultiArray robot_pose_msg;
        robot_pose_msg.data.clear();
        robot_pose_msg.data.resize(20);
        //EE_pose_vec.push_back(1.0);
        int n = EE_pose_vec.size();
        EE_pose_vec[n-1] = 1.0;
        robot_pose_msg.data = EE_pose_vec ;
        
        while(b<5000){  // 5 seconds - ish   
          usleep(milisecond);
          b++;
          robotPose_pub.publish(robot_pose_msg);
          ros::spinOnce();
        }

        std::ofstream RobotPoseCSV;   
        RobotPoseCSV.open(filename);
        RobotPoseCSV << "v_x" << "," << "v_y" << std::endl;
        for (int i = 0; i < v_x_save.size(); i++)
        {
          // std::cout << "in the loop" << std::endl;
          RobotPoseCSV << v_x_save[i] << "," << v_y_save[i] << std::endl;
        }
        return franka::MotionFinished(output);
      }
      return output;
    });

  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
