#pragma once

#include <boost/shared_ptr.hpp>

#include <baxter_core_msgs/JointCommand.h>
#include <ros/ros.h>

#include "robot_sim/robot.h"

namespace robot_sim {

class PositionController
{
 public:
 PositionController(boost::shared_ptr<Robot> robot) : 
  root_nh_(""), robot_(robot)
  {}

  bool init();

 private:
  ros::NodeHandle root_nh_;
  boost::shared_ptr<Robot> robot_;
  ros::Subscriber command_sub_;
  
  void callback(const baxter_core_msgs::JointCommand::ConstPtr&);
};

} //namespace robot_sim
