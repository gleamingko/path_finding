#include "robot_sim/position_controller.h"

namespace robot_sim {

bool PositionController::init()
{
  command_sub_ = root_nh_.subscribe("/robot/limb/left/joint_command", 10, &PositionController::callback, this);
  return true;
}

void PositionController::callback(const baxter_core_msgs::JointCommand::ConstPtr &msg)
{
  robot_->setTargetValues(msg->names, msg->command);
}

}
